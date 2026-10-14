//! Unified discrete-event engine. Pure state machine: knows about pools,
//! workers, the event heap, KV bookkeeping and time. Knows nothing about
//! request generation, metrics, real wall-clock, or how to render progress.
//! Drivers (batch [`super::simulator::Simulator`], serve `crate::serve::engine`)
//! pump it by alternating `next_event_time` / `submit` / `step`.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use super::spec::{DepthSample, DraftPlan, PlanCosts, SpecPlanner};
use crate::compute::ComputeEngine;
use crate::config::{
    ClusterSpec, DisaggTopology, ModelSpec, RouterConfig, SchedulerConfig, SourcePolicy,
    SpeculativeConfig,
};
use crate::kv_cache::{
    KVCacheManager, KvEvent, MemoryGraph, Owner, PrefixCacheStats, SharedMemoryGraph, SharedRadix,
};
use crate::request::{Request, SessionStep};
pub use crate::router::WorkerId;
use crate::router::{build_router, PrefixSignal, Router, RouterStats, WorkerSignal};
use crate::scheduler::{RecomputeFn, Scheduler, StepPlan};

pub type PoolId = usize;

/// Per-request timing breakdown produced when a request completes.
#[derive(Debug, Clone)]
pub struct RequestTiming {
    pub request_id: String,
    pub arrival_time: f64,
    /// Time the request's prefill phase finished on the prefill worker. For
    /// aggregated topologies, falls back to `first_token_time`.
    pub prefill_done_time: f64,
    /// Time the KV hand-off transfer completed and the request entered the
    /// decode worker. Equal to `prefill_done_time` for aggregated mode.
    pub handoff_done_time: f64,
    /// Time the first output token reached the client (= TTFT relative to
    /// arrival): the end of the pass that completed the prompt, or on a
    /// disaggregated topology the end of the KV hand-off, which carries it.
    pub first_token_time: f64,
    pub completion_time: f64,
    pub num_prompt_tokens: u32,
    pub num_output_tokens: u32,
    /// Prompt tokens served from the prefix cache at admission.
    pub num_cached_tokens: u32,
    /// Prompt-prefix tokens already resident on the selected decoder when
    /// the hand-off began (`None` in aggregated mode).
    pub decode_cached_tokens: Option<u32>,
    /// Session workloads: which step of which session this was.
    pub session: Option<Box<SessionStep>>,
    /// Memory-graph id of the worker that served the request.
    pub worker: Option<u32>,
    /// Times the request was preempted (and recomputed) before completing.
    pub num_preemptions: u32,
    /// Refused at submission (context larger than the worker's KV cache).
    pub rejected: bool,
    /// The prefix lookup that fixed `num_cached_tokens` (see
    /// [`crate::request::LookupRecord`]).
    pub lookup: Option<crate::request::LookupRecord>,
}

impl RequestTiming {
    pub fn ttft(&self) -> f64 {
        self.first_token_time - self.arrival_time
    }
    pub fn e2e(&self) -> f64 {
        self.completion_time - self.arrival_time
    }
    /// Mean time per output token, computed across the decode phase only
    /// (token 1 onwards).
    pub fn tpot(&self) -> Option<f64> {
        if self.num_output_tokens <= 1 {
            return None;
        }
        let decode_span = self.completion_time - self.first_token_time;
        Some(decode_span / (self.num_output_tokens - 1) as f64)
    }
}

/// One worker = one scheduler + one compute engine + (transitively) one KV
/// cache manager.
pub(crate) struct Worker {
    pub scheduler: Scheduler,
    pub compute_engine: ComputeEngine,
    /// This worker's id in the topology's memory graph (pools numbered in
    /// order: prefill workers, then decode workers; a DP-attention replica
    /// is `ranks` consecutive ids; workers added at run time after every
    /// other). Its [`WorkerId`] is the same number.
    pub global_id: usize,
    /// DP-attention: this worker's rank within its replica (0 for a whole
    /// replica). The ranks of a replica step in lockstep as one group (see
    /// `WorkerPool::groups`).
    pub rank: usize,
    pub state: WorkerState,
    /// Whether the pool's router was told the worker joined (so must be
    /// told it left).
    joined: bool,
    /// A `WorkerRetire` event is pending for this draining worker.
    retiring: bool,
    /// Requests whose first token the worker's last iteration produced,
    /// reported to the router when that iteration ends, and that end.
    prefilled: Vec<String>,
    prefilled_at: f64,
    /// Under an overlap scheduler, the requests whose first token the last
    /// iteration produced: the client sees it when the next one ends.
    overlap_first_tokens: Vec<String>,
    /// A DP replica leader's prefill-delayer count: passes run while some
    /// ranks had prefill to start and others did not.
    delay_passes: u32,
    /// A DP replica leader's passes left in a long prompt's decode interval
    /// (`SchedulerConfig::long_prefill_decode_interval`): no rank prefills.
    defer_passes: u32,
}

impl Worker {
    /// `graph` is the topology's KV memory graph and `global_id` this
    /// worker's id in it; the manager is attached to the graph only when
    /// the worker has tiers there. A DP-attention rank gets `1 / ranks` of
    /// the replica's KV capacity and the replica-wide compute engine.
    pub fn new(
        cluster: &ClusterSpec,
        model: ModelSpec,
        scheduler_config: SchedulerConfig,
        graph: &SharedMemoryGraph,
        global_id: usize,
        rank: usize,
        ranks: usize,
    ) -> Result<Self, String> {
        let kv_capacity = cluster
            .kv_cache_capacity(&scheduler_config, cluster.resident_weight_bytes(&model))
            / ranks.max(1) as u64;

        // Blocks are the model's content KV per block of tokens (the part
        // that grows for life: full-context layers, compressed history);
        // sliding windows and per-sequence state ride in auxiliary blocks.
        let policies = cluster.memory.policies();
        if let Some(bs) = scheduler_config.balance_set {
            bs.validate()?;
        }
        if !(scheduler_config.idle_pass_s.is_finite() && scheduler_config.idle_pass_s >= 0.0) {
            return Err(format!(
                "[scheduler] idle_pass_s must be >= 0, got {}",
                scheduler_config.idle_pass_s
            ));
        }
        let mut kv_cache_manager =
            KVCacheManager::for_model(kv_capacity, scheduler_config.block_size, &model, true)
                .with_hbm_eviction(policies.hbm_eviction);
        if graph.lock().unwrap().num_tiers(global_id) > 0 {
            kv_cache_manager = kv_cache_manager.with_memory(graph.clone(), global_id);
        }
        if kv_cache_manager.total_blocks() == 0 {
            return Err(format!(
                "KV cache capacity ({} bytes) holds less than one {}-token block ({} bytes) of {}",
                kv_capacity,
                scheduler_config.block_size,
                kv_cache_manager.bytes_per_block(),
                model.name
            ));
        }

        let compute_engine = cluster
            .compute_engine(model.clone())
            .with_cascade_attention(
                scheduler_config.enable_cascade_attention,
                scheduler_config.block_size,
            );
        // `source = min_time` prices recomputing a tier-held prefix at the
        // worker's own step cost: a second engine over the same hardware.
        let recompute: Option<RecomputeFn> = match policies.source {
            SourcePolicy::Promote {} => None,
            SourcePolicy::MinTime {} => {
                let pricer = cluster.compute_engine(model);
                Some(Box::new(move |req: &Request, from: u32, tokens: u32| {
                    let mut probe = req.clone();
                    probe.num_computed_tokens = from;
                    probe.kv_blocks.clear();
                    pricer.calculate_iteration_time(&[&probe], &[tokens])
                }))
            }
        };
        let scheduler = Scheduler::new(scheduler_config.clone(), kv_cache_manager)
            .with_source(policies.source, recompute)
            .with_prefetch(policies.prefetch)
            .with_hicache(
                policies.promote_fill,
                policies.storage_prefetch,
                policies.load_overlap,
            );
        Ok(Self {
            scheduler,
            compute_engine,
            global_id,
            rank,
            state: WorkerState::Active,
            joined: true,
            retiring: false,
            prefilled: Vec::new(),
            prefilled_at: 0.0,
            overlap_first_tokens: Vec::new(),
            delay_passes: 0,
            defer_passes: 0,
        })
    }

    pub fn id(&self) -> WorkerId {
        WorkerId(self.global_id as u32)
    }

    /// An empty rank need not take the shared graph lock just to discover
    /// that [`Scheduler::schedule`] has nothing to do. A queued transfer
    /// completion is graph-owned state and therefore keeps the rank live.
    fn can_skip_schedule(&self, now: f64, completed_owners: &[Owner]) -> bool {
        !self.scheduler.has_local_work_for_schedule(now)
            && !completed_owners.contains(&Owner::Worker(self.global_id))
    }
}

/// What a pool's workers are built from, kept to build more at run time.
#[derive(Clone)]
struct PoolSpec {
    cluster: ClusterSpec,
    scheduler: SchedulerConfig,
}

pub(crate) struct WorkerPool {
    /// Every worker the pool has had, by slot: the order they were added,
    /// so a fixed topology's replica index. A removed worker leaves `None`;
    /// slots are never reused.
    slots: Vec<Option<Worker>>,
    /// Slots taking new work (active), ascending.
    routable: Vec<usize>,
    /// Picks the worker each request enters. Round-robin unless
    /// [`Topology::with_router`] set another. Routes flat over every
    /// active worker, so on a DP-attention pool it places requests on
    /// ranks.
    router: Box<dyn Router>,
    router_stats: RouterStats,
    /// Tier-attached managers in a pool share one radix. Keep that handle
    /// and the routable workers' radix-local ids so a KV-aware routing
    /// decision resolves the prompt path once, then reads each worker's
    /// view. Multi-worker HBM-only pools retain private trees and use the
    /// per-manager fallback.
    routing_radix: Option<SharedRadix>,
    worker_ids: Vec<usize>,
    /// Lockstep groups: the ranks of one replica, stepped together as one
    /// iteration (they meet at every layer's FFN collective). Each group
    /// lists its members' slots; the first is the leader that carries the
    /// group's events.
    groups: Vec<Vec<usize>>,
    group_of: Vec<usize>,
    spec: PoolSpec,
    /// Where the pool's newest worker sits in the memory graph, and how
    /// many of the pool's workers that node holds: runtime additions pack
    /// onto it until it is full, as the build packs the fixed ones.
    last_node: (usize, usize),
}

impl WorkerPool {
    fn new(workers: Vec<Worker>, spec: PoolSpec, memory: &MemoryGraph) -> Self {
        let n = workers.len();
        let last_node = workers.last().map_or((memory.next_node(), 0), |last| {
            let node = memory.node_of(last.global_id);
            let on_node = workers
                .iter()
                .filter(|w| memory.node_of(w.global_id) == node)
                .count();
            (node, on_node)
        });
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut group_of = Vec::with_capacity(n);
        for (i, w) in workers.iter().enumerate() {
            if w.rank == 0 || groups.is_empty() {
                groups.push(Vec::new());
            }
            group_of.push(groups.len() - 1);
            groups.last_mut().unwrap().push(i);
        }
        let mut pool = Self {
            slots: workers.into_iter().map(Some).collect(),
            routable: Vec::new(),
            router: build_router(&RouterConfig::RoundRobin {}),
            router_stats: RouterStats::new(n),
            routing_radix: None,
            worker_ids: Vec::new(),
            groups,
            group_of,
            spec,
            last_node,
        };
        pool.refresh_routing();
        pool
    }

    /// The live worker in `slot`.
    fn worker(&self, slot: usize) -> &Worker {
        self.slots[slot].as_ref().expect("a live worker")
    }

    fn worker_mut(&mut self, slot: usize) -> &mut Worker {
        self.slots[slot].as_mut().expect("a live worker")
    }

    fn is_live(&self, slot: usize) -> bool {
        self.slots.get(slot).is_some_and(Option::is_some)
    }

    /// Live workers, in slot order.
    fn live(&self) -> impl Iterator<Item = &Worker> {
        self.slots.iter().flatten()
    }

    fn live_mut(&mut self) -> impl Iterator<Item = &mut Worker> {
        self.slots.iter_mut().flatten()
    }

    /// The worker that carries `worker`'s group's events.
    fn leader_of(&self, worker: usize) -> usize {
        self.groups[self.group_of[worker]][0]
    }

    /// The workers stepped together with `worker` (itself included).
    fn members_of(&self, worker: usize) -> &[usize] {
        &self.groups[self.group_of[worker]]
    }

    /// The live replicas' first ranks, in slot order.
    fn replicas(&self) -> impl Iterator<Item = &Worker> {
        self.groups.iter().filter_map(|g| self.slots[g[0]].as_ref())
    }

    /// Install `router` at `now`: it hears of every replica already in the
    /// pool, and the workers record KV events if it wants them.
    fn set_router(&mut self, mut router: Box<dyn Router>, now: f64) {
        let kv_events = router.wants_kv_events();
        for w in self.live_mut() {
            w.scheduler.kv_cache_manager_mut().set_kv_events(kv_events);
        }
        for group in &self.groups {
            if let Some(w) = self.slots[group[0]].as_ref().filter(|w| w.joined) {
                router.on_worker_added(w.id(), group.len() as u32, now);
            }
        }
        self.router = router;
        self.router_stats = RouterStats::new(self.slots.len());
    }

    /// Recompute the routable set and the shared routing radix after a
    /// worker joined or stopped taking work.
    fn refresh_routing(&mut self) {
        self.routable = (0..self.slots.len())
            .filter(|&s| {
                self.slots[s]
                    .as_ref()
                    .is_some_and(|w| w.state == WorkerState::Active)
            })
            .collect();
        let slots = &self.slots;
        let managers: Vec<&KVCacheManager> = self
            .routable
            .iter()
            .map(|&s| {
                slots[s]
                    .as_ref()
                    .expect("a live worker")
                    .scheduler
                    .kv_cache_manager()
            })
            .collect();
        let worker_ids = managers.iter().map(|m| m.radix_worker()).collect();
        let routing_radix = managers.first().map(|m| m.radix().clone()).filter(|radix| {
            managers
                .iter()
                .all(|m| std::sync::Arc::ptr_eq(radix, m.radix()))
        });
        self.worker_ids = worker_ids;
        self.routing_radix = routing_radix;
    }

    /// Ask the router where `req` goes at `now`: the slot of the chosen
    /// worker, or `None` when no worker takes new work. Builds one signal
    /// per routable worker; the per-worker prefix estimate is only computed
    /// for routers that ask. Tells the router where the request went.
    fn pick(&mut self, req: &Request, now: f64) -> Result<Option<usize>, String> {
        if self.routable.is_empty() {
            return Ok(None);
        }
        let prefix = self.router.prefix_signal();
        let hashes = &req.prompt_block_hashes;
        let routable = &self.routable;
        let slots = &self.slots;
        let manager = |i: usize| {
            slots[routable[i]]
                .as_ref()
                .expect("a live worker")
                .scheduler
                .kv_cache_manager()
        };
        let shared_prefix_tokens: Option<Vec<u32>> = match prefix {
            PrefixSignal::None => None,
            _ if hashes.is_empty() => Some(vec![0; routable.len()]),
            PrefixSignal::Cached => self.routing_radix.as_ref().map(|radix| {
                radix
                    .lock()
                    .unwrap()
                    .lookup_workers(&self.worker_ids, hashes)
                    .into_iter()
                    .enumerate()
                    .map(|(i, lk)| lk.cached() * manager(i).block_size())
                    .collect()
            }),
            PrefixSignal::Resident => self.routing_radix.as_ref().map(|radix| {
                radix
                    .lock()
                    .unwrap()
                    .resident_prefix_workers(&self.worker_ids, hashes)
                    .into_iter()
                    .enumerate()
                    .map(|(i, blocks)| blocks * manager(i).block_size())
                    .collect()
            }),
        };
        let (groups, group_of) = (&self.groups, &self.group_of);
        let signals: Vec<WorkerSignal> = routable
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let w = slots[s].as_ref().expect("a live worker");
                let replica = slots[groups[group_of[s]][0]]
                    .as_ref()
                    .expect("a live replica")
                    .id();
                let sched = &w.scheduler;
                let mgr = sched.kv_cache_manager();
                WorkerSignal {
                    worker: w.id(),
                    replica,
                    rank: w.rank as u32,
                    running: sched.num_running(),
                    waiting: sched.num_waiting(),
                    queued_prefill_tokens: sched.queued_prefill_tokens(),
                    kv_util: mgr.utilization(),
                    free_kv_tokens: mgr.num_free_blocks() as u64 * mgr.block_size() as u64,
                    cached_prefix_tokens: shared_prefix_tokens.as_ref().map_or_else(
                        || match prefix {
                            PrefixSignal::None => None,
                            PrefixSignal::Cached => Some(mgr.cached_prefix_tokens_estimate(hashes)),
                            PrefixSignal::Resident => Some(mgr.hbm_prefix_tokens(hashes)),
                        },
                        |tokens| Some(tokens[i]),
                    ),
                }
            })
            .collect();
        let chosen = self.router.route(req, &signals, now);
        let Some(i) = signals.iter().position(|s| s.worker == chosen) else {
            return Err(format!(
                "router chose {chosen} for {}, which is not among the workers it was offered",
                req.request_id
            ));
        };
        let slot = self.routable[i];
        self.router_stats.record(&signals, i, slot);
        self.router.on_routed(req, chosen, now);
        Ok(Some(slot))
    }
}

#[derive(Debug, Clone, Copy)]
enum PoolRole {
    Aggregated,
    DisaggPrefill,
    DisaggDecode,
}

pub(crate) enum Roles {
    Aggregated { pool: PoolId },
    Disagg { prefill: PoolId, decode: PoolId },
}

pub struct Topology {
    pub(crate) pools: Vec<WorkerPool>,
    pub(crate) roles: Roles,
    /// The KV memory graph every worker's tiers and every hand-off move
    /// over.
    pub(crate) memory: SharedMemoryGraph,
    model: ModelSpec,
    /// Pool and slot of every live worker.
    slot_of: HashMap<WorkerId, (PoolId, usize)>,
}

impl Topology {
    pub fn aggregated(
        cluster: ClusterSpec,
        model: ModelSpec,
        scheduler_config: SchedulerConfig,
    ) -> Result<Self, String> {
        let memory = MemoryGraph::build(
            &[&cluster],
            scheduler_config.block_size,
            KVCacheManager::content_curve(&model, scheduler_config.block_size),
            None,
        )?
        .shared_handle();
        let workers = Self::build_pool(&cluster, &model, &scheduler_config, &memory, 0)?;
        let spec = PoolSpec {
            cluster,
            scheduler: scheduler_config,
        };
        let pool = WorkerPool::new(workers, spec, &memory.lock().unwrap());
        Ok(Self::new(
            vec![pool],
            Roles::Aggregated { pool: 0 },
            memory,
            model,
        ))
    }

    fn new(
        pools: Vec<WorkerPool>,
        roles: Roles,
        memory: SharedMemoryGraph,
        model: ModelSpec,
    ) -> Self {
        let slot_of = pools
            .iter()
            .enumerate()
            .flat_map(|(p, pool)| {
                pool.slots
                    .iter()
                    .enumerate()
                    .filter_map(move |(s, w)| w.as_ref().map(|w| (w.id(), (p, s))))
            })
            .collect();
        Self {
            pools,
            roles,
            memory,
            model,
            slot_of,
        }
    }

    /// The workers of one pool, numbered `first_id..` in the memory graph.
    fn build_pool(
        cluster: &ClusterSpec,
        model: &ModelSpec,
        scheduler_config: &SchedulerConfig,
        memory: &SharedMemoryGraph,
        first_id: usize,
    ) -> Result<Vec<Worker>, String> {
        cluster.step_cost.validate()?;
        let n = cluster.num_workers.max(1) as usize;
        let ranks = cluster.kv_ranks() as usize;
        let mut workers = Vec::with_capacity(n * ranks);
        for w in 0..n {
            for r in 0..ranks {
                workers.push(Worker::new(
                    cluster,
                    model.clone(),
                    scheduler_config.clone(),
                    memory,
                    first_id + w * ranks + r,
                    r,
                    ranks,
                )?);
            }
        }
        Ok(workers)
    }

    pub fn from_disagg(
        topology: &DisaggTopology,
        model: ModelSpec,
        scheduler_config: SchedulerConfig,
    ) -> Result<Self, String> {
        let memory = MemoryGraph::build(
            &[&topology.prefill, &topology.decode],
            scheduler_config.block_size,
            KVCacheManager::content_curve(&model, scheduler_config.block_size),
            topology.kv_link_bw,
        )?
        .shared_handle();
        let p_count = topology.prefill.graph_workers().0;
        let p_workers = Self::build_pool(&topology.prefill, &model, &scheduler_config, &memory, 0)?;
        let d_workers = Self::build_pool(
            &topology.decode,
            &model,
            &scheduler_config,
            &memory,
            p_count,
        )?;
        // Every hand-off route must exist up front (a missing NIC template
        // and no `kv_link_bw` is a config error, not a run-time one).
        let pools = {
            let mut g = memory.lock().unwrap();
            let d_count = topology.decode.graph_workers().0;
            for p in 0..p_count {
                for d in 0..d_count {
                    g.handoff_path(p, p_count + d)?;
                }
            }
            let spec = |cluster: &ClusterSpec| PoolSpec {
                cluster: cluster.clone(),
                scheduler: scheduler_config.clone(),
            };
            vec![
                WorkerPool::new(p_workers, spec(&topology.prefill), &g),
                WorkerPool::new(d_workers, spec(&topology.decode), &g),
            ]
        };
        Ok(Self::new(
            pools,
            Roles::Disagg {
                prefill: 0,
                decode: 1,
            },
            memory,
            model,
        ))
    }

    /// The topology's memory graph.
    pub fn memory(&self) -> &SharedMemoryGraph {
        &self.memory
    }

    /// Front every pool with the router `cfg` names. On a disaggregated
    /// topology the same policy routes arrivals into the prefill pool and
    /// hand-offs into the decode pool; see `with_routers` to split them.
    pub fn with_router(mut self, cfg: &RouterConfig) -> Self {
        for pool in &mut self.pools {
            pool.set_router(build_router(cfg), 0.0);
        }
        self
    }

    /// Front the pool arrivals enter with `entry` and, on a disaggregated
    /// topology, the decode pool with `decode`.
    pub fn with_routers(mut self, entry: &RouterConfig, decode: &RouterConfig) -> Self {
        let entry_pool = self.entry_pool();
        for (i, pool) in self.pools.iter_mut().enumerate() {
            pool.set_router(
                build_router(if i == entry_pool { entry } else { decode }),
                0.0,
            );
        }
        self
    }

    /// Front `pool` with a caller's own router: any [`Router`], such as one
    /// that keeps its own view of the pool from the engine's notifications.
    /// It hears of the pool's workers at once (`on_worker_added` at t = 0).
    /// Panics if the topology has no pool `pool`.
    pub fn with_pool_router(mut self, pool: PoolId, router: Box<dyn Router>) -> Self {
        self.pools[pool].set_router(router, 0.0);
        self
    }

    /// Number of pools: one aggregated, or prefill then decode.
    pub fn num_pools(&self) -> usize {
        self.pools.len()
    }

    /// The pool arrivals enter: the aggregated pool, or the prefill pool.
    pub fn entry_pool(&self) -> PoolId {
        match self.roles {
            Roles::Aggregated { pool } => pool,
            Roles::Disagg { prefill, .. } => prefill,
        }
    }

    /// The decode pool of a disaggregated topology.
    pub fn decode_pool(&self) -> Option<PoolId> {
        match self.roles {
            Roles::Aggregated { .. } => None,
            Roles::Disagg { decode, .. } => Some(decode),
        }
    }

    fn role_for_pool(&self, pool: PoolId) -> PoolRole {
        match self.roles {
            Roles::Aggregated { .. } => PoolRole::Aggregated,
            Roles::Disagg {
                prefill, decode, ..
            } => {
                if pool == prefill {
                    PoolRole::DisaggPrefill
                } else if pool == decode {
                    PoolRole::DisaggDecode
                } else {
                    PoolRole::Aggregated
                }
            }
        }
    }
}

#[derive(Debug)]
enum EventKind {
    Arrival(Box<Request>),
    WorkerReady {
        pool: PoolId,
        worker: usize,
    },
    /// The next transfer on the memory graph (a hand-off or a tier
    /// promotion) is due to complete under the rates in force when it was
    /// scheduled. `generation` invalidates events made stale by a later
    /// change to the in-flight set.
    FlowDrain {
        generation: u64,
    },
    /// A request's prefill finishes at this instant on the P pool: pick its
    /// decode worker and put its KV on the hand-off link. Its own event (not
    /// done inside the iteration that produced it) so the link is advanced
    /// to a time no later than any transfer's pending drain event. The
    /// request waits in `Engine::queued_handoffs`.
    HandoffStart(String),
    /// A worker's next prefetch plan is due: wake it if idle. Its own
    /// event (not a `WorkerReady` armed ahead of time) so an idle worker
    /// still wakes for arrivals in between.
    PrefetchDue {
        pool: PoolId,
        worker: usize,
    },
    /// A started replica is ready: it joins routing. `worker` is its first
    /// rank's slot.
    WorkerJoin {
        pool: PoolId,
        worker: usize,
    },
    /// A draining worker holds nothing more: it leaves the pool.
    WorkerRetire {
        pool: PoolId,
        worker: usize,
    },
    /// A worker was killed (already removed, its requests failed): report
    /// it as a step.
    WorkerKilled {
        pool: PoolId,
        worker: WorkerId,
    },
}

#[derive(Debug)]
struct TimedEvent {
    time: f64,
    /// Pops before every ordinary event at its time: the report of a
    /// change the caller made between steps.
    first: bool,
    seq: u64,
    kind: EventKind,
}

impl PartialEq for TimedEvent {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.first == other.first && self.seq == other.seq
    }
}
impl Eq for TimedEvent {}
impl PartialOrd for TimedEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TimedEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is max-heap; reverse so smaller time pops first.
        other
            .time
            .partial_cmp(&self.time)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.first.cmp(&other.first))
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

/// Per-request progress yielded by a worker iteration.
#[derive(Debug, Clone)]
pub struct RequestProgress {
    pub request_id: String,
    /// Whether the request was in prefill phase BEFORE this iteration ran.
    pub was_prefill: bool,
    /// Positions resident before the iteration: with `num_tokens`, the
    /// context the iteration's positions attend (what a linear step cost
    /// charges a decoding request as attended KV tokens).
    pub context_tokens: u32,
    /// Positions computed for this request in the iteration: the prefill
    /// chunk, or the decode verify width (`1 + draft`).
    pub num_tokens: u32,
    /// Output tokens the iteration produced for this request: 1 when it
    /// completed prefill, `1 + accepted` for a decode step, 0 otherwise.
    pub num_output: u32,
}

/// Information about the iteration that ran during a `step` call. Present
/// when the popped event was a `WorkerReady` that resulted in scheduled work.
#[derive(Debug, Clone)]
pub struct IterationInfo {
    pub pool: PoolId,
    /// The worker's index in its pool: the order workers were added to it
    /// (a fixed topology's replica index). For DP-attention, the replica's
    /// leader.
    pub worker: usize,
    pub worker_id: WorkerId,
    pub start_time: f64,
    pub end_time: f64,
    pub iteration_time: f64,
    pub batch_size: usize,
    pub bandwidth_util: f64,
    pub flops_util: f64,
    pub progress: Vec<RequestProgress>,
}

/// What kind of event drove this step. Useful for drivers that want to filter
/// (e.g. throttle progress callbacks to iteration ends only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepKind {
    Arrival,
    Iteration,
    LinkComplete,
    HandoffStart,
    PrefetchDue,
    /// A replica added with [`Engine::add_worker`] became ready.
    WorkerJoined,
    /// A replica left its pool: drained, or killed.
    WorkerRemoved,
}

/// Where a replica (and so each of its ranks) is in its life in a pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    /// Added and not yet ready: its KV cache and compute exist, it takes
    /// no work and routers do not see it.
    Starting,
    /// Takes new requests.
    Active,
    /// Takes no new requests; finishes those it holds (and hand-offs in
    /// flight to or from it), then leaves the pool.
    Draining,
}

/// A change to a pool's membership, reported by the step that made it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerEvent {
    /// A started replica became ready and joined routing; `worker` names
    /// it (its first rank).
    Joined { pool: PoolId, worker: WorkerId },
    /// A replica left its pool with all its ranks: drained of its work, or
    /// killed.
    Removed {
        pool: PoolId,
        worker: WorkerId,
        killed: bool,
    },
}

/// Why the engine handed a request back unfinished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureReason {
    /// The worker holding it was killed — or, mid hand-off, the prefill
    /// worker its KV was leaving or the decode worker it was going to.
    WorkerKilled,
    /// The pool it had to enter had no worker taking new work.
    NoWorker,
}

/// A request the engine handed back unfinished.
#[derive(Debug, Clone)]
pub struct FailedRequest {
    /// The request, holding no KV anywhere (see [`Request::lose_worker`]):
    /// resubmit it to carry it on elsewhere, or count it.
    pub request: Request,
    /// The pool it was in, or had to enter.
    pub pool: PoolId,
    /// The worker (DP-attention rank) it was on or bound for; `None` if it
    /// never got one.
    pub worker: Option<WorkerId>,
    pub reason: FailureReason,
    pub time: f64,
}

#[derive(Debug)]
pub struct StepOutcome {
    pub time: f64,
    pub kind: StepKind,
    pub iteration: Option<IterationInfo>,
    pub completions: Vec<RequestTiming>,
    /// Requests handed back unfinished since the previous step.
    pub failed: Vec<FailedRequest>,
    /// The membership change this step made, if any.
    pub worker_event: Option<WorkerEvent>,
}

impl StepOutcome {
    fn new(time: f64, kind: StepKind) -> Self {
        Self {
            time,
            kind,
            iteration: None,
            completions: Vec::new(),
            failed: Vec::new(),
            worker_event: None,
        }
    }
}

pub struct Engine {
    topology: Topology,
    events: BinaryHeap<TimedEvent>,
    /// Requests whose prefill pass on the P pool is under way, to start
    /// their hand-off when it ends (their `HandoffStart`).
    queued_handoffs: HashMap<String, Request>,
    /// Requests that finished prefill on the P pool and are mid-handoff over
    /// the link, with the decode worker (slot) each was routed to when the
    /// transfer began.
    parked: HashMap<String, (Request, usize)>,
    /// Memory-graph id of the prefill worker each in-flight hand-off left.
    handoff_source: HashMap<String, usize>,
    /// Generation of the scheduled `FlowDrain` event and the time it is
    /// due; only the event carrying the current generation is acted on.
    flow_generation: u64,
    flow_due: Option<f64>,
    /// `worker_busy[pool][worker]` is true iff a `WorkerReady` for that
    /// worker is currently scheduled in the queue.
    worker_busy: Vec<Vec<bool>>,
    /// Per worker: the time of its armed `PrefetchDue` event, if any.
    prefetch_armed: Vec<Vec<Option<f64>>>,
    /// Hand-off transfers started, bytes moved, and bytes skipped because
    /// the chosen decoder already held the prompt prefix.
    handoff_stats: HandoffStats,
    /// Joint prefill/decode residency of reusable session prefixes, one
    /// counter set per decode worker slot.
    reusable_kv_by_decode_worker: Vec<ReusableKvStats>,
    /// Unique prompt positions computed by the prefill pool, partitioned by
    /// their role in the session turn.
    session_prefill_work: SessionPrefillWorkStats,
    /// Per-pool (∑ batch·dt, ∑ dt) for time-weighted mean batch size.
    pool_batch_acc: Vec<(f64, f64)>,
    current_time: f64,
    seq_counter: u64,
    /// Speculative decoding planner. `None` = no speculation.
    spec: Option<SpecPlanner>,
    /// Optional affine step-time correction (alpha, beta seconds).
    time_correction: Option<(f64, f64)>,
    /// Requests handed back unfinished, and requests that completed outside
    /// an iteration (on a worker killed as they finished), for the next
    /// step's outcome.
    failed: Vec<FailedRequest>,
    completed: Vec<RequestTiming>,
    /// Draining workers `(pool, slot)`, checked after every step for
    /// having nothing left.
    draining: Vec<(PoolId, usize)>,
    /// Buffers every iteration fills and empties, kept so a step allocates
    /// only what it hands back.
    scratch: IterationScratch,
}

/// Per-iteration working buffers (see [`Engine::run_iteration`]); empty
/// between steps.
#[derive(Default)]
struct IterationScratch {
    /// The stepping group's members, for [`Engine::handle_worker_ready`].
    group: Vec<usize>,
    /// The same, for [`Engine::run_iteration`].
    members: Vec<usize>,
    /// `(member, index into its running set, tokens)` per batch entry.
    entries: Vec<(usize, usize, u32)>,
    active_members: Vec<usize>,
    /// Verify width (decode) or chunk (prefill) per batch entry.
    cost_tokens: Vec<u32>,
    was_prefill: Vec<bool>,
    round_commits: Vec<Option<u32>>,
    accepted_extra: Vec<u32>,
    draft_widths: Vec<u32>,
    /// Decode requests per member, in member order.
    decode_counts: Vec<usize>,
    draft_plans: Vec<DraftPlan>,
    plans: Vec<(u32, Option<u32>)>,
}

impl Engine {
    pub fn new(topology: Topology) -> Self {
        let mut worker_busy = Vec::with_capacity(topology.pools.len());
        let mut prefetch_armed = Vec::with_capacity(topology.pools.len());
        for pool in &topology.pools {
            worker_busy.push(vec![false; pool.slots.len()]);
            prefetch_armed.push(vec![None; pool.slots.len()]);
        }
        let pool_batch_acc = vec![(0.0_f64, 0.0_f64); topology.pools.len()];
        let decode_workers = match topology.roles {
            Roles::Disagg { decode, .. } => topology.pools[decode].slots.len(),
            Roles::Aggregated { .. } => 0,
        };
        Self {
            topology,
            events: BinaryHeap::new(),
            queued_handoffs: HashMap::new(),
            parked: HashMap::new(),
            handoff_source: HashMap::new(),
            flow_generation: 0,
            flow_due: None,
            worker_busy,
            prefetch_armed,
            handoff_stats: HandoffStats::default(),
            reusable_kv_by_decode_worker: vec![ReusableKvStats::default(); decode_workers],
            session_prefill_work: SessionPrefillWorkStats::default(),
            pool_batch_acc,
            current_time: 0.0,
            seq_counter: 0,
            spec: None,
            time_correction: None,
            failed: Vec::new(),
            completed: Vec::new(),
            draining: Vec::new(),
            scratch: IterationScratch::default(),
        }
    }

    /// Apply an affine empirical correction to every roofline-priced
    /// iteration time: `t = alpha * t_model + beta`. `alpha` captures the
    /// kernel-efficiency gap to the roofline (dominant at large batch);
    /// `beta` captures fixed per-iteration overhead — scheduler, CPU, launch
    /// latency (dominant at small batch). Applied to the actual step time
    /// only, never to policy candidate pricing (so speculative width choices
    /// stay model-relative) and never on top of a measured step-cost table.
    /// Config: `[hardware.<name>] time_correction = { alpha, beta }`.
    pub fn set_time_correction(&mut self, alpha: f64, beta: f64) {
        self.time_correction = Some((alpha, beta));
    }

    /// Enable speculative decoding. Decode steps then verify `gamma + 1` tokens
    /// (cost) and advance by `accepted + 1` (progress) per the acceptance model.
    /// Under DP-attention each rank verifies its own requests' drafts within
    /// the replica's lockstep step, and the draft depths are planned for the
    /// replica's whole decode batch, as SGLang's DP-attention EAGLE runs.
    /// Loads the trace bank / measured cost table the config names; a bad
    /// path is returned as an error.
    pub fn enable_speculative(&mut self, cfg: SpeculativeConfig, seed: u64) -> Result<(), String> {
        self.spec = Some(SpecPlanner::new(cfg, seed)?);
        Ok(())
    }

    fn record_session_prefill_work(&mut self, req: &Request) {
        let Some(step) = req.session.as_ref() else {
            return;
        };
        let prompt = req.num_prompt_tokens;
        let shared = step.shared_tokens.min(prompt);
        let parent_prefill = step.shared_prefill_tokens.min(shared);
        let parent_decode_end = parent_prefill
            .saturating_add(step.shared_decode_tokens)
            .min(shared);
        let cached = req.num_cached_tokens.min(prompt);
        let cached_inherited = cached.min(shared);
        let amount = |start: u32, end: u32| KvAmountStats {
            tokens: end.saturating_sub(start) as u64,
            bytes: self
                .topology
                .model
                .kv_storage_bytes(end)
                .saturating_sub(self.topology.model.kv_storage_bytes(start)),
        };
        self.session_prefill_work += SessionPrefillWorkStats {
            new_prompt: amount(cached.max(shared), prompt),
            parent_prefill_recompute: amount(cached_inherited, parent_prefill),
            parent_decode_recompute: amount(
                cached_inherited.max(parent_prefill),
                parent_decode_end,
            ),
            unattributed_recompute: amount(cached_inherited.max(parent_decode_end), shared),
        };
    }

    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Push the clock forward to `at_time` if it's strictly larger. Useful for
    /// real-time drivers that want sim-time to track wall-time when there is
    /// no event to advance through.
    pub fn advance_to(&mut self, at_time: f64) {
        if at_time > self.current_time {
            self.current_time = at_time;
        }
    }

    pub fn next_event_time(&self) -> Option<f64> {
        self.events.peek().map(|e| e.time)
    }

    /// Submit a request for arrival at `req.arrival_time`. The request is
    /// enqueued as an `Arrival` event; it does not enter any worker until a
    /// `step()` call processes that event. If `arrival_time` is in the past
    /// (relative to the engine clock) the event fires immediately at `now`;
    /// the request keeps its own `arrival_time`, so TTFT is still computed
    /// against the intended emission moment.
    pub fn submit(&mut self, req: Request) {
        let when = req.arrival_time.max(self.current_time);
        self.push(when, EventKind::Arrival(Box::new(req)));
    }

    // ------------------------------------------------------------------
    // Pool membership

    /// Number of pools: one aggregated, or prefill then decode.
    pub fn num_pools(&self) -> usize {
        self.topology.num_pools()
    }

    /// The pool arrivals enter: the aggregated pool, or the prefill pool.
    pub fn entry_pool(&self) -> PoolId {
        self.topology.entry_pool()
    }

    /// The decode pool of a disaggregated topology.
    pub fn decode_pool(&self) -> Option<PoolId> {
        self.topology.decode_pool()
    }

    /// The replicas in `pool` and their states, in the order they were
    /// added to it; replicas that left are absent. A replica is one worker,
    /// or under DP-attention its rank group, named by its first rank's id
    /// (see [`Self::replica_ranks`]).
    pub fn pool_members(&self, pool: PoolId) -> Vec<(WorkerId, WorkerState)> {
        self.topology.pools.get(pool).map_or_else(Vec::new, |p| {
            p.replicas().map(|w| (w.id(), w.state)).collect()
        })
    }

    /// The ranks of the replica `worker` belongs to (`worker` may name any
    /// of them), in rank order: consecutive ids from the replica's own, one
    /// id for a replica without DP-attention. Each rank is routed to, and
    /// reported, by its own id. `None` once the replica has left.
    pub fn replica_ranks(&self, worker: WorkerId) -> Option<Vec<WorkerId>> {
        let &(pool, slot) = self.topology.slot_of.get(&worker)?;
        let p = &self.topology.pools[pool];
        Some(
            p.members_of(slot)
                .iter()
                .map(|&m| p.worker(m).id())
                .collect(),
        )
    }

    /// `worker`'s state (its replica's); `None` once it has left its pool.
    pub fn worker_state(&self, worker: WorkerId) -> Option<WorkerState> {
        let &(pool, slot) = self.topology.slot_of.get(&worker)?;
        Some(self.topology.pools[pool].worker(slot).state)
    }

    /// The sequence hash of every block `worker`'s HBM prefix cache holds,
    /// sorted: what a KV-event consumer's index of it should contain. Each
    /// DP-attention rank has its own. `None` once the worker has left.
    pub fn worker_cached_blocks(&self, worker: WorkerId) -> Option<Vec<u64>> {
        let &(pool, slot) = self.topology.slot_of.get(&worker)?;
        Some(
            self.topology.pools[pool]
                .worker(slot)
                .scheduler
                .kv_cache_manager()
                .resident_block_hashes(),
        )
    }

    /// Add a replica to `pool`, built as the pool's own (hardware, parallel
    /// layout, KV tiers, scheduler, step cost): one worker, or under
    /// DP-attention a group of `tp` one-GPU ranks stepping in lockstep. Its
    /// KV caches and compute exist from now, and it joins routing at
    /// `ready_at_s` (or now, if that has passed) — the `WorkerJoined` step.
    /// Until then it is [`WorkerState::Starting`]. Its GPUs go on the
    /// pool's newest node while that has free ones, then on fresh nodes.
    /// Returns the replica's id, its first rank's; the other ranks follow
    /// it consecutively.
    pub fn add_worker(&mut self, pool: PoolId, ready_at_s: f64) -> Result<WorkerId, String> {
        let p = self
            .topology
            .pools
            .get(pool)
            .ok_or_else(|| format!("no pool {pool}"))?;
        let spec = p.spec.clone();
        let ranks = spec.cluster.kv_ranks().max(1) as usize;
        let per_node = spec.cluster.graph_workers_per_node();
        let mut placement = p.last_node;
        // A disaggregated pool's ranks must reach, or be reached from,
        // every live rank of the other pool.
        let role = self.topology.role_for_pool(pool);
        let other = match (role, self.topology.decode_pool()) {
            (PoolRole::DisaggPrefill, Some(decode)) => Some(decode),
            (PoolRole::DisaggDecode, _) => Some(self.topology.entry_pool()),
            _ => None,
        };
        let others: Vec<usize> = other.map_or_else(Vec::new, |o| {
            self.topology.pools[o].live().map(|w| w.global_id).collect()
        });
        let gids = {
            let mut g = self.topology.memory.lock().unwrap();
            let mut gids = Vec::with_capacity(ranks);
            for _ in 0..ranks {
                let node = if placement.1 < per_node {
                    placement.0
                } else {
                    g.next_node()
                };
                let placed = g.add_worker(&spec.cluster, node).and_then(|gid| {
                    gids.push(gid);
                    others.iter().try_for_each(|&o| {
                        let (from, to) = match role {
                            PoolRole::DisaggPrefill => (gid, o),
                            _ => (o, gid),
                        };
                        g.handoff_path(from, to).map(|_| ())
                    })
                });
                if let Err(e) = placed {
                    for &gid in &gids {
                        g.retire_worker(gid);
                    }
                    return Err(e);
                }
                placement = if node == placement.0 {
                    (node, placement.1 + 1)
                } else {
                    (node, 1)
                };
            }
            gids
        };
        let mut workers = Vec::with_capacity(ranks);
        for (rank, &gid) in gids.iter().enumerate() {
            let built = Worker::new(
                &spec.cluster,
                self.topology.model.clone(),
                spec.scheduler.clone(),
                &self.topology.memory,
                gid,
                rank,
                ranks,
            );
            match built {
                Ok(mut w) => {
                    w.state = WorkerState::Starting;
                    w.joined = false;
                    workers.push(w);
                }
                Err(e) => {
                    let mut g = self.topology.memory.lock().unwrap();
                    for &gid in &gids {
                        g.retire_worker(gid);
                    }
                    return Err(e);
                }
            }
        }
        let id = workers[0].id();
        let p = &mut self.topology.pools[pool];
        let kv_events = p.router.wants_kv_events();
        let first = p.slots.len();
        let group = p.groups.len();
        for mut w in workers {
            w.scheduler.kv_cache_manager_mut().set_kv_events(kv_events);
            self.topology.slot_of.insert(w.id(), (pool, p.slots.len()));
            p.slots.push(Some(w));
            p.group_of.push(group);
            p.router_stats.per_worker.push(0);
            self.worker_busy[pool].push(false);
            self.prefetch_armed[pool].push(None);
            if matches!(role, PoolRole::DisaggDecode) {
                self.reusable_kv_by_decode_worker
                    .push(ReusableKvStats::default());
            }
        }
        p.groups.push((first..first + ranks).collect());
        p.last_node = placement;
        self.push(
            ready_at_s.max(self.current_time),
            EventKind::WorkerJoin {
                pool,
                worker: first,
            },
        );
        Ok(id)
    }

    /// Stop routing new requests to the replica `worker` names (any of its
    /// ranks) now. It finishes what it holds — running and queued requests
    /// on every rank, and hand-offs in flight to or from them — and then
    /// leaves the pool: a `WorkerRemoved` step, and
    /// [`Router::on_worker_removed`]. A starting replica leaves at once.
    /// Draining a draining replica does nothing.
    pub fn drain_worker(&mut self, pool: PoolId, worker: WorkerId) -> Result<(), String> {
        let leader = self.lifecycle_slot(pool, worker)?;
        let p = &mut self.topology.pools[pool];
        let was = p.worker(leader).state;
        if was == WorkerState::Draining {
            return Ok(());
        }
        for m in p.members_of(leader).to_vec() {
            p.worker_mut(m).state = WorkerState::Draining;
        }
        if was == WorkerState::Active {
            p.refresh_routing();
        }
        self.draining.push((pool, leader));
        self.check_draining();
        Ok(())
    }

    /// Remove the replica `worker` names (any of its ranks) now, as a crash
    /// would. Every request it held — in a rank's queues, in a pass still
    /// running, or in a hand-off to or from a rank — fails. The next step
    /// is the `WorkerRemoved` report, and its [`StepOutcome::failed`]
    /// carries them (reason [`FailureReason::WorkerKilled`], `worker` the
    /// rank), free to be resubmitted. The router hears
    /// [`Router::on_worker_removed`] at once, and the replica's pending
    /// events are dropped.
    pub fn kill_worker(&mut self, pool: PoolId, worker: WorkerId) -> Result<(), String> {
        let leader = self.lifecycle_slot(pool, worker)?;
        let now = self.current_time;
        let members = self.topology.pools[pool].members_of(leader).to_vec();
        let ranks: Vec<WorkerId> = members
            .iter()
            .map(|&m| self.topology.pools[pool].worker(m).id())
            .collect();
        for (&m, &rank) in members.iter().zip(&ranks) {
            // First tokens of an iteration that ended by now were produced.
            self.report_first_tokens(pool, m);
            let scheduler = &mut self.topology.pools[pool].worker_mut(m).scheduler;
            scheduler.abandon_transfers();
            let held = scheduler.take_all_requests();
            for mut req in held {
                // A request whose last pass ended by now finished (or was
                // refused) before the kill, its completion merely not yet
                // collected: it completes. The rest fail.
                let finished = req.is_finished() && req.last_pass.is_none_or(|(end, _)| end <= now);
                if req.rejected || finished {
                    req.last_pass = None;
                    self.topology.pools[pool].router.on_request_complete(
                        &req.request_id,
                        rank,
                        now,
                    );
                    let timing = self.finalise(req, now);
                    self.completed.push(timing);
                } else {
                    self.fail(req, pool, Some(rank), FailureReason::WorkerKilled);
                }
            }
        }
        // Hand-offs bound for its ranks: their prefill workers let the KV
        // go.
        if self.topology.decode_pool() == Some(pool) {
            let mut bound: Vec<(String, WorkerId)> = self
                .parked
                .iter()
                .filter_map(|(id, (_, s))| {
                    let i = members.iter().position(|m| m == s)?;
                    Some((id.clone(), ranks[i]))
                })
                .collect();
            bound.sort();
            for (id, rank) in bound {
                let (mut req, _) = self.parked.remove(&id).expect("parked");
                self.topology.memory.lock().unwrap().cancel_handoff(&id);
                if let Some(source) = self.handoff_source.remove(&id) {
                    if let Some((sp, si)) = self.locate_worker(source) {
                        self.release_handoff_source(sp, si, &mut req);
                    }
                }
                self.fail(req, pool, Some(rank), FailureReason::WorkerKilled);
            }
        }
        let from_ranks = |g: usize| ranks.contains(&WorkerId(g as u32));
        // Hand-offs queued on its prefill passes: those passes never end.
        let mut queued: Vec<(String, usize)> = self
            .handoff_source
            .iter()
            .filter(|&(id, &g)| from_ranks(g) && self.queued_handoffs.contains_key(id))
            .map(|(id, &g)| (id.clone(), g))
            .collect();
        queued.sort();
        for (id, g) in &queued {
            let req = self.queued_handoffs.remove(id).expect("queued");
            self.handoff_source.remove(id);
            self.fail(
                req,
                pool,
                Some(WorkerId(*g as u32)),
                FailureReason::WorkerKilled,
            );
        }
        // Hand-offs leaving its ranks: the KV they carry is gone.
        let mut leaving: Vec<(String, usize)> = self
            .handoff_source
            .iter()
            .filter(|&(id, &g)| from_ranks(g) && self.parked.contains_key(id))
            .map(|(id, &g)| (id.clone(), g))
            .collect();
        leaving.sort();
        for (id, g) in leaving {
            let (req, decode_slot) = self.parked.remove(&id).expect("parked");
            self.handoff_source.remove(&id);
            self.topology.memory.lock().unwrap().cancel_handoff(&id);
            if let Some(decode) = self.topology.decode_pool() {
                let d = &mut self.topology.pools[decode];
                if let Some(target) = d.slots[decode_slot].as_ref().map(Worker::id) {
                    d.router.on_request_complete(&id, target, now);
                }
            }
            self.fail(
                req,
                pool,
                Some(WorkerId(g as u32)),
                FailureReason::WorkerKilled,
            );
        }
        let queued: Vec<String> = queued.into_iter().map(|(id, _)| id).collect();
        let replica = self.remove_worker(pool, leader, &queued);
        self.push_first(EventKind::WorkerKilled {
            pool,
            worker: replica,
        });
        Ok(())
    }

    /// The slot of the first rank of the replica `worker` names, which must
    /// be live in `pool`.
    fn lifecycle_slot(&self, pool: PoolId, worker: WorkerId) -> Result<usize, String> {
        let &(p, slot) = self
            .topology
            .slot_of
            .get(&worker)
            .ok_or_else(|| format!("no worker {worker} in the topology"))?;
        if p != pool {
            return Err(format!("worker {worker} is in pool {p}, not pool {pool}"));
        }
        Ok(self.topology.pools[pool].leader_of(slot))
    }

    /// A started replica's ready time: it joins routing.
    fn join_worker(&mut self, pool: PoolId, leader: usize) -> Option<WorkerEvent> {
        let now = self.current_time;
        let p = &mut self.topology.pools[pool];
        if !p.is_live(leader) || p.worker(leader).state != WorkerState::Starting {
            return None;
        }
        let members = p.members_of(leader).to_vec();
        for &m in &members {
            let w = p.worker_mut(m);
            w.state = WorkerState::Active;
            w.joined = true;
        }
        let id = p.worker(leader).id();
        p.refresh_routing();
        p.router.on_worker_added(id, members.len() as u32, now);
        Some(WorkerEvent::Joined { pool, worker: id })
    }

    /// Whether the replica led by `(pool, leader)` holds nothing: no
    /// request on any rank, and no hand-off in flight to or from one.
    fn holds_nothing(&self, pool: PoolId, leader: usize) -> bool {
        let p = &self.topology.pools[pool];
        let decode = self.topology.decode_pool() == Some(pool);
        p.members_of(leader).iter().all(|&m| {
            let w = p.worker(m);
            let gid = w.global_id;
            w.scheduler.is_drained()
                && !(decode && self.parked.values().any(|&(_, s)| s == m))
                && !self.handoff_source.values().any(|&g| g == gid)
        })
    }

    /// Schedule the removal of every draining replica left with nothing.
    fn check_draining(&mut self) {
        let now = self.current_time;
        for i in 0..self.draining.len() {
            let (pool, slot) = self.draining[i];
            if !self.topology.pools[pool].worker(slot).retiring && self.holds_nothing(pool, slot) {
                self.topology.pools[pool].worker_mut(slot).retiring = true;
                self.push(now, EventKind::WorkerRetire { pool, worker: slot });
            }
        }
    }

    /// A draining replica's retirement is due: remove it if it still holds
    /// nothing.
    fn retire_drained_worker(&mut self, pool: PoolId, leader: usize) -> Option<WorkerEvent> {
        if !self.topology.pools[pool].is_live(leader) {
            return None;
        }
        if !self.holds_nothing(pool, leader) {
            self.topology.pools[pool].worker_mut(leader).retiring = false;
            return None;
        }
        let worker = self.remove_worker(pool, leader, &[]);
        Some(WorkerEvent::Removed {
            pool,
            worker,
            killed: false,
        })
    }

    /// Take the replica led by `(pool, leader)` out of the topology: its
    /// ranks' state is dropped, their HBM leaves the memory graph's tree,
    /// and the router hears the replica left (if it had heard it joined).
    /// Its pending events go too — and the hand-off starts of `handoffs` —
    /// so the clock never runs on for a replica that is gone. Its cancelled
    /// transfers may have moved the next drain, so the drain event is
    /// re-armed by the step's flow pump. Returns the replica's id.
    fn remove_worker(&mut self, pool: PoolId, leader: usize, handoffs: &[String]) -> WorkerId {
        let now = self.current_time;
        let p = &mut self.topology.pools[pool];
        let members = p.members_of(leader).to_vec();
        // Every rank lets go of its sources before any rank's HBM leaves.
        let mut removed = Vec::with_capacity(members.len());
        for &m in &members {
            let mut w = p.slots[m].take().expect("a live worker");
            w.scheduler.abandon_transfers();
            removed.push(w);
        }
        let id = removed[0].id();
        p.refresh_routing();
        if removed[0].joined {
            p.router.on_worker_removed(id, now);
        }
        let ids: Vec<WorkerId> = removed.iter().map(Worker::id).collect();
        drop(removed);
        {
            let mut g = self.topology.memory.lock().unwrap();
            for rank in &ids {
                g.retire_worker(rank.0 as usize);
            }
        }
        for rank in &ids {
            self.topology.slot_of.remove(rank);
        }
        for &m in &members {
            self.worker_busy[pool][m] = false;
            self.prefetch_armed[pool][m] = None;
        }
        self.draining.retain(|&d| d != (pool, leader));
        self.events.retain(|e| match &e.kind {
            EventKind::WorkerReady { pool: p, worker: w }
            | EventKind::PrefetchDue { pool: p, worker: w }
            | EventKind::WorkerJoin { pool: p, worker: w }
            | EventKind::WorkerRetire { pool: p, worker: w } => *p != pool || !members.contains(w),
            EventKind::HandoffStart(id) => !handoffs.contains(id),
            EventKind::FlowDrain { .. } => false,
            _ => true,
        });
        self.flow_due = None;
        id
    }

    fn push(&mut self, time: f64, kind: EventKind) {
        self.seq_counter += 1;
        self.events.push(TimedEvent {
            time,
            first: false,
            seq: self.seq_counter,
            kind,
        });
    }

    /// Push `kind` to be the next event processed.
    fn push_first(&mut self, kind: EventKind) {
        self.seq_counter += 1;
        self.events.push(TimedEvent {
            time: self.current_time,
            first: true,
            seq: self.seq_counter,
            kind,
        });
    }

    /// Time-weighted mean batch size per pool, across the wall-clock the
    /// engine has run so far. `None` for pools that never ran an iteration.
    pub fn pool_batch_means(&self) -> Vec<Option<f64>> {
        self.pool_batch_acc
            .iter()
            .map(|&(num, den)| if den > 0.0 { Some(num / den) } else { None })
            .collect()
    }

    /// Sum of `running` across all pools (for progress reporting). This
    /// counts requests currently held by any worker, including those mid-iter.
    pub fn aggregate_running(&self) -> usize {
        self.topology
            .pools
            .iter()
            .flat_map(|p| p.live())
            .map(|w| w.scheduler.num_running())
            .sum::<usize>()
            + self.parked.len()
    }

    /// One line per worker holding work, for a stall report: what is
    /// waiting, parked, and free.
    pub fn describe_stuck_workers(&self) -> String {
        let mut out = String::new();
        for (p, pool) in self.topology.pools.iter().enumerate() {
            for (i, w) in pool.slots.iter().enumerate() {
                let Some(w) = w else { continue };
                let s = &w.scheduler;
                if s.num_running() == 0 && s.num_waiting() == 0 {
                    continue;
                }
                let mgr = s.kv_cache_manager();
                let parked = s.pending_transfers();
                let first_parked = parked.first().map(|r| {
                    format!(
                        "{}(cached {}, ready_at {:?}, blocks {})",
                        r.request_id,
                        r.num_cached_tokens,
                        r.ready_at,
                        r.kv_blocks.len()
                    )
                });
                let first_waiting = s.waiting().front().map(|r| {
                    let l = mgr.peek_prefix_cache(r);
                    format!(
                        "{}(prompt {}, computed {}, blocks {}, needs {} blocks; lookup: cached {}, hbm {}, in_flight {}, promote {:?}, needs_staging {}, abandoned {}, ready_at {:?})",
                        r.request_id,
                        r.num_prompt_tokens,
                        r.num_computed_tokens,
                        r.kv_blocks.len(),
                        mgr.blocks_for_context(r.planned_positions()),
                        l.total_cached_tokens,
                        l.hbm_tokens,
                        l.in_flight_tokens,
                        l.promote_tokens_per_tier,
                        mgr.needs_staging(&l),
                        r.storage_prefetch_abandoned,
                        r.ready_at,
                    )
                });
                let staging = s.pending_storage();
                let first_staging = staging.first().map(|r| {
                    format!(
                        "{}(cached {}, ready_at {:?}, abandoned {}, lookups {:?})",
                        r.request_id,
                        r.num_cached_tokens,
                        r.ready_at,
                        r.storage_prefetch_abandoned,
                        r.lookup.as_ref().map(|l| l.lookups)
                    )
                });
                let (held, refs, free) = mgr.ref_summary();
                let by_queue = s.held_blocks_by_queue();
                out.push_str(&format!(
                    "\n  staging {}: [0] {:?}",
                    staging.len(),
                    first_staging
                ));
                out.push_str(&format!(
                    "\n  refs: {held} blocks referenced ({refs} refs), {free} free; held by (running, waiting, parked, prefetches) = {by_queue:?}"
                ));
                out.push_str(&format!(
                    "\n  pool {p} worker {i} (gid {}): running {}, waiting {}, parked {}, free {}/{} blocks, busy {}; parked[0] {:?}; waiting[0] {:?}",
                    w.global_id,
                    s.num_running(),
                    s.num_waiting(),
                    parked.len(),
                    mgr.num_free_blocks(),
                    mgr.total_blocks(),
                    self.worker_busy[p][pool.leader_of(i)],
                    first_parked,
                    first_waiting
                ));
            }
        }
        out
    }

    /// Total preemptions across all pools.
    pub fn aggregate_preemptions(&self) -> u64 {
        self.topology
            .pools
            .iter()
            .flat_map(|p| p.live())
            .map(|w| w.scheduler.num_preemptions())
            .sum()
    }

    /// Retune the per-worker concurrency cap across every pool: the
    /// serve-mode scale up/down. Admission reads the cap live, so this
    /// takes effect from each worker's next iteration and never evicts an
    /// already-running request (see [`Scheduler::set_max_num_seqs`]).
    pub fn set_max_num_seqs(&mut self, max_num_seqs: u32) {
        for pool in &mut self.topology.pools {
            pool.spec.scheduler.max_num_seqs = max_num_seqs;
            for worker in pool.live_mut() {
                worker.scheduler.set_max_num_seqs(max_num_seqs);
            }
        }
    }

    /// The concurrency cap of the first worker. Every worker in a topology
    /// is built from the same `SchedulerConfig`, and `set_max_num_seqs`
    /// writes all of them, so this is the topology-wide value.
    pub fn max_num_seqs(&self) -> u32 {
        self.topology
            .pools
            .iter()
            .flat_map(|p| p.live())
            .map(|w| w.scheduler.max_num_seqs())
            .next()
            .unwrap_or(0)
    }

    /// Sum of `waiting` across all pools.
    pub fn aggregate_waiting(&self) -> usize {
        self.topology
            .pools
            .iter()
            .flat_map(|p| p.live())
            .map(|w| w.scheduler.num_waiting())
            .sum()
    }

    /// Number of running requests currently in the prefill phase, summed
    /// across every worker. Used by drivers that report prefill/decode
    /// breakdown in their progress streams.
    pub fn aggregate_prefilling(&self) -> usize {
        self.topology
            .pools
            .iter()
            .flat_map(|p| p.live())
            .map(|w| {
                w.scheduler
                    .running()
                    .iter()
                    .filter(|r| r.is_prefill())
                    .count()
            })
            .sum()
    }

    /// Per-second speculative draft-depth series; empty when speculation is
    /// off.
    pub fn spec_depth_series(&self) -> Vec<DepthSample> {
        self.spec
            .as_ref()
            .map(|s| s.depth_series())
            .unwrap_or_default()
    }

    /// Routing statistics for the pool arrivals enter (the prefill pool on
    /// a disaggregated topology).
    pub fn router_stats(&self) -> &RouterStats {
        &self.topology.pools[self.topology.entry_pool()].router_stats
    }

    /// Routing statistics per pool, in pool order (prefill then decode on a
    /// disaggregated topology).
    pub fn pool_router_stats(&self) -> Vec<&RouterStats> {
        self.topology
            .pools
            .iter()
            .map(|p| &p.router_stats)
            .collect()
    }

    /// Routing statistics for the decode pool of a disaggregated topology.
    pub fn decode_router_stats(&self) -> Option<&RouterStats> {
        match self.topology.roles {
            Roles::Disagg { decode, .. } => Some(&self.topology.pools[decode].router_stats),
            Roles::Aggregated { .. } => None,
        }
    }

    /// Hand-off transfer totals (zero on an aggregated topology).
    pub fn handoff_stats(&self) -> HandoffStats {
        self.handoff_stats
    }

    pub fn handoffs_in_flight(&self) -> usize {
        self.parked.len()
    }

    /// Joint reusable-prefix residency totals and per-decode-worker detail.
    pub fn reusable_kv_stats(
        &self,
    ) -> Option<(
        ReusableKvStats,
        Vec<RankReusableKvStats>,
        SessionPrefillWorkStats,
    )> {
        let decode = match self.topology.roles {
            Roles::Disagg { decode, .. } => decode,
            Roles::Aggregated { .. } => return None,
        };
        // Totals cover every decode worker the run had; the detail, the
        // ones still in the pool.
        let mut total = ReusableKvStats::default();
        for stats in &self.reusable_kv_by_decode_worker {
            total += *stats;
        }
        let ranks = self.topology.pools[decode]
            .slots
            .iter()
            .zip(&self.reusable_kv_by_decode_worker)
            .filter_map(|(worker, stats)| {
                worker.as_ref().map(|worker| RankReusableKvStats {
                    worker: worker.global_id as u32,
                    rank: worker.rank as u32,
                    stats: *stats,
                })
            })
            .collect();
        Some((total, ranks, self.session_prefill_work))
    }

    /// HBM state and eviction accounting by topology pool and worker.
    pub fn hbm_pool_stats(&self) -> Vec<HbmPoolStats> {
        self.topology
            .pools
            .iter()
            .enumerate()
            .map(|(pool_id, pool)| {
                let role = match self.topology.role_for_pool(pool_id) {
                    PoolRole::Aggregated => "aggregated",
                    PoolRole::DisaggPrefill => "prefill",
                    PoolRole::DisaggDecode => "decode",
                }
                .to_string();
                let workers = pool
                    .live()
                    .map(|worker| {
                        let manager = worker.scheduler.kv_cache_manager();
                        HbmWorkerStats {
                            worker: worker.global_id as u32,
                            rank: worker.rank as u32,
                            running: worker.scheduler.num_running() as u64,
                            waiting: worker.scheduler.num_waiting() as u64,
                            capacity_bytes: manager.capacity_bytes(),
                            resident_prefix_bytes: manager.resident_prefix_bytes(),
                            active_or_reserved_bytes: manager.active_or_reserved_bytes(),
                            eviction_events: manager.hbm_eviction_events(),
                            evicted_bytes: manager.hbm_evicted_bytes(),
                        }
                    })
                    .collect();
                HbmPoolStats { role, workers }
            })
            .collect()
    }

    /// The memory graph's stores and links over the run so far, `None`
    /// when no worker has tiers.
    pub fn memory_metrics(&self) -> Option<crate::metrics::MemoryMetrics> {
        let g = self.topology.memory.lock().unwrap();
        let tiered = (0..g.num_workers()).any(|w| g.num_tiers(w) > 0);
        if !tiered {
            return None;
        }
        let elapsed = self.current_time.max(f64::MIN_POSITIVE);
        let stores = g
            .store_totals()
            .into_iter()
            .map(|(name, t)| crate::metrics::StoreMetrics {
                name,
                instances: t.instances,
                capacity_blocks: t.capacity_blocks,
                held_blocks: t.held_blocks,
                bytes_written: t.bytes_written,
                bytes_read: t.bytes_read,
                dead_bytes: t.dead_bytes,
                evictions: t.evictions,
                expired: t.expired,
            })
            .collect();
        let links = g
            .edge_totals()
            .into_iter()
            .map(|(name, t)| crate::metrics::EdgeMetrics {
                name,
                instances: t.instances,
                capacity: t.capacity,
                bytes_moved: t.bytes_moved,
                utilisation: if t.capacity > 0.0 && t.instances > 0 {
                    // `instances` counts both directions of each link.
                    t.bytes_moved / (t.instances as f64 * t.capacity * elapsed)
                } else {
                    0.0
                },
            })
            .collect();
        let write_policy = g.write_policy(0).name().to_string();
        let eviction_policy = g
            .stores()
            .first()
            .map(|s| s.eviction.name().to_string())
            .unwrap_or_default();
        Some(crate::metrics::MemoryMetrics {
            write_policy,
            eviction_policy,
            stores,
            links,
            bytes_written: g.flows().bytes_submitted_write,
            bytes_promoted: g.flows().bytes_submitted_worker,
            write_race_waits: g.write_race_waits,
            peak_transfers_in_flight: g.flows().peak_in_flight() as u64,
            peer_hbm_bytes_promoted: g.peer_hbm_bytes_promoted,
            pin_stalls: g.pin_stalls(),
            partial_landings: g.partial_landings,
        })
    }

    /// Prefix-cache lookup statistics summed over every worker's KV manager.
    pub fn aggregate_prefix_cache(&self) -> PrefixCacheStats {
        let mut total = PrefixCacheStats::default();
        for pool in &self.topology.pools {
            for worker in pool.live() {
                total += worker.scheduler.kv_cache_manager().prefix_cache_stats();
            }
        }
        total
    }

    /// Aggregate KV cache utilisation across pools, weighted by capacity.
    /// Returns 0.0 if no KV cache is configured anywhere.
    /// KV bytes written into HBM so far, summed over workers: every fresh
    /// block allocation, whether for computed or promoted content. Monotone;
    /// the difference between two readings is the distinct content that
    /// entered the caches in between (a reuse distance, in bytes).
    pub fn kv_bytes_written(&self) -> u64 {
        self.topology
            .pools
            .iter()
            .flat_map(|p| p.live())
            .map(|w| w.scheduler.kv_cache_manager().bytes_written())
            .sum()
    }

    /// `kv_bytes_written` plus the bytes of free (evictable) blocks that
    /// prefix hits pulled back into use, summed over workers. Fresh writes
    /// alone undercount the LRU stack distance: a hit moves its content to
    /// the recently-used end too. The two readings bracket it.
    pub fn kv_bytes_touched(&self) -> u64 {
        self.topology
            .pools
            .iter()
            .flat_map(|p| p.live())
            .map(|w| w.scheduler.kv_cache_manager().bytes_touched())
            .sum()
    }

    pub fn kv_cache_util(&self) -> f64 {
        let mut used = 0.0_f64;
        let mut total = 0.0_f64;
        for pool in &self.topology.pools {
            for worker in pool.live() {
                let mgr = worker.scheduler.kv_cache_manager();
                let u = mgr.utilization();
                let cap = mgr.total_blocks() as f64;
                used += u * cap;
                total += cap;
            }
        }
        if total > 0.0 {
            used / total
        } else {
            0.0
        }
    }

    /// Whether any work is in flight. `false` when the heap is empty AND
    /// every pool has no running/waiting requests AND no parked handoffs.
    pub fn is_idle(&self) -> bool {
        self.events.is_empty()
            && self.parked.is_empty()
            && self.aggregate_running() == 0
            && self.aggregate_waiting() == 0
    }

    /// Pop the next event and process it. Returns information about what
    /// happened, including any completed requests.
    pub fn step(&mut self) -> Result<StepOutcome, String> {
        let ev = self.events.pop().ok_or_else(|| {
            format!(
                "step called with empty event queue (running={}, waiting={}, parked={}, t={})",
                self.aggregate_running(),
                self.aggregate_waiting(),
                self.parked.len(),
                self.current_time
            )
        })?;
        if ev.time + 1e-9 < self.current_time {
            return Err(format!(
                "event at t={} earlier than clock t={}",
                ev.time, self.current_time
            ));
        }
        self.current_time = ev.time;

        let now = self.current_time;
        let mut outcome = match ev.kind {
            EventKind::Arrival(mut req) => {
                if let Some(step) = &mut req.session {
                    if let Some(at_parent) = step.parent_bytes_written {
                        step.reuse_distance_bytes =
                            Some(self.kv_bytes_written().saturating_sub(at_parent));
                    }
                    if let Some(at_parent) = step.parent_bytes_touched {
                        step.reuse_touched_bytes =
                            Some(self.kv_bytes_touched().saturating_sub(at_parent));
                    }
                }
                self.handle_arrival(*req)?;
                StepOutcome::new(now, StepKind::Arrival)
            }
            EventKind::WorkerReady { pool, worker } => {
                let (iteration, completions) = self.handle_worker_ready(pool, worker);
                StepOutcome {
                    iteration,
                    completions,
                    ..StepOutcome::new(now, StepKind::Iteration)
                }
            }
            EventKind::FlowDrain { generation } => {
                if generation == self.flow_generation {
                    self.flow_due = None;
                }
                StepOutcome::new(now, StepKind::LinkComplete)
            }
            EventKind::PrefetchDue { pool, worker } => {
                self.prefetch_armed[pool][worker] = None;
                if self.topology.pools[pool].is_live(worker) {
                    self.maybe_wake_worker(pool, worker, now);
                }
                StepOutcome::new(now, StepKind::PrefetchDue)
            }
            EventKind::HandoffStart(id) => {
                let req = self
                    .queued_handoffs
                    .remove(&id)
                    .ok_or_else(|| format!("hand-off of {id} was never queued"))?;
                self.start_handoff(req)?;
                StepOutcome::new(now, StepKind::HandoffStart)
            }
            EventKind::WorkerJoin { pool, worker } => StepOutcome {
                worker_event: self.join_worker(pool, worker),
                ..StepOutcome::new(now, StepKind::WorkerJoined)
            },
            EventKind::WorkerRetire { pool, worker } => StepOutcome {
                worker_event: self.retire_drained_worker(pool, worker),
                ..StepOutcome::new(now, StepKind::WorkerRemoved)
            },
            EventKind::WorkerKilled { pool, worker } => StepOutcome {
                worker_event: Some(WorkerEvent::Removed {
                    pool,
                    worker,
                    killed: true,
                }),
                ..StepOutcome::new(now, StepKind::WorkerRemoved)
            },
        };
        // Whatever the event was, move the memory graph's transfers to now,
        // hand finished hand-offs to their decode workers, wake workers with
        // finished promotions, and re-arm the drain event.
        self.pump_flows()?;
        self.check_draining();
        outcome.failed = std::mem::take(&mut self.failed);
        outcome.completions.append(&mut self.completed);
        Ok(outcome)
    }

    /// Bring the memory graph to `now`; deliver completed hand-offs; wake
    /// workers whose promotions completed (their next `schedule()` collects
    /// them); (re)schedule the `FlowDrain` for the next completion.
    fn pump_flows(&mut self) -> Result<(), String> {
        let now = self.current_time;
        let (owners, next) = {
            let mut g = self.topology.memory.lock().unwrap();
            g.advance(now);
            (g.owners_with_completions(), g.next_completion_delay())
        };
        for owner in owners {
            match owner {
                Owner::Handoff => {
                    let done = self
                        .topology
                        .memory
                        .lock()
                        .unwrap()
                        .take_completed(Owner::Handoff);
                    if let Some(done) = done {
                        self.deliver_drained(done, now)?;
                    }
                }
                Owner::Worker(w) => match self.locate_worker(w) {
                    Some((pool, idx)) => self.maybe_wake_worker(pool, idx, now),
                    // A removed worker's transfers drain unobserved.
                    None => {
                        self.topology.memory.lock().unwrap().take_completed(owner);
                    }
                },
                // Writes land inside the graph's own `advance`.
                Owner::Write => {}
            }
        }
        match next {
            Some(delay) => {
                let due = now + delay;
                if self.flow_due.is_none_or(|t| (t - due).abs() > 1e-12) {
                    self.flow_generation += 1;
                    self.flow_due = Some(due);
                    self.push(
                        due,
                        EventKind::FlowDrain {
                            generation: self.flow_generation,
                        },
                    );
                }
            }
            None => {
                if self.flow_due.is_some() {
                    self.flow_generation += 1;
                    self.flow_due = None;
                }
            }
        }
        Ok(())
    }

    /// `(pool, slot)` of the live worker with memory-graph id `global`.
    fn locate_worker(&self, global: usize) -> Option<(PoolId, usize)> {
        self.topology.slot_of.get(&WorkerId(global as u32)).copied()
    }

    fn handle_arrival(&mut self, req: Request) -> Result<(), String> {
        let entry = self.topology.entry_pool();
        let now = self.current_time;
        match self.topology.pools[entry].pick(&req, now)? {
            Some(slot) => self.deliver_to_worker(entry, slot, req),
            None => self.fail(req, entry, None, FailureReason::NoWorker),
        }
        Ok(())
    }

    fn deliver_to_worker(&mut self, pool_id: PoolId, worker_idx: usize, mut req: Request) {
        let worker = self.topology.pools[pool_id].worker_mut(worker_idx);
        req.worker = Some(worker.global_id as u32);
        worker.scheduler.add_request_at(req, self.current_time);
        self.flush_kv_events(pool_id, worker_idx);
        self.maybe_wake_worker(pool_id, worker_idx, self.current_time);
    }

    /// Hand `req` back unfinished in the next step's outcome.
    fn fail(
        &mut self,
        mut req: Request,
        pool: PoolId,
        worker: Option<WorkerId>,
        reason: FailureReason,
    ) {
        req.lose_worker(self.current_time);
        self.failed.push(FailedRequest {
            request: req,
            pool,
            worker,
            reason,
            time: self.current_time,
        });
    }

    /// Tell the router of the first tokens `slot`'s last iteration
    /// produced, if that iteration has ended by now; forget them either
    /// way.
    fn report_first_tokens(&mut self, pool: PoolId, slot: usize) {
        let now = self.current_time;
        let WorkerPool { slots, router, .. } = &mut self.topology.pools[pool];
        let w = slots[slot].as_mut().expect("a live worker");
        let id = w.id();
        let ended = w.prefilled_at <= now;
        for request_id in w.prefilled.drain(..) {
            if ended {
                router.on_prefill_complete(&request_id, id, now);
            }
        }
    }

    /// Report `slot`'s recorded prefix-cache changes to its pool's router.
    fn flush_kv_events(&mut self, pool: PoolId, slot: usize) {
        let now = self.current_time;
        let WorkerPool { slots, router, .. } = &mut self.topology.pools[pool];
        let Some(worker) = slots[slot].as_mut() else {
            return;
        };
        let events = worker.scheduler.kv_cache_manager_mut().take_kv_events();
        let id = worker.id();
        for event in events {
            match event {
                KvEvent::Stored { parent, blocks } => router.on_kv_stored(id, parent, &blocks, now),
                KvEvent::Removed { sequence_hashes } => {
                    router.on_kv_removed(id, &sequence_hashes, now)
                }
            }
        }
    }

    /// Wake `worker`'s lockstep group (through its leader) at `when`,
    /// unless it is already busy or armed.
    fn maybe_wake_worker(&mut self, pool: PoolId, worker: usize, when: f64) {
        let leader = self.topology.pools[pool].leader_of(worker);
        if !self.worker_busy[pool][leader] {
            self.worker_busy[pool][leader] = true;
            self.push(
                when,
                EventKind::WorkerReady {
                    pool,
                    worker: leader,
                },
            );
        }
    }

    fn handle_worker_ready(
        &mut self,
        pool: PoolId,
        worker: usize,
    ) -> (Option<IterationInfo>, Vec<RequestTiming>) {
        // `WorkerReady` fires at the END of the worker's prior iteration (or
        // at t=0 when the worker first wakes). Mark idle and re-evaluate.
        self.worker_busy[pool][worker] = false;
        if !self.topology.pools[pool].is_live(worker) {
            return (None, Vec::new());
        }
        let now = self.current_time;
        let role = self.topology.role_for_pool(pool);
        let mut members = std::mem::take(&mut self.scratch.group);
        members.clear();
        members.extend_from_slice(self.topology.pools[pool].members_of(worker));

        // Under an overlap scheduler the previous iteration's first tokens
        // wait for this iteration (below); the rest are out now.
        let any_overlapped = members.iter().any(|&m| {
            !self.topology.pools[pool]
                .worker(m)
                .overlap_first_tokens
                .is_empty()
        });
        let overlapped: Vec<(usize, Vec<String>)> = if any_overlapped {
            members
                .iter()
                .map(|&m| {
                    let w = self.topology.pools[pool].worker_mut(m);
                    let ids = std::mem::take(&mut w.overlap_first_tokens);
                    w.prefilled.retain(|id| !ids.contains(id));
                    (m, ids)
                })
                .collect()
        } else {
            Vec::new()
        };
        // First tokens the previous iteration produced: it ends now.
        for &m in &members {
            self.report_first_tokens(pool, m);
        }

        let outcome = self.run_iteration(pool, worker, role, now);
        // An overlap scheduler emits the last iteration's first tokens once
        // it has launched this one, which a busy engine has done when this
        // one ends; an idle engine emits them now.
        let next_end = outcome.iteration.as_ref().map(|i| i.end_time);
        for (m, ids) in overlapped {
            if ids.is_empty() {
                continue;
            }
            let w = self.topology.pools[pool].worker_mut(m);
            match next_end {
                Some(end) => {
                    for id in &ids {
                        w.scheduler.defer_first_token(id, end);
                    }
                    w.prefilled.extend(ids);
                    w.prefilled_at = end;
                }
                None => {
                    let gid = WorkerId(w.global_id as u32);
                    let router = &mut self.topology.pools[pool].router;
                    for id in ids {
                        router.on_prefill_complete(&id, gid, now);
                    }
                }
            }
        }

        // Completions from `schedule()` finished at the end of the *previous*
        // iteration, i.e. `now`. Handoffs finished prefill in the iter that
        // ran *during* this step, so they start at `end_time`: each gets its
        // own event there (queued before this worker's next `WorkerReady`,
        // so it fires first at that instant).
        let mut timings = Vec::with_capacity(outcome.completed.len());
        for req in outcome.completed {
            if matches!(role, PoolRole::DisaggPrefill) {
                self.record_session_prefill_work(&req);
            }
            if let Some(on) = req.worker {
                self.topology.pools[pool].router.on_request_complete(
                    &req.request_id,
                    WorkerId(on),
                    now,
                );
            }
            timings.push(self.finalise(req, now));
        }
        // The schedules' allocations and evictions happened now too.
        for &m in &members {
            self.flush_kv_events(pool, m);
        }
        let handoff_time = outcome
            .iteration
            .as_ref()
            .map(|i| i.end_time)
            .unwrap_or(now);
        for (source, req) in outcome.handed_off {
            let id = req.request_id.clone();
            self.handoff_source.insert(id.clone(), source);
            self.queued_handoffs.insert(id.clone(), req);
            self.push(handoff_time, EventKind::HandoffStart(id));
        }
        if let Some(end) = outcome
            .iteration
            .as_ref()
            .map(|i| i.end_time)
            .or(outcome.spin_until)
        {
            self.worker_busy[pool][worker] = true;
            self.push(end, EventKind::WorkerReady { pool, worker });
        } else if outcome.preempted {
            // The scheduler preempted but ran nothing (e.g. every runner was
            // preempted, which also blocks admission for the step). State
            // changed, so re-arm immediately; otherwise the worker only
            // wakes on a new arrival and the engine can stall with work
            // queued. This can't loop at one timestamp: a preempt-only pass
            // empties `running`, and the re-run either schedules something
            // or preempts nothing.
            let p = &self.topology.pools[pool];
            let has_work = p.members_of(worker).iter().any(|&m| {
                let sch = &p.worker(m).scheduler;
                sch.num_running() > 0 || sch.num_waiting() > 0
            });
            if has_work {
                self.maybe_wake_worker(pool, worker, now);
            }
        }
        // Requests parked on a KV promotion need no timer: the memory
        // graph's `FlowDrain` wakes this group when the transfer completes.
        // Arm each member's next prefetch plan as its own event, so it
        // fires whether or not the group is busy then.
        for &m in &members {
            if let Some(t) = self.topology.pools[pool]
                .worker(m)
                .scheduler
                .next_prefetch_at()
            {
                if t > now && self.prefetch_armed[pool][m] != Some(t) {
                    self.prefetch_armed[pool][m] = Some(t);
                    self.push(t, EventKind::PrefetchDue { pool, worker: m });
                }
            }
        }
        self.scratch.group = members;
        (outcome.iteration, timings)
    }

    /// The step kinds a DP replica tries, in order, for its next lockstep
    /// step: a prefill step on every rank (admitting new prompts unless the
    /// prefill delayer holds them), and failing that a decode step; and
    /// whether the pass takes the scheduler's `idle_pass_s` when neither
    /// runs (the replica spins while it holds prompts back). Advances the
    /// delayer's count: a pass that held new prompts back counts, a pass
    /// that admits them resets it. In a long prompt's decode interval no
    /// rank prefills; while a long prompt continues on a rank, every rank's
    /// new prompts prefill with its chunk (SGLang's delayer, told to allow).
    fn replica_step_plans(
        &mut self,
        pool: PoolId,
        leader: usize,
        members: &[usize],
    ) -> (Vec<StepPlan>, bool) {
        let ws = &self.topology.pools[pool];
        let config = ws.worker(leader).scheduler.config();
        let (max_passes, idle_pass_s) = (config.prefill_delayer_max_passes, config.idle_pass_s);
        let long_min =
            (config.long_prefill_decode_interval > 0).then_some(config.long_prefill_min_tokens);
        let (mut with_prefill, mut chunking, mut decoding, mut long) =
            (0usize, false, false, false);
        for &m in members {
            let sched = &ws.worker(m).scheduler;
            with_prefill += usize::from(sched.has_prefill_work());
            chunking |= sched.has_chunk_in_progress();
            decoding |= sched.has_decode_work();
            if let Some(min) = long_min {
                long |= sched.has_chunk_of_prompt_at_least(min);
            }
        }
        let w = self.topology.pools[pool].worker_mut(leader);
        if w.defer_passes > 0 {
            if decoding || idle_pass_s > 0.0 {
                w.defer_passes -= 1;
                return (vec![StepPlan::Decode], true);
            }
            // Passes that run nothing take no time: the interval is over.
            w.defer_passes = 0;
        }
        let mixed = with_prefill > 0 && with_prefill < members.len();
        // Held back: some ranks have prefill and some don't, no long prompt
        // continues, the count is short of the limit, and the pass does
        // something or takes time (one that runs nothing in no time would
        // spin all its passes at once).
        let hold = max_passes > 0
            && mixed
            && !long
            && w.delay_passes + 1 < max_passes
            && (decoding || chunking || idle_pass_s > 0.0);
        if hold {
            w.delay_passes += 1;
        } else if with_prefill > 0 {
            w.delay_passes = 0;
        }
        (
            vec![StepPlan::Prefill { admit: !hold }, StepPlan::Decode],
            hold,
        )
    }

    /// One iteration of `worker`'s lockstep group (a whole replica, or the
    /// `tp` ranks of a DP-attention replica stepped together). Members with
    /// no local work or queued transfer completion are skipped;
    /// the remaining schedules form a union batch priced on the replica-wide
    /// roofline, plus — with more than one rank — the attention skew: the
    /// slowest rank's own-GPU attention time over the mean rank's, since
    /// the ranks meet at every layer's FFN collective.
    fn run_iteration(
        &mut self,
        pool: PoolId,
        worker: usize,
        role: PoolRole,
        now: f64,
    ) -> RunIterationOutcome {
        let mut scratch = std::mem::take(&mut self.scratch);
        let outcome = self.run_iteration_in(&mut scratch, pool, worker, role, now);
        self.scratch = scratch;
        outcome
    }

    fn run_iteration_in(
        &mut self,
        scratch: &mut IterationScratch,
        pool: PoolId,
        worker: usize,
        role: PoolRole,
        now: f64,
    ) -> RunIterationOutcome {
        let IterationScratch {
            members,
            entries,
            active_members,
            cost_tokens,
            was_prefill,
            round_commits,
            accepted_extra,
            draft_widths,
            decode_counts,
            draft_plans,
            plans,
            ..
        } = scratch;
        let correction = self.time_correction;
        members.clear();
        members.extend_from_slice(self.topology.pools[pool].members_of(worker));
        let members = &*members;
        let grouped = members.len() > 1;

        // Every member's schedule used to advance the shared graph before
        // collecting its own completions. Advance once up front so a
        // completion landing exactly at `now` is visible before deciding an
        // otherwise-empty rank can be skipped; inspect without draining so
        // the live rank's schedule remains the sole collector.
        let completed_owners = if grouped {
            let mut graph = self.topology.memory.lock().unwrap();
            graph.advance(now);
            graph.owners_with_completions()
        } else {
            Vec::new()
        };

        // Schedule each live member; the batch is the union, each entry
        // tagged with its member and index into that member's running set.
        let mut completed = Vec::new();
        let mut preempted = false;
        entries.clear();
        active_members.clear();
        // A DP replica without mixed batches runs one kind of step on every
        // rank (SGLang under speculation: a rank's prefill idles the rest).
        let replica_plans = grouped
            && !self.topology.pools[pool]
                .worker(worker)
                .scheduler
                .mixes_prefill_with_decode();
        let (step_plans, spins) = if replica_plans {
            self.replica_step_plans(pool, worker, members)
        } else {
            (vec![StepPlan::Auto], false)
        };
        for plan in step_plans {
            for &m in members {
                let w = self.topology.pools[pool].worker_mut(m);
                if grouped && w.can_skip_schedule(now, &completed_owners) {
                    continue;
                }
                let decision = w.scheduler.schedule_step(now, plan);
                if !decision.batch.is_empty() {
                    active_members.push(m);
                }
                completed.extend(decision.completed);
                preempted |= decision.num_preempted > 0;
                entries.extend(decision.batch.iter().map(|s| (m, s.idx, s.num_tokens)));
                w.scheduler.recycle_batch(decision.batch);
            }
            if !entries.is_empty() {
                break;
            }
        }
        if entries.is_empty() {
            let idle_pass_s = self.topology.pools[pool]
                .worker(worker)
                .scheduler
                .config()
                .idle_pass_s;
            return RunIterationOutcome {
                iteration: None,
                completed,
                handed_off: Vec::new(),
                preempted,
                spin_until: (spins && idle_pass_s > 0.0).then_some(now + idle_pass_s),
            };
        }
        let batch_size = entries.len();

        // Capture per-request progress (and was_prefill) before mutating.
        let mut progress = Vec::with_capacity(batch_size);
        round_commits.clear();
        {
            let ws = &self.topology.pools[pool];
            for &(m, idx, tokens) in entries.iter() {
                let req = &ws.worker(m).scheduler.running()[idx];
                progress.push(RequestProgress {
                    request_id: req.request_id.clone(),
                    was_prefill: req.is_prefill(),
                    context_tokens: req.num_computed_tokens,
                    num_tokens: tokens,
                    num_output: 0,
                });
                round_commits.push(req.pending_round_commits);
            }
        }

        // Speculative decoding (vLLM-faithful). Each decode request's draft
        // length was decided at the END of the previous iteration and stored
        // as `pending_draft_len`; the scheduler has already reserved a
        // `1 + draft` verify pass in the token budget and KV (trimming `draft`
        // to fit if capacity was tight), which is exactly what the batch's
        // `num_tokens` now carries. So here we only realise the *outcome*:
        // how many of the reserved draft tokens are accepted, and advance by
        // `accepted + 1`. The verify pass itself (`1 + draft` tokens) is the
        // cost. Prefill and chunked-prefill continuations (was_prefill) are
        // never speculated. Under DP-attention the batch is every rank's, each
        // request verified on its own rank within the lockstep step.
        // Verify width per request.
        cost_tokens.clear();
        cost_tokens.extend(entries.iter().map(|e| e.2));
        accepted_extra.clear();
        accepted_extra.resize(batch_size, 0);
        draft_widths.clear(); // decode sequences only
        if self.spec.is_some() {
            for j in 0..batch_size {
                if progress[j].was_prefill {
                    continue;
                }
                let draft = cost_tokens[j].saturating_sub(1);
                draft_widths.push(draft);
                accepted_extra[j] = SpecPlanner::accepted(draft, round_commits[j]);
            }
        }

        let (mut iter_time, calibrated, bandwidth_util, flops_util) = {
            let ws = &self.topology.pools[pool];
            let batch_refs: Vec<&Request> = entries
                .iter()
                .map(|&(m, idx, _)| &ws.worker(m).scheduler.running()[idx])
                .collect();
            was_prefill.clear();
            was_prefill.extend(progress.iter().map(|p| p.was_prefill));
            let ce = &ws.worker(worker).compute_engine;
            let (mut t, calibrated, bw_util, fl_util) = Self::price_step(
                self.spec.as_ref(),
                ce,
                &batch_refs,
                cost_tokens,
                was_prefill,
                correction,
            );
            if grouped && !calibrated {
                // Attention skew across the ranks: max own-GPU attention
                // time minus the mean the union pricing already charged.
                // Empty ranks contribute exactly zero. The common one-live-
                // rank case can reuse the union vectors and avoid building
                // and filtering one pair of vectors per replica rank.
                let (max, sum) = if active_members.len() == 1 {
                    let attention = ce.attention_seconds_on_one_gpu(&batch_refs, cost_tokens);
                    (attention, attention)
                } else {
                    let mut max = 0.0_f64;
                    let mut sum = 0.0_f64;
                    for &m in active_members.iter() {
                        let refs: Vec<&Request> = entries
                            .iter()
                            .filter(|e| e.0 == m)
                            .map(|&(_, idx, _)| &ws.worker(m).scheduler.running()[idx])
                            .collect();
                        let toks: Vec<u32> =
                            entries.iter().filter(|e| e.0 == m).map(|e| e.2).collect();
                        let attention = ce.attention_seconds_on_one_gpu(&refs, &toks);
                        max = max.max(attention);
                        sum += attention;
                    }
                    (max, sum)
                };
                let mean = sum / members.len() as f64;
                t += (max - mean).max(0.0);
            }
            // A DP replica's prefill step that a captured graph can run:
            // pure prefill, no rank over the graph's size.
            if let Some(lin) = ce
                .linear_step_cost()
                .filter(|l| l.graph_prefill_max_tokens > 0)
            {
                if grouped && calibrated && was_prefill.iter().all(|&p| p) {
                    let per_member = |m: usize| -> u32 {
                        entries.iter().filter(|e| e.0 == m).map(|e| e.2).sum()
                    };
                    if members
                        .iter()
                        .all(|&m| per_member(m) <= lin.graph_prefill_max_tokens)
                    {
                        // Never below what the step had left after any load
                        // overlap credit.
                        t = (t + lin.t0p_graph - lin.t0p).max(t.min(lin.t0 + lin.t0p_graph));
                    }
                }
            }
            (t, calibrated, bw_util, fl_util)
        };
        if let Some(spec) = &mut self.spec {
            let ce = &self.topology.pools[pool].worker(worker).compute_engine;
            // Drafter overhead on roofline-priced speculated steps, for the
            // whole batch on the replica's GPUs (every rank's drafts, under
            // DP-attention, as the verify is priced). Table- and fit-priced
            // steps skip this — the measured wall gap already embodies the
            // full engine step, drafter included.
            if !calibrated && draft_widths.iter().any(|&d| d > 0) {
                let peak = ce.bf16_peak_flops();
                let bw = ce.mem_bandwidth();
                iter_time += spec.drafter_seconds(draft_widths, peak, bw, iter_time);
            }
            // Constrained-GatedAggregate per-switch stall: a width change
            // decided at the end of the previous round costs the engine a
            // rebuild on the first round executed at the new width.
            iter_time += spec.take_pending_switch_cost((pool, worker));
        }
        let end_time = now + iter_time;

        for (j, &(m, idx, tokens)) in entries.iter().enumerate() {
            // Decode: advance by the verified tokens (bonus + accepted), NOT
            // the verify width (`num_tokens` = 1 + draft, the cost). Prefill
            // (including chunked continuations and recompute after
            // preemption): advance by the scheduled chunk.
            let adv = if progress[j].was_prefill {
                tokens
            } else {
                1 + accepted_extra[j]
            };
            let w = self.topology.pools[pool].worker_mut(m);
            progress[j].num_output = w.scheduler.record_progress(idx, adv, end_time);
            // A first token: the router hears of it when the pass ends. A
            // prefill pool's hand-offs report theirs as they leave.
            if progress[j].was_prefill && progress[j].num_output > 0 {
                let req = &w.scheduler.running()[idx];
                if !matches!(role, PoolRole::DisaggPrefill) || req.is_finished() {
                    let id = req.request_id.clone();
                    if w.scheduler.overlap_schedule() && !req.is_finished() {
                        w.overlap_first_tokens.push(id.clone());
                    }
                    w.prefilled.push(id);
                    w.prefilled_at = end_time;
                }
            }
        }

        // A chunk of a long prompt with chunks left: no rank prefills for the
        // decode interval that follows.
        if replica_plans {
            let p = &self.topology.pools[pool];
            let config = p.worker(worker).scheduler.config();
            let (interval, min) = (
                config.long_prefill_decode_interval,
                config.long_prefill_min_tokens,
            );
            if interval > 0
                && entries
                    .iter()
                    .zip(progress.iter())
                    .any(|(&(m, idx, _), done)| {
                        let req = &p.worker(m).scheduler.running()[idx];
                        done.was_prefill && req.is_prefill() && req.num_prompt_tokens >= min
                    })
            {
                self.topology.pools[pool].worker_mut(worker).defer_passes = interval;
            }
        }

        let mut handed_off = Vec::new();
        if matches!(role, PoolRole::DisaggPrefill) {
            // Anything whose prefill is now complete leaves its worker via
            // the link; its KV stays allocated here until the transfer
            // drains (`deliver_drained`).
            for &m in members {
                let w = self.topology.pools[pool].worker_mut(m);
                let gid = w.global_id;
                handed_off.extend(
                    w.scheduler
                        .take_prefill_complete()
                        .into_iter()
                        .map(|r| (gid, r)),
                );
            }
        }

        // Decide the decode batch's draft depth for its NEXT step. Drafting
        // happens here, at the end of the step -- the one instant when the
        // drafter is about to run AND the carry-over decode set is known.
        // The next scheduler pass reads `pending_draft_len` and reserves
        // `1 + draft` of budget + KV.
        //
        // Under DP-attention the decode set is every rank's, member by
        // member: the replica drafts and verifies them in one lockstep step,
        // so its depths are planned over the whole batch and handed back to
        // each rank for its own requests.
        if let Some(spec) = &mut self.spec {
            let p = &self.topology.pools[pool];
            // The decode set, member by member, and how many each holds.
            let mut dec: Vec<&Request> = Vec::new();
            decode_counts.clear();
            for &m in members {
                let before = dec.len();
                dec.extend(
                    p.worker(m)
                        .scheduler
                        .running()
                        .iter()
                        .filter(|r| !r.is_prefill() && !r.is_finished()),
                );
                decode_counts.push(dec.len() - before);
            }
            let w = p.worker(worker);
            // The measured table's KV-length correction (0 without a table
            // or ref_seq_len): width-independent, but it moves the argmax.
            let kv_delta = match (spec.ref_seq_len(), spec.measured_cost()) {
                (Some(ref_len), Some(_)) => {
                    w.compute_engine.kv_read_seq_delta_seconds(&dec, ref_len)
                }
                _ => 0.0,
            };
            // Analytic verify cost of the decode sub-batch at width 1 + g,
            // through the real roofline (MoE-coupon- and MLA-aware), and
            // across DP-attention ranks the attention skew the step will
            // pay. Prefill is deliberately not priced in: the mandatory
            // prefill is proportional to committed output, so it cancels in
            // the argmax, and routing prefill tokens through the cost model
            // trips the MoE coupon (verify looks free).
            let ce = &w.compute_engine;
            let mut rest = dec.as_slice();
            let part_refs: Vec<&[&Request]> = decode_counts
                .iter()
                .map(|&count| {
                    let (part, others) = rest.split_at(count);
                    rest = others;
                    part
                })
                .collect();
            let roofline = |g: u32| lockstep_verify_seconds(ce, &part_refs, g + 1);
            let costs = PlanCosts {
                roofline: &roofline,
                kv_delta,
                peak: ce.bf16_peak_flops(),
                bw: ce.mem_bandwidth(),
            };
            spec.plan_into((pool, worker), &dec, &costs, end_time, draft_plans);
            plans.clear();
            plans.extend(draft_plans.iter().map(|p| (p.draft_len, Some(p.commits))));
            let mut rest = plans.as_slice();
            for (&m, &count) in members.iter().zip(decode_counts.iter()) {
                let (mine, others) = rest.split_at(count);
                self.topology.pools[pool]
                    .worker_mut(m)
                    .scheduler
                    .set_draft_plans(mine);
                rest = others;
            }
        }

        // Time-weighted batch accumulator.
        let dt = (end_time - now).max(0.0);
        let acc = &mut self.pool_batch_acc[pool];
        acc.0 += batch_size as f64 * dt;
        acc.1 += dt;

        RunIterationOutcome {
            preempted,
            spin_until: None,
            iteration: Some(IterationInfo {
                pool,
                worker,
                worker_id: self.topology.pools[pool].worker(worker).id(),
                start_time: now,
                end_time,
                iteration_time: iter_time,
                batch_size,
                bandwidth_util,
                flops_util,
                progress,
            }),
            completed,
            handed_off,
        }
    }

    /// Wall-clock time of a step, plus utilisation figures.
    ///
    /// When a measured step-cost table is present, the step's DECODE portion
    /// is priced from the table at (decode batch size, mean decode verify
    /// width) — the same source the policy's C(g) consults — so speculative
    /// and plain-decode steps are priced commensurately (a no-spec step reads
    /// the table's plain-decode rows, not the analytic roofline). Ragged
    /// verify widths interpolate between the nearest *measured* draft-length
    /// cells; a mean width outside the table's measured range falls back to
    /// the roofline for the whole step. Prefill tokens sharing the step are
    /// priced as the roofline time of the prefill sub-batch alone and ADDED
    /// to the table-priced decode portion (the table embodies a full decode
    /// step; the roofline supplies the prefill increment). Pure-prefill steps
    /// always price through the roofline. When the table declares the
    /// sequence length it was benchmarked at (`ref_seq_len`), the decode
    /// portion also gets a bandwidth-roofline KV-read correction for the live
    /// batch's actual KV lengths, floored at 25% of the table value.
    ///
    /// A worker with a linear step-cost model prices the step from the fit
    /// (the compute engine's step time), and the affine `time_correction`
    /// does not apply on top: the fit is already calibrated.
    ///
    /// Returns `(iteration_time, calibrated, bandwidth_util, flops_util)`,
    /// `calibrated` when the price came from a measured table or a fit and
    /// so already embodies the engine's overheads.
    fn price_step(
        spec: Option<&SpecPlanner>,
        ce: &ComputeEngine,
        batch_refs: &[&Request],
        cost_tokens: &[u32],
        was_prefill: &[bool],
        correction: Option<(f64, f64)>,
    ) -> (f64, bool, f64, f64) {
        let batch_size = batch_refs.len();
        let measured_time: Option<f64> = spec.and_then(|spec| {
            let table = spec.measured_cost()?;
            let dec_idx: Vec<usize> = (0..batch_size).filter(|&j| !was_prefill[j]).collect();
            if dec_idx.is_empty() {
                return None; // pure prefill: roofline
            }
            let mean_w: f64 =
                dec_idx.iter().map(|&j| cost_tokens[j] as f64).sum::<f64>() / dec_idx.len() as f64;
            let g = (mean_w - 1.0).max(0.0);
            let mut t_dec = table.step_time_frac(dec_idx.len() as u32, g)?;
            if let Some(ref_len) = spec.ref_seq_len() {
                let dec_refs: Vec<&Request> = dec_idx.iter().map(|&j| batch_refs[j]).collect();
                let delta = ce.kv_read_seq_delta_seconds(&dec_refs, ref_len);
                // Recontextualisation, not a license to price below any
                // plausible step: floor at 25% of the table value.
                t_dec = (t_dec + delta).max(0.25 * t_dec);
            }
            let t_pre = if dec_idx.len() < batch_size {
                let pre_refs: Vec<&Request> = (0..batch_size)
                    .filter(|&j| was_prefill[j])
                    .map(|j| batch_refs[j])
                    .collect();
                let pre_tokens: Vec<u32> = (0..batch_size)
                    .filter(|&j| was_prefill[j])
                    .map(|j| cost_tokens[j])
                    .collect();
                ce.calculate_iteration_time(&pre_refs, &pre_tokens)
            } else {
                0.0
            };
            Some(t_dec + t_pre)
        });
        let cost = ce.step_cost(batch_refs, cost_tokens);
        // A measured table or a linear fit already embodies the engine's
        // overheads; the correction calibrates roofline-priced steps only.
        let linear = ce.prices_linearly();
        let mut iter_time = match (measured_time, correction) {
            (Some(t), _) => t,
            (None, _) if linear => cost.time,
            (None, Some((alpha, beta))) => alpha * cost.time + beta,
            (None, None) => cost.time,
        };
        let overlap_credit = batch_refs
            .iter()
            .zip(was_prefill)
            .filter(|(_, prefill)| **prefill)
            .map(|(request, _)| request.load_overlap_credit)
            .fold(0.0_f64, f64::max);
        if overlap_credit > 0.0 {
            let pre_idx: Vec<usize> = (0..batch_size).filter(|&i| was_prefill[i]).collect();
            let pre_refs: Vec<&Request> = pre_idx.iter().map(|&i| batch_refs[i]).collect();
            let pre_tokens: Vec<u32> = pre_idx.iter().map(|&i| cost_tokens[i]).collect();
            let pre_time = ce.calculate_iteration_time(&pre_refs, &pre_tokens);
            let dec_idx: Vec<usize> = (0..batch_size).filter(|&i| !was_prefill[i]).collect();
            let decode_floor = if dec_idx.is_empty() {
                0.0
            } else {
                let refs: Vec<&Request> = dec_idx.iter().map(|&i| batch_refs[i]).collect();
                let tokens: Vec<u32> = dec_idx.iter().map(|&i| cost_tokens[i]).collect();
                ce.calculate_iteration_time(&refs, &tokens)
            };
            iter_time = (iter_time - overlap_credit.min(pre_time)).max(decode_floor);
        }
        let bw = ce.bandwidth_utilization(&cost, iter_time);
        let flops = ce.flops_utilization(&cost, iter_time);
        (iter_time, measured_time.is_some() || linear, bw, flops)
    }

    /// A request's prefill on the P pool finishes now. Pick its decode
    /// worker (the router sees the decode pool's load and, for the KV-aware
    /// policies, which decoders already hold part of this context), size the
    /// transfer at the context minus the prompt prefix that decoder already
    /// has resident, and start it on the hand-off link. The request is
    /// delivered to that worker when the transfer drains.
    fn start_handoff(&mut self, mut req: Request) -> Result<(), String> {
        self.record_session_prefill_work(&req);
        let prefill_done_at = self.current_time;
        req.prefill_done_time = Some(prefill_done_at);
        let decode = match self.topology.roles {
            Roles::Disagg { decode, .. } => decode,
            _ => return Err("hand-off on an aggregated topology".to_string()),
        };
        let id = req.request_id.clone();
        let from = self
            .handoff_source
            .get(&id)
            .copied()
            .ok_or_else(|| format!("hand-off of {id} has no source worker"))?;
        let source_id = WorkerId(from as u32);
        // A killed source fails its queued hand-offs with it.
        let (sp, si) = self
            .locate_worker(from)
            .ok_or_else(|| format!("hand-off source worker {from} not found"))?;
        self.topology.pools[sp]
            .router
            .on_prefill_complete(&id, source_id, prefill_done_at);
        let Some(worker_idx) = self.topology.pools[decode].pick(&req, prefill_done_at)? else {
            // No decoder takes work: the prefill worker lets the KV go.
            self.handoff_source.remove(&id);
            self.release_handoff_source(sp, si, &mut req);
            self.fail(req, decode, None, FailureReason::NoWorker);
            return Ok(());
        };
        let resident = self.topology.pools[decode]
            .worker(worker_idx)
            .scheduler
            .kv_cache_manager()
            .hbm_prefix_tokens(&req.prompt_block_hashes)
            .min(req.num_computed_tokens);
        req.decode_cached_tokens = Some(resident);
        if let Some(step) = req.session.as_ref() {
            let shared = step.shared_tokens.min(req.num_computed_tokens);
            let parent_prefill = step.shared_prefill_tokens.min(shared);
            let parent_decode = step.shared_decode_tokens.min(shared);
            let prefill = req
                .lookup
                .map(|lookup| lookup.hbm_tokens + lookup.tier_tokens)
                .unwrap_or(req.num_cached_tokens)
                .min(req.num_computed_tokens);
            let decoder = resident.min(shared);
            self.reusable_kv_by_decode_worker[worker_idx] += classify_reusable_kv(
                &self.topology.model,
                ReusableKvObservation {
                    prompt: req.num_computed_tokens,
                    shared,
                    prefill,
                    decoder,
                    parent_prefill,
                    parent_decode,
                },
            );
        }
        let full_bytes = self
            .topology
            .model
            .kv_storage_bytes(req.num_computed_tokens);
        let kv_bytes = full_bytes.saturating_sub(self.topology.model.kv_storage_bytes(resident));
        self.handoff_stats.transfers += 1;
        self.handoff_stats.bytes += kv_bytes;
        self.handoff_stats.bytes_skipped += full_bytes - kv_bytes;
        let to = self.topology.pools[decode].worker(worker_idx).global_id;
        self.topology.memory.lock().unwrap().submit_handoff(
            &id,
            from,
            to,
            kv_bytes,
            prefill_done_at,
        )?;
        self.parked.insert(id, (req, worker_idx));
        Ok(())
    }

    /// The prefill worker in `(pool, slot)` lets go of the KV it held for
    /// `req`'s hand-off: the transfer drained, or will never run. Its
    /// hashes stay hittable there.
    fn release_handoff_source(&mut self, pool: PoolId, slot: usize, req: &mut Request) {
        let now = self.current_time;
        let w = self.topology.pools[pool].worker_mut(slot);
        w.scheduler.release_handed_off(req);
        let id = w.id();
        // Those blocks may be what its queue was waiting for; nothing else
        // wakes it (the next arrival would, some time later).
        let waiting = w.scheduler.num_waiting() > 0;
        self.topology.pools[pool]
            .router
            .on_request_complete(&req.request_id, id, now);
        if waiting {
            self.maybe_wake_worker(pool, slot, now);
        }
    }

    /// Deliver drained hand-offs, in a deterministic order, each to the
    /// decode worker chosen when its transfer began.
    fn deliver_drained(
        &mut self,
        done: std::collections::HashSet<String>,
        now: f64,
    ) -> Result<(), String> {
        let decode_pool = match self.topology.roles {
            Roles::Disagg { decode, .. } => decode,
            _ => return Err("hand-off drain on an aggregated topology".to_string()),
        };
        let mut done: Vec<String> = done.into_iter().collect();
        done.sort();
        for request_id in done {
            let (mut req, worker_idx) = self
                .parked
                .remove(&request_id)
                .ok_or_else(|| format!("hand-off complete for unknown request {request_id}"))?;
            // The prefill worker held the request's KV for the transfer;
            // release it there now.
            let source = self
                .handoff_source
                .remove(&request_id)
                .ok_or_else(|| format!("hand-off of {request_id} has no source worker"))?;
            let (sp, si) = self
                .locate_worker(source)
                .ok_or_else(|| format!("hand-off source worker {source} not found"))?;
            self.release_handoff_source(sp, si, &mut req);
            req.handoff_done_time = Some(now);
            // The first token travels with the KV: the decode side emits it
            // on receipt (Dynamo / sglang PD), so that is when the client
            // sees it. Decode-side TPOT then runs from here.
            req.first_token_time = Some(now);
            self.deliver_to_worker(decode_pool, worker_idx, req);
        }
        Ok(())
    }

    fn finalise(&mut self, req: Request, completion_time: f64) -> RequestTiming {
        let first_token = req.first_token_time.unwrap_or(completion_time);
        // On an aggregated topology the prefill pass produces the first
        // token, so prefill ends when the first token is produced.
        let prefill_done = req.prefill_done_time.unwrap_or(first_token);
        let handoff_done = req.handoff_done_time.unwrap_or(prefill_done);
        RequestTiming {
            request_id: req.request_id,
            arrival_time: req.arrival_time,
            prefill_done_time: prefill_done,
            handoff_done_time: handoff_done,
            first_token_time: first_token,
            completion_time,
            num_prompt_tokens: req.num_prompt_tokens,
            num_output_tokens: req.num_output_tokens,
            num_cached_tokens: req.num_cached_tokens,
            decode_cached_tokens: req.decode_cached_tokens,
            session: req.session,
            worker: req.worker,
            num_preemptions: req.num_preemptions,
            rejected: req.rejected,
            lookup: req.lookup,
        }
    }
}

/// Hand-off transfer totals over the run.
#[derive(Debug, Clone, Copy, Default)]
pub struct HandoffStats {
    pub transfers: u64,
    /// Bytes put on the hand-off link.
    pub bytes: u64,
    /// Bytes not transferred because the chosen decoder already held that
    /// prompt prefix in HBM.
    pub bytes_skipped: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct KvAmountStats {
    pub tokens: u64,
    pub bytes: u64,
}

impl std::ops::AddAssign for KvAmountStats {
    fn add_assign(&mut self, other: Self) {
        self.tokens += other.tokens;
        self.bytes += other.bytes;
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ReusableKvStats {
    pub requests: u64,
    pub reusable: KvAmountStats,
    pub both: KvAmountStats,
    pub decoder_only: KvAmountStats,
    pub prefill_only: KvAmountStats,
    pub neither: KvAmountStats,
    pub prefiller_miss_parent_prefill: KvAmountStats,
    pub prefiller_miss_parent_decode: KvAmountStats,
    pub prefiller_miss_unattributed: KvAmountStats,
}

impl std::ops::AddAssign for ReusableKvStats {
    fn add_assign(&mut self, other: Self) {
        self.requests += other.requests;
        self.reusable += other.reusable;
        self.both += other.both;
        self.decoder_only += other.decoder_only;
        self.prefill_only += other.prefill_only;
        self.neither += other.neither;
        self.prefiller_miss_parent_prefill += other.prefiller_miss_parent_prefill;
        self.prefiller_miss_parent_decode += other.prefiller_miss_parent_decode;
        self.prefiller_miss_unattributed += other.prefiller_miss_unattributed;
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RankReusableKvStats {
    pub worker: u32,
    pub rank: u32,
    pub stats: ReusableKvStats,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SessionPrefillWorkStats {
    pub new_prompt: KvAmountStats,
    pub parent_prefill_recompute: KvAmountStats,
    pub parent_decode_recompute: KvAmountStats,
    pub unattributed_recompute: KvAmountStats,
}

impl std::ops::AddAssign for SessionPrefillWorkStats {
    fn add_assign(&mut self, other: Self) {
        self.new_prompt += other.new_prompt;
        self.parent_prefill_recompute += other.parent_prefill_recompute;
        self.parent_decode_recompute += other.parent_decode_recompute;
        self.unattributed_recompute += other.unattributed_recompute;
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HbmWorkerStats {
    pub worker: u32,
    pub rank: u32,
    pub running: u64,
    pub waiting: u64,
    pub capacity_bytes: u64,
    pub resident_prefix_bytes: u64,
    pub active_or_reserved_bytes: u64,
    pub eviction_events: u64,
    pub evicted_bytes: u64,
}

#[derive(Debug, Clone, Default)]
pub struct HbmPoolStats {
    pub role: String,
    pub workers: Vec<HbmWorkerStats>,
}

#[derive(Debug, Clone, Copy)]
struct ReusableKvObservation {
    prompt: u32,
    shared: u32,
    prefill: u32,
    decoder: u32,
    parent_prefill: u32,
    parent_decode: u32,
}

fn classify_reusable_kv(model: &ModelSpec, observation: ReusableKvObservation) -> ReusableKvStats {
    let ReusableKvObservation {
        prompt,
        shared,
        prefill,
        decoder,
        parent_prefill,
        parent_decode,
    } = observation;
    let prompt = prompt.max(shared);
    let prefill_total = prefill.min(prompt);
    let prefill = prefill_total.min(shared);
    let decoder = decoder.min(shared);
    let parent_prefill = parent_prefill.min(shared);
    let parent_decode_end = parent_prefill.saturating_add(parent_decode).min(shared);
    let lo = prefill.min(decoder);
    let hi = prefill.max(decoder);
    let amount = |start: u32, end: u32| KvAmountStats {
        tokens: end.saturating_sub(start) as u64,
        bytes: model
            .kv_storage_bytes(end)
            .saturating_sub(model.kv_storage_bytes(start)),
    };
    let mut stats = ReusableKvStats {
        requests: u64::from(shared > 0),
        reusable: amount(0, shared),
        both: amount(0, lo),
        neither: amount(hi, shared),
        prefiller_miss_parent_prefill: amount(prefill, parent_prefill),
        prefiller_miss_parent_decode: amount(prefill.max(parent_prefill), parent_decode_end),
        prefiller_miss_unattributed: amount(prefill.max(parent_decode_end), shared),
        ..Default::default()
    };
    if decoder > prefill {
        stats.decoder_only = amount(lo, hi);
    } else {
        stats.prefill_only = amount(lo, hi);
    }
    stats
}

/// Roofline seconds of a lockstep step in which rank `m` verifies its
/// decode requests `parts[m]` at `width` positions each: the union batch on
/// the replica's step cost plus, across DP-attention ranks priced on the
/// roofline, the attention skew — the slowest rank's own-GPU attention over
/// the mean rank's — as [`Engine::run_iteration`] charges it. One part is a
/// replica without DP-attention. What the draft planner weighs each
/// candidate depth by.
fn lockstep_verify_seconds(ce: &ComputeEngine, parts: &[&[&Request]], width: u32) -> f64 {
    let all: Vec<&Request> = parts.iter().flat_map(|p| p.iter().copied()).collect();
    let mut seconds = ce.calculate_iteration_time(&all, &vec![width; all.len()]);
    if parts.len() > 1 && !ce.prices_linearly() {
        let (mut max, mut sum) = (0.0_f64, 0.0_f64);
        for part in parts.iter().filter(|p| !p.is_empty()) {
            let attention = ce.attention_seconds_on_one_gpu(part, &vec![width; part.len()]);
            max = max.max(attention);
            sum += attention;
        }
        seconds += (max - sum / parts.len() as f64).max(0.0);
    }
    seconds
}

struct RunIterationOutcome {
    iteration: Option<IterationInfo>,
    completed: Vec<Request>,
    /// Requests whose prefill finished, with the memory-graph id of the
    /// worker they leave.
    handed_off: Vec<(usize, Request)>,
    /// The scheduler preempted at least one request this pass. Relevant when
    /// `iteration` is `None`: state changed even though nothing ran, so the
    /// worker must be re-armed rather than left to wait for a new arrival.
    preempted: bool,
    /// A DP replica's pass that ran nothing but held prompts back: it ends
    /// then, and the replica passes again.
    spin_until: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, FabricConfig, FabricLink, SpeculativeConfig};

    #[test]
    fn reusable_kv_classification_is_a_complete_joint_partition() {
        let model = Config::test_default().model;
        let observe = |prompt, shared, prefill, decoder, parent_prefill, parent_decode| {
            ReusableKvObservation {
                prompt,
                shared,
                prefill,
                decoder,
                parent_prefill,
                parent_decode,
            }
        };
        let p_longer = classify_reusable_kv(&model, observe(125, 100, 70, 40, 80, 20));
        assert_eq!(p_longer.requests, 1);
        assert_eq!(p_longer.reusable.tokens, 100);
        assert_eq!(p_longer.both.tokens, 40);
        assert_eq!(p_longer.prefill_only.tokens, 30);
        assert_eq!(p_longer.decoder_only.tokens, 0);
        assert_eq!(p_longer.neither.tokens, 30);
        assert_eq!(p_longer.prefiller_miss_parent_prefill.tokens, 10);
        assert_eq!(p_longer.prefiller_miss_parent_decode.tokens, 20);
        assert_eq!(p_longer.prefiller_miss_unattributed.tokens, 0);
        assert_eq!(
            p_longer.reusable.bytes,
            p_longer.both.bytes
                + p_longer.prefill_only.bytes
                + p_longer.decoder_only.bytes
                + p_longer.neither.bytes
        );
        assert_eq!(
            p_longer.decoder_only.bytes + p_longer.neither.bytes,
            p_longer.prefiller_miss_parent_prefill.bytes
                + p_longer.prefiller_miss_parent_decode.bytes
                + p_longer.prefiller_miss_unattributed.bytes
        );

        let d_longer = classify_reusable_kv(&model, observe(125, 100, 30, 80, 60, 40));
        assert_eq!(d_longer.both.tokens, 30);
        assert_eq!(d_longer.decoder_only.tokens, 50);
        assert_eq!(d_longer.prefill_only.tokens, 0);
        assert_eq!(d_longer.neither.tokens, 20);
        assert_eq!(d_longer.prefiller_miss_parent_prefill.tokens, 30);
        assert_eq!(d_longer.prefiller_miss_parent_decode.tokens, 40);
        assert_eq!(d_longer.prefiller_miss_unattributed.tokens, 0);

        let inherited = classify_reusable_kv(&model, observe(125, 100, 0, 0, 0, 0));
        assert_eq!(inherited.prefiller_miss_unattributed.tokens, 100);
    }

    /// One replica of `tp` GPUs with a fabric, DP-attention on or off.
    fn cluster(tp: u32, dp_attention: bool) -> (ClusterSpec, ModelSpec, SchedulerConfig) {
        let base = Config::test_default();
        let mut cluster = base.cluster();
        cluster.hardware.fabric = Some(FabricConfig {
            gpus_per_node: 8,
            scale_up: FabricLink {
                bandwidth: 1e12,
                latency: 1e-6,
                in_network_reduction: false,
            },
            scale_out: None,
        });
        cluster.parallel.tp = tp;
        cluster.parallel.dp_attention = dp_attention;
        cluster.num_workers = 1;
        (cluster, base.model.clone(), base.scheduler.clone())
    }

    fn engine(tp: u32, dp_attention: bool) -> Engine {
        let (c, m, s) = cluster(tp, dp_attention);
        Engine::new(Topology::aggregated(c, m, s).unwrap())
    }

    /// Run until idle; return every iteration in order.
    fn run(engine: &mut Engine) -> Vec<IterationInfo> {
        let mut iters = Vec::new();
        while !engine.is_idle() {
            if let Some(it) = engine.step().unwrap().iteration {
                iters.push(it);
            }
        }
        iters
    }

    #[test]
    fn dp_attention_replica_is_tp_rank_workers_stepped_as_one_group() {
        let e = engine(2, true);
        let pool = &e.topology.pools[0];
        assert_eq!(pool.slots.len(), 2, "one worker per rank");
        assert_eq!(pool.groups, vec![vec![0, 1]]);
        assert!(pool.groups.iter().any(|g| g.len() > 1));
        assert_eq!(pool.worker(0).global_id, 0);
        assert_eq!(pool.worker(1).global_id, 1);
        // Each rank holds half the replica's KV.
        let (c, m, s) = cluster(2, true);
        let whole = c.kv_cache_capacity(&s, c.resident_weight_bytes(&m));
        let per_block = m.kv_storage_bytes(s.block_size);
        let rank_blocks = pool.worker(0).scheduler.kv_cache_manager().total_blocks() as u64;
        assert_eq!(rank_blocks, whole / 2 / per_block);
        // Without DP-attention the same tp is one worker with all of it.
        let e1 = engine(2, false);
        assert_eq!(e1.topology.pools[0].slots.len(), 1);
        assert!(e1.topology.pools[0].groups.iter().all(|g| g.len() == 1));
        assert_eq!(
            e1.topology.pools[0]
                .worker(0)
                .scheduler
                .kv_cache_manager()
                .total_blocks() as u64,
            whole / per_block
        );
    }

    #[test]
    fn idle_rank_skip_guard_includes_graph_completions() {
        let e = engine(2, true);
        let rank = e.topology.pools[0].worker(1);
        assert!(rank.can_skip_schedule(0.0, &[]));
        assert!(!rank.can_skip_schedule(0.0, &[Owner::Worker(rank.global_id)]));
    }

    #[test]
    fn ranks_route_flat_and_an_iteration_covers_every_rank() {
        let mut e = engine(2, true);
        for i in 0..4 {
            e.submit(Request::new(format!("r{i}"), 0, 0.0, 64, 4));
        }
        let iters = run(&mut e);
        // Round-robin over ranks: 2 each.
        assert_eq!(e.router_stats().per_worker, vec![2, 2]);
        // The first iteration is one group step carrying all four prefills,
        // reported under the leader.
        let first = &iters[0];
        assert_eq!(first.worker, 0);
        assert_eq!(first.batch_size, 4);
        assert!(
            iters.iter().all(|i| i.worker == 0),
            "events ride the leader"
        );
    }

    #[test]
    fn a_lopsided_rank_pays_the_attention_skew() {
        // One long prefill on a 2-rank replica lands on one rank: the step
        // costs the union price plus (max − mean) of the ranks' own-GPU
        // attention time = a/2 for one loaded rank and one idle one.
        let mut e = engine(2, true);
        e.submit(Request::new("long".into(), 0, 0.0, 4096, 1));
        let iters = run(&mut e);
        let step = &iters[0];
        assert_eq!(step.batch_size, 1);
        // The first step is the first prefill chunk (chunked prefill).
        let chunk = step.progress[0].num_tokens;
        let (c, m, _) = cluster(2, true);
        let ce = c.compute_engine(m);
        let req = Request::new("probe".into(), 0, 0.0, 4096, 1);
        let union = ce.calculate_iteration_time(&[&req], &[chunk]);
        let a = ce.attention_seconds_on_one_gpu(&[&req], &[chunk]);
        assert!(a > 0.0);
        assert!(
            (step.iteration_time - (union + a / 2.0)).abs() < 1e-12,
            "{} vs {} + {}",
            step.iteration_time,
            union,
            a
        );
        // Two equal prefills, one per rank: no skew — the union price alone.
        let mut e = engine(2, true);
        e.submit(Request::new("a".into(), 0, 0.0, 4096, 1));
        e.submit(Request::new("b".into(), 0, 0.0, 4096, 1));
        let iters = run(&mut e);
        let step = &iters[0];
        assert_eq!(step.batch_size, 2);
        let (ca, cb) = (step.progress[0].num_tokens, step.progress[1].num_tokens);
        assert_eq!(ca, cb);
        let r2 = Request::new("p".into(), 0, 0.0, 4096, 1);
        let union2 = ce.calculate_iteration_time(&[&r2, &r2], &[ca, cb]);
        assert!((step.iteration_time - union2).abs() < 1e-12);
    }

    /// The draft planner weighs a depth by what the lockstep step will
    /// charge for it: on a lopsided DP replica, the union price plus the
    /// idle rank's share of the attention skew.
    #[test]
    fn a_dp_draft_is_planned_at_the_price_its_step_pays() {
        let mut e = engine(2, true);
        let spec: SpeculativeConfig =
            toml::from_str("gamma = 3\n[acceptance]\nkind = \"constant\"\nalpha = 0.7\n").unwrap();
        e.enable_speculative(spec, 1).unwrap();
        // One request: rank 0 decodes, rank 1 idles.
        e.submit(Request::new("r".into(), 0, 0.0, 512, 32));
        let (c, m, _) = cluster(2, true);
        let ce = c.compute_engine(m);
        let mut checked = 0;
        for it in run(&mut e) {
            let p = &it.progress[0];
            if p.was_prefill || p.num_tokens != 4 {
                continue;
            }
            let mut probe = Request::new("p".into(), 0, 0.0, 512, 32);
            probe.record_generated_tokens(p.context_tokens, 0.0);
            assert!(!probe.is_prefill());
            let planned = lockstep_verify_seconds(&ce, &[&[&probe], &[]], 4);
            let alone = lockstep_verify_seconds(&ce, &[&[&probe]], 4);
            assert!(planned > alone, "the skew is in the plan");
            assert!(
                (it.iteration_time - planned).abs() < 1e-15,
                "{} vs {planned}",
                it.iteration_time
            );
            checked += 1;
        }
        assert!(checked > 3, "{checked}");
    }

    #[test]
    fn layerwise_load_credit_prices_load_and_prefill_as_a_pipeline() {
        let (cluster, model, _) = cluster(1, false);
        let ce = cluster.compute_engine(model);
        let mut request = Request::new("prefill".into(), 0, 0.0, 1024, 1);
        let tokens = 512;
        let compute = ce.calculate_iteration_time(&[&request], &[tokens]);

        let load = compute * 0.4;
        request.load_overlap_credit = load;
        let (after_load, ..) = Engine::price_step(None, &ce, &[&request], &[tokens], &[true], None);
        assert!((load + after_load - compute).abs() < 1e-12);

        let load = compute * 2.0;
        request.load_overlap_credit = load;
        let (after_load, ..) = Engine::price_step(None, &ce, &[&request], &[tokens], &[true], None);
        assert!(after_load.abs() < 1e-12);
        assert!((load + after_load - load.max(compute)).abs() < 1e-12);
    }
}
