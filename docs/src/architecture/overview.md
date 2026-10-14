# Architecture Overview

Inference Lab is a discrete-event simulator of an LLM serving engine. The
engine's queueing, scheduling and KV-cache logic are simulated faithfully
(vLLM v1 shapes); the forward pass is replaced by a roofline cost model
built from the hardware spec and the model's architecture.

## Layers

```
catalog       shipped hardware / model presets (embedded TOML)
config        TOML/JSON -> Config, ClusterSpec, DisaggTopology
  |
  |  ModelSpec: weight streams (per-precision FLOPs and bytes, MoE routing)
  |  + layer classes (attention FLOPs, KV read / storage, per-sequence state)
  v
compute       ComputeEngine: roofline step cost for a batch on a cluster
kv_cache      Radix (KV state as ranges of a hash tree), KVCacheManager (a worker's view), MemoryGraph (stores, paths), Flows (edges, rates)
scheduler     Scheduler: waiting/running sets, one schedule() per iteration
  |
  v
simulation    Engine: worker pools + hand-off links + event heap
              SpecPlanner: speculative-decoding draft depth and outcomes
              drivers: Simulator (batch), RealtimeEngine (HTTP), simulate_closed_loop
metrics       MetricsCollector over RequestTiming -> MetricsSummary
```

Every consumer of a model goes through `ModelSpec`'s cost methods. There
are no named architectures: a model is a composition of weight streams and
layer classes (`attention` with optional window and shared KV, `mla` with
optional window / history / indexer, `linear` state), so nothing else in the
simulator knows an architecture's knobs.

## One worker

A worker is one `Scheduler` plus one `ComputeEngine`; the scheduler owns the
worker's `KVCacheManager`. Each iteration:

1. `Scheduler::schedule(now)` reaps finished requests (freeing their KV),
   grows the running set inside the token budget (preempting under KV
   pressure), and admits waiting requests into the leftover budget. It
   returns the batch as `(index, positions)` pairs.
2. `ComputeEngine::step_cost` prices the batch: per-precision streams,
   `max(compute, memory)` per stream summed, plus collectives. Speculative
   decoding may instead price the decode portion from a measured table,
   and a pool with a linear `step_cost` prices the whole step from its fit
   (prefill tokens, decoding requests, attended KV tokens).
3. The engine commits progress to each request (`record_progress`), hands
   completed prefills to the decode pool on a disaggregated topology, and,
   when speculating, asks the `SpecPlanner` for next step's drafts.

## Topologies

`Topology::aggregated` is one pool of identical workers; a `Router`
(round-robin by default; least-loaded, prefix-affinity or KV-aware by
config) picks the worker each arrival enters. `Topology::from_disagg` is a
prefill pool and a decode pool; a hand-off moves over the topology's
`MemoryGraph` from the prefill worker's GPU through its hardware's NIC to
the network core (`kv_link_bw`, optional) and into the decode worker,
sharing every edge with the other transfers in flight (max-min fair,
event-driven); `[router]` fronts the prefill pool and `[decode_router]`
the decode pool.

## Driving a fleet

A driver can change a pool while the engine runs, and route it itself.

- **Replicas come and go.** `Engine::add_worker(pool, ready_at)` builds a
  replica like the pool's own — one worker, or under `dp_attention` its
  `tp` rank workers stepping in lockstep — with its KV caches, schedulers,
  compute, memory-graph vertices and tiers, packed onto the pool's newest
  node while it has free GPUs, and returns its `WorkerId`. It is
  `Starting` until `ready_at`, then `Active` (a `WorkerJoined` step).
  `drain_worker` stops routing to it at once and removes it once it holds
  nothing — no running or queued request on any rank, no hand-off in
  flight to or from one (a `WorkerRemoved` step). `kill_worker` removes it
  now: the unfinished requests its ranks held, and hand-offs to or from
  them, come back in the next step's `failed` list, holding no KV, ready
  to be resubmitted (a pass still running at the kill is unwound, so a
  resubmitted request carries on from the tokens the client saw);
  requests already finished, or refused, complete as usual. Its in-flight
  promotions are cancelled and release the sibling HBM and store blocks
  they pinned, and its pending events are dropped. `pool_members` lists
  each replica's state. A `WorkerId` is a worker's memory-graph id: fixed
  workers are `0..` in pool order, added ones follow, and none is reused.
  A DP-attention replica's ranks are workers with consecutive ids, the
  first naming the replica (`replica_ranks` lists them); any of them names
  it to the lifecycle calls. A removed worker's GPU is not reused either: a
  replacement comes up on the newest node with free GPUs, or a fresh one.
  To start a pool empty, build it with one replica and drain that at once.
- **Any router.** `Topology::with_pool_router(pool, Box<dyn Router>)`
  installs a caller's router. `route` sees one `WorkerSignal` per active
  worker — every rank of every DP-attention replica, with its `replica` and
  `rank` — and answers with one's `WorkerId`, so it can pin a request to a
  rank of a replica, as Dynamo routes to a `(worker, dp_rank)`. The engine
  also tells the router, in time order and with the time, when replicas
  join (with their rank count, Dynamo's `dp_size`) and leave, where each
  request went, its first token, and when it left the worker (requests
  failed by a replica's removal go with it, unreported); these name the
  rank, as does every KV event, each rank holding its own cache. A router
  that asks (`wants_kv_events`) also hears every block the workers' prefix
  caches store — `(local hash, sequence hash)` under its parent's sequence
  hash — and every eviction, from the cache operations themselves, so its
  own index of each worker stays equal to the cache: Dynamo's KV router
  fed by vLLM's KV events. Requests carry `prompt_local_block_hashes`
  (each block's content alone) beside the cumulative
  `prompt_block_hashes`.

## Time

The engine is a pure event loop over arrivals, worker-ready events and link
drains; the drivers decide what to submit and when. The batch `Simulator`
pumps a `RequestGenerator` through the engine and feeds `RequestTiming`s
into a `MetricsCollector`; the serve driver paces engine time to the wall
clock and streams tokens to HTTP clients.
