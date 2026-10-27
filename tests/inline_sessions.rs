//! Inline `workload.sessions` is the filesystem-free path the browser build
//! uses. A session file and the same sessions inline must agree, and the
//! existing dataset / session-file error stays worded as it was.

use std::io::Write;

use inference_lab::config::Config;
use inference_lab::simulation::Simulator;
use serde_json::{json, Value};

fn scheduler() -> Value {
    json!({
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 128,
        "policy": "fcfs",
        "enable_chunked_prefill": true,
        "block_size": 16
    })
}

fn sessions() -> Value {
    json!([{
        "id": "s1",
        "steps": [
            {"input": 32, "new": 32, "output": 8, "gap": 0.0},
            {"input": 48, "new": 8, "output": 8, "gap": 0.01}
        ]
    }])
}

fn workload(extra: Value) -> Value {
    let mut workload = json!({
        "arrival_pattern": "poisson",
        "arrival_rate": 1.0,
        "seed": 1,
        "num_sessions": 1,
        "input_len_dist": {"type": "fixed", "value": 1},
        "output_len_dist": {"type": "fixed", "value": 1}
    });
    for (k, v) in extra.as_object().unwrap() {
        workload[k] = v.clone();
    }
    workload
}

fn run(workload: Value) -> Result<inference_lab::metrics::MetricsSummary, String> {
    simulate(json!({
        "hardware": "b200",
        "model": "qwen3.5-4b",
        "scheduler": scheduler(),
        "workload": workload
    }))
}

fn simulate(raw: Value) -> Result<inference_lab::metrics::MetricsSummary, String> {
    let mut config: Config = serde_json::from_value(raw).map_err(|e| e.to_string())?;
    config.finalize();
    let mut sim = Simulator::new(config, None)?;
    sim.run_with_callback(|_| {})?;
    Ok(sim.summary())
}

#[test]
fn inline_sessions_match_a_session_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sessions.jsonl");
    let mut file = std::fs::File::create(&path).unwrap();
    writeln!(file, "{}", serde_json::to_string(&sessions()[0]).unwrap()).unwrap();

    let from_file = run(workload(json!({
        "sessions_path": path.to_str().unwrap()
    })))
    .unwrap();
    let inline = run(workload(json!({
        "sessions": sessions(),
        "sessions_path": "/no/such/sessions.jsonl"
    })))
    .unwrap();

    assert_eq!(from_file.requests.completed, 2);
    assert_eq!(inline.requests.completed, 2);
    assert_eq!(
        inline.latency_metrics.ttft_ms.p50,
        from_file.latency_metrics.ttft_ms.p50
    );
    assert_eq!(inline.sessions.unwrap().started, 1);
}

#[test]
fn empty_inline_sessions_are_rejected() {
    let err = run(workload(json!({"sessions": []}))).unwrap_err();
    assert!(err.contains("workload.sessions is empty"), "{err}");
}

#[test]
fn dataset_and_session_file_keep_their_error() {
    let err = run(workload(json!({
        "dataset_path": "missing.jsonl",
        "sessions_path": "missing-sessions.jsonl"
    })))
    .unwrap_err();
    assert_eq!(err, "workload sets both dataset_path and sessions_path");
}

#[test]
fn dataset_and_inline_sessions_are_rejected() {
    let err = run(workload(json!({
        "dataset_path": "missing.jsonl",
        "sessions": sessions()
    })))
    .unwrap_err();
    assert_eq!(err, "workload sets both dataset_path and sessions");
}

#[test]
fn shipped_workload_files_omit_inline_sessions() {
    let workload = inference_lab::config::WorkloadConfig::from_file("workloads/quick.toml")
        .expect("quick.toml");
    assert!(workload.sessions.is_none());
    assert!(workload.sessions_path.is_none());
}

#[test]
fn tiered_run_reports_stores_links_workers_and_router() {
    let mut w = workload(json!({
        "num_requests": 4,
        "input_len_dist": {"type": "fixed", "value": 64},
        "output_len_dist": {"type": "fixed", "value": 8}
    }));
    // Synthetic, not sessions: drop the session cap so the request cap applies.
    w.as_object_mut().unwrap().remove("num_sessions");
    let summary = simulate(json!({
        "hardware": "b200",
        "model": "qwen3.5-4b",
        "scheduler": scheduler(),
        "memory": {"tiers": ["host_dram"]},
        "replicas": 2,
        "workload": w
    }))
    .unwrap();
    let memory = summary.memory.expect("tiers produce memory metrics");
    assert!(
        memory.stores.iter().any(|s| s.name == "host_dram"),
        "stores: {:?}",
        memory.stores.iter().map(|s| &s.name).collect::<Vec<_>>()
    );
    assert!(!memory.links.is_empty());
    assert!(memory.links.iter().all(|l| l.capacity > 0.0));
    let pool = summary.hbm.pools.first().expect("an hbm pool");
    assert!(!pool.workers.is_empty());
    assert!(pool.workers.iter().any(|w| w.capacity_bytes > 0));
    assert_eq!(summary.router.per_replica.len(), 2);
}

#[test]
fn readme_b200_example_matches_the_browser_golden() {
    // tests/wasm/browser_package.mjs replays this shape. Keep it aligned with
    // the README command: configs/qwen3.6-35b-a3b-fp8.toml --hardware b200
    // --workload workloads/chat-closed-256.toml.
    let deployment =
        inference_lab::config::ModelConfig::from_file("configs/qwen3.6-35b-a3b-fp8.toml")
            .unwrap()
            .deployment(Some("b200"))
            .unwrap();
    let workload =
        inference_lab::config::WorkloadConfig::from_file("workloads/chat-closed-256.toml").unwrap();
    assert_eq!(deployment.hardware.name, "B200");
    assert_eq!(deployment.parallel.tp, 1);
    assert_eq!(deployment.scheduler.max_num_batched_tokens, 16384);
    assert_eq!(deployment.scheduler.max_num_seqs, 4096);
    assert_eq!(deployment.scheduler.block_size, 64);
    assert!((deployment.scheduler.gpu_memory_utilization - 0.8).abs() < 1e-12);
    assert_eq!(workload.num_requests, Some(2000));
    assert_eq!(workload.num_concurrent_users, Some(256));
    assert!((workload.closed_loop_jitter_secs.unwrap() - 0.05).abs() < 1e-12);
    assert_eq!(workload.seed, 7);
    assert!(workload.sessions.is_none());
    assert!(workload.sessions_path.is_none());
}
