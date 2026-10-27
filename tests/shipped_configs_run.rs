//! Every model config shipped under configs/ must run on every one of its
//! hardware entries: the model's weights fit the replica with room for KV,
//! the hardware has a FLOP rate for every precision the model streams, and a
//! handful of requests prefill, decode and complete. A catalog entry with a
//! wrong parameter count, a missing rate or a layout that cannot hold its
//! weights fails here by name. Speculative decoding is switched off: some
//! configs read acceptance traces kept outside the repository, and the
//! drafter is not what this checks.

use inference_lab::config::{ModelConfig, WorkloadConfig};
use inference_lab::simulation::Simulator;

const WORKLOAD: &str = r#"
arrival_pattern = "closed_loop"
num_concurrent_users = 2
num_requests = 4
seed = 1
input_len_dist = { type = "fixed", value = 2048 }
output_len_dist = { type = "fixed", value = 8 }
"#;

#[test]
fn every_shipped_config_runs_on_each_of_its_hardware_entries() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("configs");
    let mut paths: Vec<_> = std::fs::read_dir(&root)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("toml"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty());
    let mut failures = Vec::new();
    let mut runs = 0;
    for p in &paths {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let cfg = ModelConfig::from_file(p).unwrap_or_else(|e| panic!("{e}"));
        for hw in cfg.hardware_names() {
            let workload: WorkloadConfig = toml::from_str(WORKLOAD).unwrap();
            let mut deployment = match cfg.deployment(Some(hw)) {
                Ok(d) => d,
                Err(e) => {
                    failures.push(format!("{name} [{hw}]: {e}"));
                    continue;
                }
            };
            deployment.speculative = None;
            let cluster = deployment.cluster();
            let weights = cluster.resident_weight_bytes(&deployment.model);
            let kv = cluster.kv_cache_capacity(&deployment.scheduler, weights);
            if kv == 0 {
                failures.push(format!(
                    "{name} [{hw}]: {weights} B of weights leave no KV at \
                     gpu_memory_utilization {}",
                    deployment.scheduler.gpu_memory_utilization
                ));
                continue;
            }
            let mut sim = match Simulator::new(deployment.with_workload(workload), None) {
                Ok(s) => s,
                Err(e) => {
                    failures.push(format!("{name} [{hw}]: {e}"));
                    continue;
                }
            };
            if let Err(e) = sim.run_with_callback(|_| {}) {
                failures.push(format!("{name} [{hw}]: {e}"));
                continue;
            }
            let s = sim.summary();
            let tps = s.throughput_metrics.output_tokens_per_sec;
            if s.requests.completed != 4
                || s.requests.rejected != 0
                || !(tps.is_finite() && tps > 0.0)
            {
                failures.push(format!(
                    "{name} [{hw}]: completed {} of {}, rejected {}, {tps} output tok/s",
                    s.requests.completed, s.requests.total, s.requests.rejected
                ));
            }
            runs += 1;
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    assert!(runs > 0);
}

/// Every catalog model and hardware preset is exercised by the test above:
/// some shipped config deploys each model, and some config entry runs on
/// each hardware preset. A preset with no published dense FLOP rate is
/// exempt: a roofline deployment cannot name it.
#[test]
fn every_catalog_entry_appears_in_a_shipped_config() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("configs");
    let mut models = std::collections::BTreeSet::new();
    let mut hardware = std::collections::BTreeSet::new();
    for e in std::fs::read_dir(&root).unwrap().flatten() {
        let p = e.path();
        if p.extension().and_then(|x| x.to_str()) != Some("toml") {
            continue;
        }
        let raw: toml::Table = toml::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        if let Some(m) = raw.get("model").and_then(|m| m.as_str()) {
            models.insert(m.to_string());
        }
        let cfg = ModelConfig::from_file(&p).unwrap_or_else(|e| panic!("{e}"));
        hardware.extend(cfg.hardware_names().into_iter().map(String::from));
    }
    // A preset with no published dense FLOP rate cannot back a roofline
    // deployment, so no file under configs/ names it. Memory figures are
    // still part of the catalog. The moment a rate is filled in, this list
    // must be empty and a config has to reference the preset.
    let no_flop_rate: Vec<&str> = inference_lab::catalog::hardware_names()
        .into_iter()
        .filter(|n| {
            let hw = inference_lab::catalog::hardware(n).unwrap();
            hw.flops_fp4.is_none()
                && hw.flops_fp8.is_none()
                && hw.flops_bf16.is_none()
                && hw.flops_fp16.is_none()
        })
        .collect();
    assert_eq!(
        no_flop_rate,
        vec!["m5-pro"],
        "only the preset with no published dense rates may lack a config"
    );
    let unused: Vec<String> = inference_lab::catalog::model_names()
        .into_iter()
        .filter(|n| !models.contains(*n))
        .map(|n| format!("model {n}"))
        .chain(
            inference_lab::catalog::hardware_names()
                .into_iter()
                .filter(|n| !hardware.contains(*n) && !no_flop_rate.contains(n))
                .map(|n| format!("hardware {n}")),
        )
        .collect();
    assert!(
        unused.is_empty(),
        "catalog entries no config under configs/ uses: {}",
        unused.join(", ")
    );
}
