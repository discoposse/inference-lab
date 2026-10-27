//! Hardware and model presets that ship with the crate.
//!
//! The source of truth is `catalog/hardware/*.toml` and
//! `catalog/models/*.toml` in the repository; the build script embeds them,
//! so `catalog::hardware("b200")` and `catalog::model("deepseek-v4-flash")`
//! work in any build, wasm included, and a serving config can name an entry
//! (`hardware = "b200"`, `model = "deepseek-v4-flash"`) instead of copying
//! its numbers.

use crate::config::{HardwareConfig, ModelSpec};

include!(concat!(env!("OUT_DIR"), "/catalog_entries.rs"));

fn source(kind: &str, name: &str) -> Option<&'static str> {
    ENTRIES
        .iter()
        .find(|(k, n, _)| *k == kind && *n == name)
        .map(|(_, _, src)| *src)
}

fn names(kind: &str) -> Vec<&'static str> {
    ENTRIES
        .iter()
        .filter(|(k, _, _)| *k == kind)
        .map(|(_, n, _)| *n)
        .collect()
}

/// Names of the shipped hardware presets, sorted.
pub fn hardware_names() -> Vec<&'static str> {
    names("hardware")
}

/// Names of the shipped model presets, sorted.
pub fn model_names() -> Vec<&'static str> {
    names("models")
}

/// A hardware preset by name.
pub fn hardware(name: &str) -> Result<HardwareConfig, String> {
    let src = source("hardware", name).ok_or_else(|| {
        format!(
            "unknown hardware preset {name:?}; available: {}",
            hardware_names().join(", ")
        )
    })?;
    let hw: HardwareConfig =
        toml::from_str(src).map_err(|e| format!("catalog hardware {name:?}: {e}"))?;
    if let Some(m) = &hw.memory {
        m.validate()
            .map_err(|e| format!("catalog hardware {name:?}: {e}"))?;
    }
    Ok(hw)
}

/// JSON document of the shipped presets, for a browser or any other
/// JavaScript caller.
///
/// * `gpus` and `hardware` are the same list of accelerator presets (the
///   files under `catalog/hardware`). `gpus` is the name a browser catalog
///   reads; `hardware` is the config field those presets fill.
/// * `models` is the model presets under `catalog/models`.
///
/// Each entry is `{ "id", "preset" }` and `preset` is the parsed TOML
/// document, so the numbers are the ones the simulator loads.
pub fn export_value() -> serde_json::Value {
    let hardware = preset_list("hardware");
    serde_json::json!({
        "gpus": hardware.clone(),
        "hardware": hardware,
        "models": preset_list("models"),
    })
}

/// [`export_value`] as pretty-printed JSON.
pub fn export_json() -> String {
    let mut json =
        serde_json::to_string_pretty(&export_value()).expect("catalog presets are valid JSON");
    json.push('\n');
    json
}

fn preset_list(kind: &str) -> Vec<serde_json::Value> {
    ENTRIES
        .iter()
        .filter(|(k, _, _)| *k == kind)
        .map(|(_, name, src)| {
            let preset: toml::Value = toml::from_str(src)
                .unwrap_or_else(|e| panic!("catalog {kind} {name:?} is not valid TOML: {e}"));
            serde_json::json!({
                "id": name,
                "preset": preset,
            })
        })
        .collect()
}

/// A model preset by name, validated.
pub fn model(name: &str) -> Result<ModelSpec, String> {
    let src = source("models", name).ok_or_else(|| {
        format!(
            "unknown model preset {name:?}; available: {}",
            model_names().join(", ")
        )
    })?;
    let spec: ModelSpec =
        toml::from_str(src).map_err(|e| format!("catalog model {name:?}: {e}"))?;
    spec.validate()?;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_shipped_entry_parses_and_validates() {
        assert!(!hardware_names().is_empty());
        assert!(!model_names().is_empty());
        for n in hardware_names() {
            let hw = hardware(n).unwrap_or_else(|e| panic!("{e}"));
            // A single-SoC preset whose public spec publishes no tier
            // topology has no `[memory]` graph. Every other preset offers
            // at least one store a GPU can reach, and a route to the network
            // for hand-offs.
            let Some(m) = hw.memory.as_ref() else {
                assert_eq!(n, "m5-pro", "{n}: shipped preset is missing [memory]");
                assert!(
                    hw.flops_fp4.is_none()
                        && hw.flops_fp8.is_none()
                        && hw.flops_bf16.is_none()
                        && hw.flops_fp16.is_none(),
                    "{n}: do not invent dense FLOP rates"
                );
                assert!(hw.fabric.is_none(), "{n}: do not invent a fabric");
                continue;
            };
            assert!(
                m.stores.iter().any(|s| m.gpu_reaches(&s.name)),
                "{n}: no store reachable from a gpu"
            );
            assert!(m.gpu_reaches("network"), "{n}: no route to the network");
        }
        for n in model_names() {
            model(n).unwrap_or_else(|e| panic!("{e}"));
        }
    }

    #[test]
    fn unknown_names_list_the_alternatives() {
        let e = hardware("nope").unwrap_err();
        assert!(e.contains("available"));
        assert!(model("nope").is_err());
    }

    #[test]
    fn export_lists_every_gpu_hardware_and_model_preset() {
        let doc: serde_json::Value = serde_json::from_str(&export_json()).unwrap();
        let ids = |key| {
            doc[key]
                .as_array()
                .unwrap_or_else(|| panic!("{key} is not a list"))
                .iter()
                .map(|e| e["id"].as_str().unwrap().to_string())
                .collect::<Vec<_>>()
        };
        let owned = |names: Vec<&str>| names.into_iter().map(str::to_string).collect::<Vec<_>>();
        assert_eq!(ids("gpus"), ids("hardware"));
        assert_eq!(ids("hardware"), owned(hardware_names()));
        assert_eq!(ids("models"), owned(model_names()));
        let m5 = doc["hardware"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["id"] == "m5-pro")
            .expect("m5-pro preset");
        let preset = &m5["preset"];
        assert_eq!(preset["name"], "M5 Pro");
        let bandwidth = preset["memory_bandwidth"].as_f64().unwrap();
        assert!((bandwidth - 307e9).abs() < 1.0, "{bandwidth}");
        let capacity = preset["memory_capacity"].as_i64().unwrap();
        assert_eq!(capacity, 24_000_000_000);
        assert!(preset.get("flops_bf16").is_none());
        assert!(preset.get("flops_fp8").is_none());
        assert!(preset.get("memory").is_none());
        assert!(preset.get("fabric").is_none());
    }
}
