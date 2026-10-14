//! How a pool's workers price one forward pass: the datasheet roofline
//! (`model = "roofline"`, the default), or a linear fit to a measured
//! engine:
//!
//! ```toml
//! [step_cost]
//! model = "linear"
//! t0 = 4.0e-3          # seconds per step
//! tp = 2.0e-5          # seconds per prefill token computed in the step
//! td = 1.0e-4          # seconds per decoding request in the step
//! tk = 3.0e-8          # seconds per KV token those decoding requests attend
//! t0p = 6.0e-2         # seconds more for a step that computes any prefill tokens
//! ```

use serde::Deserialize;

use crate::request::Request;

/// The step-time model of a pool's workers. See [`LinearStepCost`] for the
/// calibrated alternative to the roofline.
#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
#[serde(tag = "model", rename_all = "snake_case")]
pub enum StepCostModel {
    /// Every step at peak FLOP rate and HBM bandwidth, collectives added
    /// serially (see [`crate::compute::ComputeEngine`]).
    #[default]
    Roofline,
    /// A linear function of the batch, fitted to a measured engine.
    Linear(LinearStepCost),
}

impl StepCostModel {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            StepCostModel::Roofline => Ok(()),
            StepCostModel::Linear(linear) => linear.validate(),
        }
    }
}

/// A step's wall time as a linear function of its batch, every coefficient
/// in seconds:
///
/// `t = t0 + t0p·[prefill tokens > 0] + tp × prefill tokens + td × decoding requests + tk × KV tokens`
///
/// where *prefill tokens* are the prompt (or recompute) positions the step
/// computes, *decoding requests* the requests in the step past their
/// prefill, and *KV tokens* the context those decoding requests attend: each
/// one's resident positions plus the positions it computes this step. The
/// fitted model already embodies the engine's overheads, so the roofline's
/// `time_correction`, drafter and attention-skew terms do not apply on top.
/// A DP-attention replica's lockstep step is priced on the replica's whole
/// batch, every rank's requests together.
///
/// `t0p` is the fixed cost of a pass that computes prompt positions, over and
/// above `t0`. Engines that capture decode in CUDA graphs but run prefill
/// eagerly (SGLang on DeepSeek-V4, for one) pay a per-pass launch cost on
/// prefill that a decode step does not.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinearStepCost {
    /// Fixed cost of a step.
    #[serde(default)]
    pub t0: f64,
    /// Per prefill token computed.
    #[serde(default)]
    pub tp: f64,
    /// Per decoding request.
    #[serde(default)]
    pub td: f64,
    /// Per KV token attended by a decoding request.
    #[serde(default)]
    pub tk: f64,
    /// Extra fixed cost of a step that computes any prefill tokens.
    #[serde(default)]
    pub t0p: f64,
    /// A DP-attention replica's prefill step in which no rank computes more
    /// than this many prompt tokens runs as a captured CUDA graph and costs
    /// `t0p_graph` instead of `t0p`. 0: no such steps. (Measured on SGLang
    /// GLM-5.2 tp 4 dp 4: steps of up to 2048 tokens a rank take about
    /// 0.19 s, larger ones about 0.58 s, however many ranks prefill.)
    #[serde(default)]
    pub graph_prefill_max_tokens: u32,
    #[serde(default)]
    pub t0p_graph: f64,
}

impl LinearStepCost {
    pub fn validate(&self) -> Result<(), String> {
        for (name, v) in [
            ("t0", self.t0),
            ("tp", self.tp),
            ("td", self.td),
            ("tk", self.tk),
            ("t0p", self.t0p),
            ("t0p_graph", self.t0p_graph),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(format!("step_cost.{name} must be >= 0 (got {v})"));
            }
        }
        if [self.t0, self.tp, self.td, self.tk] == [0.0; 4] {
            return Err("step_cost: a linear model needs a positive coefficient".into());
        }
        Ok(())
    }

    /// Seconds for a step of the given shape.
    pub fn seconds(&self, prefill_tokens: u64, decode_requests: u64, decode_kv_tokens: u64) -> f64 {
        let prefill_pass = if prefill_tokens > 0 { self.t0p } else { 0.0 };
        self.t0
            + prefill_pass
            + self.tp * prefill_tokens as f64
            + self.td * decode_requests as f64
            + self.tk * decode_kv_tokens as f64
    }

    /// Seconds for computing `tokens[i]` positions of each `batch[i]`, the
    /// requests as they stand before the step runs.
    pub fn batch_seconds(&self, batch: &[&Request], tokens: &[u32]) -> f64 {
        debug_assert_eq!(batch.len(), tokens.len(), "one token count per request");
        let (mut prefill, mut decode, mut kv) = (0u64, 0u64, 0u64);
        for (req, &n) in batch.iter().zip(tokens) {
            if req.is_prefill() {
                prefill += n as u64;
            } else {
                decode += 1;
                kv += (req.num_computed_tokens + n) as u64;
            }
        }
        self.seconds(prefill, decode, kv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct Holder {
        #[serde(default)]
        step_cost: StepCostModel,
    }

    #[test]
    fn parses_roofline_by_default_and_linear_when_asked() {
        let h: Holder = toml::from_str("").unwrap();
        assert_eq!(h.step_cost, StepCostModel::Roofline);
        let h: Holder = toml::from_str("[step_cost]\nmodel = \"roofline\"").unwrap();
        assert_eq!(h.step_cost, StepCostModel::Roofline);
        let h: Holder =
            toml::from_str("[step_cost]\nmodel = \"linear\"\nt0 = 0.004\ntp = 2e-5\ntk = 3e-8")
                .unwrap();
        assert_eq!(
            h.step_cost,
            StepCostModel::Linear(LinearStepCost {
                t0: 0.004,
                tp: 2e-5,
                td: 0.0,
                tk: 3e-8,
                t0p: 0.0,
                graph_prefill_max_tokens: 0,
                t0p_graph: 0.0,
            })
        );
        assert!(toml::from_str::<Holder>("[step_cost]\nmodel = \"linear\"\nt1 = 1.0").is_err());
        assert!(toml::from_str::<Holder>("[step_cost]\nmodel = \"cubic\"").is_err());
    }

    #[test]
    fn rejects_negative_or_all_zero_coefficients() {
        let bad = LinearStepCost {
            t0: 0.0,
            tp: -1e-6,
            td: 0.0,
            tk: 0.0,
            t0p: 0.0,
            graph_prefill_max_tokens: 0,
            t0p_graph: 0.0,
        };
        assert!(StepCostModel::Linear(bad).validate().is_err());
        let free = LinearStepCost { tp: 0.0, ..bad };
        assert!(
            StepCostModel::Linear(free).validate().is_err(),
            "steps never cost"
        );
        let prefill_only = LinearStepCost { tp: 1e-6, ..bad };
        assert!(StepCostModel::Linear(prefill_only).validate().is_ok());
        assert!(StepCostModel::Roofline.validate().is_ok());
    }

    #[test]
    fn prices_prefill_tokens_decoding_requests_and_attended_kv() {
        let cost = LinearStepCost {
            t0: 0.003,
            tp: 2e-5,
            td: 1e-4,
            tk: 1e-7,
            t0p: 0.0,
            graph_prefill_max_tokens: 0,
            t0p_graph: 0.0,
        };
        // A 100-token prefill chunk of a 300-token prompt, a fresh decode at
        // 500 resident positions and a 3-wide verify at 1000.
        let prefill = Request::new("p".into(), 0, 0.0, 300, 8);
        let mut decode = Request::new("d".into(), 0, 0.0, 400, 200);
        decode.record_generated_tokens(500, 0.0);
        let mut verify = Request::new("v".into(), 0, 0.0, 900, 200);
        verify.record_generated_tokens(1000, 0.0);
        assert!(prefill.is_prefill() && !decode.is_prefill() && !verify.is_prefill());
        let t = cost.batch_seconds(&[&prefill, &decode, &verify], &[100, 1, 3]);
        // 0.003 + 2e-5·100 + 1e-4·2 + 1e-7·(501 + 1003)
        let expected = 0.003 + 0.002 + 0.0002 + 1e-7 * 1504.0;
        assert!((t - expected).abs() < 1e-15, "{t} vs {expected}");
        assert_eq!(cost.seconds(0, 0, 0), 0.003);
    }

    #[test]
    fn a_prefill_pass_pays_its_fixed_cost_once() {
        let cost = LinearStepCost {
            t0: 0.01,
            tp: 1e-5,
            td: 1e-3,
            tk: 0.0,
            t0p: 0.05,
            graph_prefill_max_tokens: 0,
            t0p_graph: 0.0,
        };
        assert_eq!(cost.seconds(0, 4, 0), 0.01 + 0.004);
        assert!((cost.seconds(1000, 0, 0) - (0.01 + 0.05 + 0.01)).abs() < 1e-15);
        assert!((cost.seconds(2000, 2, 0) - (0.01 + 0.05 + 0.02 + 0.002)).abs() < 1e-15);
    }
}
