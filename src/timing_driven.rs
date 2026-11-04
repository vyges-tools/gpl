//! Timing-driven placement's state (`TimingBase`): the overflows at which the nets are reweighted,
//! the weights from the resizer's slacks, and what the resizer reports at a reweight.

use crate::placer_base::Design;

/// What the resizer reports at a reweight (`Resizer::findResizeSlacks`): the lines it printed
/// (repair_design's table, the rebuffering's), each net's slack at its driver in the block's net
/// order (its id order: the tie-break of the worst nets), the callbacks its edits raised, and the
/// design as it left it, re-read.
#[derive(Debug, Clone)]
pub struct ResizeReport {
    pub lines: Vec<String>,
    /// The database callbacks the edits raised, in order (the placer's own observer; each net
    /// event ends with the net's id).
    pub events: Vec<String>,
    /// Each net's database id in the block's net order, before the edits and after: the gnets'
    /// identity (`dbNet*` in the reference), which a rename keeps.
    pub net_ids_before: Vec<u32>,
    pub net_ids_after: Vec<u32>,
    pub slacks: Vec<(String, f32)>,
    pub design: Design,
}

/// `PlaceOptions`' timing-driven fields and `TimingBase`'s state.
#[derive(Debug, Clone)]
pub struct TdState {
    /// `timingNetWeightOverflow_`: percents, decreasing, each fired once (`timingOverflowChk_`).
    pub overflows: Vec<i32>,
    pub fired: Vec<bool>,
    pub net_weight_max: f32,
    pub nets_percentage: f32,
    pub keep_resize_below_overflow: f32,
    pub repair_timing: bool,
    pub repair_tns_end_percent: f32,
    /// `timingDrivenIterCounter`.
    pub counter: i32,
    /// `timingDrivenMode`: cleared when the slacks give out (GPL-0102 / GPL-0105).
    pub enabled: bool,
    /// `-virtual_cts`: the skew fraction (clamped to [0, 1]); and `virtual_cts_count`.
    pub virtual_cts: Option<f32>,
    pub virtual_cts_count: i32,
}

impl TdState {
    /// `setTimingNetWeightOverflows`: the overflows sorted decreasing, each once.
    pub fn new(overflows: &[i32]) -> TdState {
        let mut v: Vec<i32> = Vec::new();
        let mut sorted = overflows.to_vec();
        sorted.sort_by(|a, b| b.cmp(a));
        for o in sorted {
            if !v.contains(&o) {
                v.push(o);
            }
            v.sort_by(|a, b| b.cmp(a));
        }
        let n = v.len();
        TdState { overflows: v, fired: vec![false; n], net_weight_max: 5.0, nets_percentage: 10.0, keep_resize_below_overflow: 1.0, repair_timing: false, repair_tns_end_percent: 1.0, counter: 0, enabled: true, virtual_cts: None, virtual_cts_count: 0 }
    }

    /// `isTimingNetWeightOverflow`: the overflow as a rounded percent; true when it has fallen
    /// below an overflow not yet fired (each fires once, the larger ones first).
    pub fn is_overflow_trigger(&mut self, overflow: f32) -> bool {
        let p = (overflow * 100.0).round() as i32;
        if self.overflows.is_empty() || p > self.overflows[0] {
            return false;
        }
        let mut need = false;
        for i in 0..self.overflows.len() {
            if self.overflows[i] > p {
                if !self.fired[i] {
                    self.fired[i] = true;
                    need = true;
                }
                continue;
            }
            return need;
        }
        need
    }
}

/// `resizeWorstSlackNets`: the nets by slack, ties by block order (`dbNet::getId`), the top
/// `ceil(n · percent / 100)`. Indices into `slacks`.
pub fn worst_slack_nets(slacks: &[(String, f32)], percent: f32) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..slacks.len()).collect();
    idx.sort_by(|&a, &b| slacks[a].1.partial_cmp(&slacks[b].1).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    let n = (f64::from(slacks.len() as f32 * percent) / 100.0).ceil() as usize;
    idx.truncate(n.min(idx.len()));
    idx
}

/// OpenSTA `fuzzyInf`: equal, within a relative 1e-6, to plus or minus `INF` (1e30).
pub fn fuzzy_inf(v: f32) -> bool {
    let eq = |a: f32, b: f32| a == b || (a - b).abs() < 1e-6 * a.abs().max(b.abs());
    eq(v, 1e30) || eq(v, -1e30)
}

/// `kSlackMinPercent`: the weight ramp's worst end sits this percentage of the whole design's
/// constrained nets in from the worst net.
pub const SLACK_MIN_PERCENT: f64 = 0.2;

/// `slackRampAnchorIndex` (upstream 1ec9452b345): the index in the worst set (sorted by
/// increasing slack) whose slack anchors the ramp's worst end. The worst set holds
/// `nets_percentage` % of the nets, so the anchor is a fixed share of the design however large
/// that set is; a small design anchors on its worst net. Never past the set's last net.
pub fn slack_ramp_anchor_index(net_count: usize, nets_percentage: f32) -> usize {
    if nets_percentage <= 0.0 {
        return 0;
    }
    let anchor = (net_count as f64 * SLACK_MIN_PERCENT / f64::from(nets_percentage)) as usize;
    (net_count - 1).min(anchor)
}

/// `computeTimingNetWeight`: 1 plus the extra weight times the net's criticality — where its
/// slack sits between the ramp's best end (0) and its worst end (1), clamped to that range, so
/// a net worse than the anchor saturates. A flat or inverted ramp weighs every net 1.
pub fn compute_timing_net_weight(net_slack: f32, slack_min: f32, slack_max: f32, net_weight_max: f32) -> f32 {
    if slack_max <= slack_min {
        return 1.0;
    }
    let criticality = ((slack_max - net_slack) / (slack_max - slack_min)).clamp(0.0, 1.0);
    1.0 + (net_weight_max - 1.0) * criticality
}

/// fmt's `{}` for a float: the shortest form, an exponent from 1e16 or below 1e-4.
pub fn fmt_float(v: f32) -> String {
    let a = v.abs();
    if v != 0.0 && !(1e-4..1e16).contains(&a) {
        let s = format!("{v:e}");
        let (m, e) = s.split_once('e').unwrap_or((&s, "0"));
        let e: i32 = e.parse().unwrap_or(0);
        return format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
    }
    format!("{v}")
}

#[cfg(test)]
mod tests {
    use super::*;

    // upstream src/gpl/test/timing_base_test.cpp, the same cases

    fn worst_count(total: usize, pct: f32) -> usize {
        total.min((total as f64 * f64::from(pct) / 100.0).ceil() as usize)
    }

    #[test]
    fn ramp_anchor_is_a_share_of_the_design() {
        for pct in [10.0f32, 20.0, 25.0, 50.0, 70.0, 100.0] {
            assert_eq!(slack_ramp_anchor_index(worst_count(150_000, pct), pct), 300, "nets_percentage {pct}");
        }
        assert_eq!(slack_ramp_anchor_index(150_000, 100.0), 300);
        assert_eq!(slack_ramp_anchor_index(50_000, 50.0), 200);
        assert_eq!(slack_ramp_anchor_index(1, 100.0), 0);
        assert_eq!(slack_ramp_anchor_index(284, 100.0), 0);
        assert_eq!(slack_ramp_anchor_index(29, 10.0), 0);
        assert_eq!(slack_ramp_anchor_index(1000, 0.0), 0);
        for n in 1..=1000 {
            for pct in [0.1f32, 1.0, 10.0, 100.0] {
                assert!(slack_ramp_anchor_index(n, pct) < n, "net_count {n} nets_percentage {pct}");
            }
        }
    }

    #[test]
    fn unconstrained_is_fuzzy_inf() {
        assert!(fuzzy_inf(1e30) && fuzzy_inf(-1e30) && fuzzy_inf(1e30 * (1.0 - 5e-7)));
        assert!(!fuzzy_inf(1e30 * (1.0 - 2e-6)) && !fuzzy_inf(1.38e-9) && !fuzzy_inf(0.0));
    }

    #[test]
    fn net_weight_ramps_and_saturates() {
        let w = |s| compute_timing_net_weight(s, -1.0, 1.0, 5.0);
        for (s, e) in [(-1.0, 5.0), (1.0, 1.0), (0.0, 3.0), (0.5, 2.0), (-2.0, 5.0), (-100.0, 5.0), (2.0, 1.0)] {
            assert!((w(s) - e).abs() < 1e-5, "slack {s}");
        }
        assert_eq!(compute_timing_net_weight(1.0, 1.0, 1.0, 5.0), 1.0);
        // the criticality is divided out BEFORE it scales: (4 * 1/3) is not 4/3 in f32 for
        // every operand, and the reference rounds this way
        let c = (1.0f32 - 0.3) / (1.0 + 0.7);
        assert_eq!(compute_timing_net_weight(0.3, -0.7, 1.0, 5.0).to_bits(), (1.0 + 4.0 * c).to_bits());
    }

    /// Rule (`isTimingNetWeightOverflow`): a percent above the first overflow never fires; one
    /// below fires every overflow above it at once, once each.
    #[test]
    fn overflow_triggers_fire_once() {
        let mut s = TdState::new(&[20, 64]);
        assert_eq!(s.overflows, vec![64, 20]);
        assert!(!s.is_overflow_trigger(0.70));
        assert!(s.is_overflow_trigger(0.633));
        assert!(!s.is_overflow_trigger(0.50));
        assert!(s.is_overflow_trigger(0.192));
        assert!(!s.is_overflow_trigger(0.10));
    }

    /// Rule (`resizeWorstSlackNets`): ties by block order; the share rounds up.
    #[test]
    fn worst_nets_tie_by_block_order() {
        let s = vec![("a".to_string(), 2.0), ("b".to_string(), 1.0), ("c".to_string(), 1.0), ("d".to_string(), 3.0)];
        assert_eq!(worst_slack_nets(&s, 50.0), vec![1, 2]);
        assert_eq!(worst_slack_nets(&s, 10.0), vec![1]);
    }

    /// Rule (fmt `{}`): the shortest round-trip, exponent for small values.
    #[test]
    fn float_format() {
        assert_eq!(fmt_float(1.379_749_5e-9), "1.3797495e-09");
        assert_eq!(fmt_float(0.7), "0.7");
        assert_eq!(fmt_float(1.0), "1");
    }
}
