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
        TdState { overflows: v, fired: vec![false; n], net_weight_max: 5.0, nets_percentage: 10.0, keep_resize_below_overflow: 1.0, repair_timing: false, repair_tns_end_percent: 1.0, counter: 0, enabled: true }
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

/// `weight(slack)` of `executeTimingDriven`: from 1 at the worst set's best slack up to the
/// maximum at its worst (1 when they are equal), for a net strictly below that best slack.
pub fn timing_weight(slack: f32, slack_min: f32, slack_max: f32, net_weight_max: f32) -> Option<f32> {
    if slack >= slack_max {
        return None;
    }
    Some(if slack_max == slack_min { 1.0 } else { 1.0 + (net_weight_max - 1.0) * (slack_max - slack) / (slack_max - slack_min) })
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
