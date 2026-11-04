//! Concurrent IO placement's geometry (`-place_ios`): where a movable port may sit — the die
//! perimeter less the blocked pin regions, narrowed by the port's own constraint region and by
//! its mirror partner's — and the projection of a point onto it. Every value is the reference's
//! float.

/// A die edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Edge {
    Left,
    Right,
    Bottom,
    Top,
}

impl Edge {
    /// `isHorizontalEdge`: the bottom and top edges.
    pub fn horizontal(self) -> bool {
        matches!(self, Edge::Bottom | Edge::Top)
    }
    fn index(self) -> usize {
        match self {
            Edge::Left => 0,
            Edge::Right => 1,
            Edge::Bottom => 2,
            Edge::Top => 3,
        }
    }
}

/// `PerimSegment`: an interval along one die edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Seg {
    pub edge: Edge,
    pub lo: f32,
    pub hi: f32,
}

/// The die as floats: `(lx, ly, ux, uy)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Die {
    pub lx: f32,
    pub ly: f32,
    pub ux: f32,
    pub uy: f32,
}

/// `rectToPerimSegment`: a degenerate rectangle as an edge interval — the edge by proximity (a tie
/// goes left / bottom); `None` for a rectangle with area.
pub fn rect_to_seg(die: &Die, (x0, y0, x1, y1): (i32, i32, i32, i32)) -> Option<Seg> {
    if x0 == x1 {
        let x = x0 as f32;
        let edge = if (x - die.lx).abs() <= (x - die.ux).abs() { Edge::Left } else { Edge::Right };
        return Some(Seg { edge, lo: (y0 as f32).min(y1 as f32), hi: (y0 as f32).max(y1 as f32) });
    }
    if y0 == y1 {
        let y = y0 as f32;
        let edge = if (y - die.ly).abs() <= (y - die.uy).abs() { Edge::Bottom } else { Edge::Top };
        return Some(Seg { edge, lo: (x0 as f32).min(x1 as f32), hi: (x0 as f32).max(x1 as f32) });
    }
    None
}

/// `mirrorSegments`: each segment on the opposite edge.
pub fn mirror_segs(segs: &[Seg]) -> Vec<Seg> {
    segs.iter()
        .map(|s| {
            let edge = match s.edge {
                Edge::Left => Edge::Right,
                Edge::Right => Edge::Left,
                Edge::Bottom => Edge::Top,
                Edge::Top => Edge::Bottom,
            };
            Seg { edge, ..*s }
        })
        .collect()
}

/// `intersectSegments`: every same-edge pair's overlap, `a` outer; a single point is kept.
pub fn intersect_segs(a: &[Seg], b: &[Seg]) -> Vec<Seg> {
    let mut out = Vec::new();
    for sa in a {
        for sb in b {
            if sa.edge != sb.edge {
                continue;
            }
            let (lo, hi) = (sa.lo.max(sb.lo), sa.hi.min(sb.hi));
            if lo <= hi {
                out.push(Seg { edge: sa.edge, lo, hi });
            }
        }
    }
    out
}

/// `initIoConstraints`' free perimeter: each edge (left, right, bottom, top) less its blocked
/// intervals, sorted by `(lo, hi)`; a blocked interval that does not reach past the cursor is
/// skipped.
pub fn free_segments(die: &Die, blocked: &[(i32, i32, i32, i32)]) -> Vec<Seg> {
    let mut by_edge: [Vec<(f32, f32)>; 4] = Default::default();
    for &r in blocked {
        if let Some(s) = rect_to_seg(die, r) {
            by_edge[s.edge.index()].push((s.lo, s.hi));
        }
    }
    let mut out = Vec::new();
    for (edge, lo, hi) in [(Edge::Left, die.ly, die.uy), (Edge::Right, die.ly, die.uy), (Edge::Bottom, die.lx, die.ux), (Edge::Top, die.lx, die.ux)] {
        let blk = &mut by_edge[edge.index()];
        blk.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let mut cursor = lo;
        for &(b_lo, b_hi) in blk.iter() {
            let (l, h) = (lo.max(b_lo), hi.min(b_hi));
            if l >= h || h <= cursor {
                continue;
            }
            if l > cursor {
                out.push(Seg { edge, lo: cursor, hi: l });
            }
            cursor = cursor.max(h);
        }
        if cursor < hi {
            out.push(Seg { edge, lo: cursor, hi });
        }
    }
    out
}

/// `projectOntoSegment`: onto the segment's edge, the other coordinate clamped to it.
pub fn project_onto(die: &Die, s: &Seg, x: f32, y: f32) -> (f32, f32) {
    match s.edge {
        Edge::Left => (die.lx, y.max(s.lo).min(s.hi)),
        Edge::Right => (die.ux, y.max(s.lo).min(s.hi)),
        Edge::Bottom => (x.max(s.lo).min(s.hi), die.ly),
        Edge::Top => (x.max(s.lo).min(s.hi), die.uy),
    }
}

/// `nearestSegment`: the first segment whose projection is strictly nearest (squared float
/// distance), with that projection.
pub fn nearest(die: &Die, segs: &[Seg], x: f32, y: f32) -> Option<(Seg, (f32, f32))> {
    let mut best: Option<(Seg, (f32, f32))> = None;
    let mut best_dist = f32::MAX;
    for s in segs {
        let p = project_onto(die, s, x, y);
        let (dx, dy) = (p.0 - x, p.1 - y);
        let dist = dx * dx + dy * dy;
        if dist < best_dist {
            best_dist = dist;
            best = Some((*s, p));
        }
    }
    best
}

/// `slotIndexAt`: where `q` falls along sorted slot coordinates, as a fractional slot index —
/// linear between neighbours, clamped to the first and last slot. Of equal slots (two layers at one
/// position) it lands on the last.
pub fn slot_index_at(slots: &[f32], q: f32) -> f32 {
    let last = *slots.last().expect("a slot");
    if q <= slots[0] {
        return 0.0;
    }
    if q >= last {
        return (slots.len() - 1) as f32;
    }
    let hi = slots.partition_point(|&s| s <= q);
    let lo = hi - 1;
    lo as f32 + (q - slots[lo]) / (slots[hi] - slots[lo])
}

/// `slotCoordinateAt`: the inverse — the coordinate at a fractional slot index, clamped to the
/// slot range.
pub fn slot_coordinate_at(slots: &[f32], g: f32) -> f32 {
    let c = g.clamp(0.0, (slots.len() - 1) as f32);
    let i = c as usize;
    if i + 1 >= slots.len() {
        return *slots.last().expect("a slot");
    }
    slots[i] + (c - i as f32) * (slots[i + 1] - slots[i])
}

/// `unrollRingAtWidestGap`: open a ring of `n` slots in the middle of the widest gap between the
/// sorted pins `(slot, io index)`, so neighbours on the ring stay neighbours on the line. The pins
/// are rotated to start after the cut, those that wrapped moved one lap on (+n). Returns the slot
/// the line starts at — the gap's middle, rounded down; it can be negative.
pub fn unroll_ring_at_widest_gap(pins: &mut [(f32, usize)], n: i32) -> f32 {
    let mut widest = n as f32 - pins.last().expect("a pin").0 + pins[0].0;
    let mut cut = 0;
    for r in 1..pins.len() {
        let gap = pins[r].0 - pins[r - 1].0;
        if gap > widest {
            widest = gap;
            cut = r;
        }
    }
    let gap_from = if cut == 0 { pins.last().expect("a pin").0 - n as f32 } else { pins[cut - 1].0 };
    let at = pins[cut].0;
    pins.rotate_left(cut);
    for p in pins.iter_mut() {
        if p.0 < at {
            p.0 += n as f32;
        }
    }
    ((gap_from + at) / 2.0).floor()
}

/// `poolAdjacentViolators`: replace `z` by the nearest non-decreasing sequence (least squares,
/// pooling each violating run into its mean, in double), every value clamped to `[lo, hi]`.
pub fn pool_adjacent_violators(z: &mut [f32], lo: f32, hi: f32) {
    let mut blocks: Vec<(f64, i32)> = Vec::with_capacity(z.len());
    for &v in z.iter() {
        blocks.push((f64::from(v), 1));
        while blocks.len() > 1 {
            let (lm, lc) = blocks[blocks.len() - 1];
            let (pm, pc) = blocks[blocks.len() - 2];
            if pm <= lm {
                break;
            }
            let count = pc + lc;
            let k = blocks.len() - 2;
            blocks[k] = ((pm * f64::from(pc) + lm * f64::from(lc)) / f64::from(count), count);
            blocks.pop();
        }
    }
    let mut r = 0;
    for (mean, count) in blocks {
        let fit = (mean as f32).clamp(lo, hi);
        for _ in 0..count {
            z[r] = fit;
            r += 1;
        }
    }
}

/// `fitSpacing`: move the sorted pins as little as possible so they sit one slot apart within the
/// `total` slots from `lo` — whole slots when they fit; with more pins than slots, spread evenly
/// over them (fractional).
pub fn fit_spacing(pins: &[(f32, usize)], lo: f32, total: i32) -> Vec<f32> {
    let n = pins.len() as i32;
    let fits = n <= total;
    let step = if fits { 1.0f32 } else { (total - 1) as f32 / (n - 1) as f32 };
    let mut z: Vec<f32> = pins.iter().enumerate().map(|(r, p)| p.0 - r as f32 * step).collect();
    pool_adjacent_violators(&mut z, lo, lo + if fits { (total - n) as f32 } else { 0.0 });
    for (r, v) in z.iter_mut().enumerate() {
        *v = if fits { v.round() } else { *v } + r as f32 * step;
    }
    z
}

#[cfg(test)]
mod tests {
    // upstream src/gpl/test/place_ios_test.cc (IoSlotFit), the same cases

    fn pins_at(slots: &[f32]) -> Vec<(f32, usize)> {
        slots.iter().enumerate().map(|(i, &s)| (s, i)).collect()
    }

    #[test]
    fn slot_index_interpolates_clamps_and_takes_the_last_equal() {
        let slots = [0.0, 10.0, 20.0, 40.0];
        assert_eq!(slot_index_at(&slots, -5.0), 0.0);
        assert_eq!(slot_index_at(&slots, 50.0), 3.0);
        assert_eq!(slot_index_at(&slots, 10.0), 1.0);
        assert_eq!(slot_index_at(&slots, 30.0), 2.5);
        assert_eq!(slot_index_at(&[0.0, 10.0, 10.0, 20.0], 10.0), 2.0);
        for q in [0.0f32, 5.0, 10.0, 30.0, 40.0] {
            assert_eq!(slot_coordinate_at(&slots, slot_index_at(&slots, q)), q);
        }
        assert_eq!(slot_coordinate_at(&slots, -1.0), 0.0);
        assert_eq!(slot_coordinate_at(&slots, 7.0), 40.0);
    }

    #[test]
    fn pool_adjacent_violators_averages_and_clamps() {
        let mut z = vec![3.0, 1.0, 2.0];
        pool_adjacent_violators(&mut z, 0.0, 10.0);
        assert_eq!(z, vec![2.0, 2.0, 2.0]);
        let mut z = vec![5.0];
        pool_adjacent_violators(&mut z, 0.0, 3.0);
        assert_eq!(z, vec![3.0]);
    }

    #[test]
    fn fit_spacing_spreads_keeps_and_stays_inside() {
        assert_eq!(fit_spacing(&pins_at(&[5.0, 5.0, 5.0, 9.0]), 0.0, 20), vec![4.0, 5.0, 6.0, 9.0]);
        assert_eq!(fit_spacing(&pins_at(&[1.0, 3.0, 7.0]), 0.0, 20), vec![1.0, 3.0, 7.0]);
        assert_eq!(fit_spacing(&pins_at(&[9.0, 9.0, 9.0]), 0.0, 10), vec![7.0, 8.0, 9.0]);
        let fit = fit_spacing(&pins_at(&[0.0; 4]), 0.0, 3);
        assert_eq!(fit.len(), 4);
        for (r, v) in fit.iter().enumerate() {
            assert!((v - r as f32 * 2.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn unroll_cuts_the_ring_at_its_widest_gap() {
        let mut pins = pins_at(&[0.0, 1.0, 98.0, 99.0]);
        assert_eq!(unroll_ring_at_widest_gap(&mut pins, 100), 49.0);
        assert_eq!(pins, vec![(98.0, 2), (99.0, 3), (100.0, 0), (101.0, 1)]);
        let mut pins = pins_at(&[0.0; 5]);
        let start = unroll_ring_at_widest_gap(&mut pins, 100);
        assert_eq!(start, -50.0);
        assert_eq!(fit_spacing(&pins, start, 100), vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    }

    use super::*;

    const DIE: Die = Die { lx: 0.0, ly: 0.0, ux: 100.0, uy: 80.0 };

    /// Rule (`rectToPerimSegment`): the edge is the nearer one, a tie the left; a box with area is
    /// no segment.
    #[test]
    fn degenerate_rect_names_the_nearer_edge() {
        assert_eq!(rect_to_seg(&DIE, (50, 10, 50, 30)).map(|s| s.edge), Some(Edge::Left));
        assert_eq!(rect_to_seg(&DIE, (51, 30, 51, 10)), Some(Seg { edge: Edge::Right, lo: 10.0, hi: 30.0 }));
        assert_eq!(rect_to_seg(&DIE, (0, 80, 100, 80)).map(|s| s.edge), Some(Edge::Top));
        assert_eq!(rect_to_seg(&DIE, (1, 1, 2, 2)), None);
    }

    /// Rule (`initIoConstraints`): the blocked intervals are cut out of each edge in sorted order;
    /// a fully blocked edge leaves nothing.
    #[test]
    fn free_perimeter_cuts_the_blocked_intervals() {
        let free = free_segments(&DIE, &[(0, 80, 100, 80), (0, 20, 0, 30), (0, 0, 0, 10)]);
        assert_eq!(free[0], Seg { edge: Edge::Left, lo: 10.0, hi: 20.0 });
        assert_eq!(free[1], Seg { edge: Edge::Left, lo: 30.0, hi: 80.0 });
        assert_eq!(free.iter().filter(|s| s.edge == Edge::Top).count(), 0);
        assert_eq!(free.len(), 4);
    }

    /// Rule (`nearestSegment`): strictly nearer wins, so of two equally near the first is kept.
    #[test]
    fn nearest_keeps_the_first_of_a_tie() {
        let segs = [Seg { edge: Edge::Left, lo: 0.0, hi: 80.0 }, Seg { edge: Edge::Bottom, lo: 0.0, hi: 100.0 }];
        let (s, p) = nearest(&DIE, &segs, 5.0, 5.0).expect("a segment");
        assert_eq!((s.edge, p), (Edge::Left, (0.0, 5.0)));
    }
}
