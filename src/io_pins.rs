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

#[cfg(test)]
mod tests {
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
