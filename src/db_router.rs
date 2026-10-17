//! The router routability-driven placement measures with, over the design database: the
//! instances written where the placer has them, and the global router's grid — its tile size and
//! its edges' capacity and reduction after adjustment — set up as `initFastRoute` sets it up.

use crate::log::{refuse, Log, Res};
use crate::route_base::{GrtCongestion, GrtLayer, Router, RudyInputs};
use vyges_grt::adjust::EdgeState;
use vyges_grt::capacity::Direction;
use vyges_grt::global_route::{route_design, setup_adjust, setup_tech, RouteOptions};
use vyges_grt::{update_db_congestion, CongestionEdge, DbCongestionLayer};
use vyges_opendb::Db;

/// The router's 3D edges as they stand: grid size, each router layer's direction, and the
/// horizontal and vertical edges `[l][y][x]`.
struct Edges {
    x_grid: i32,
    y_grid: i32,
    horizontal: Vec<bool>,
    h3: Vec<EdgeState>,
    v3: Vec<EdgeState>,
}

pub struct DbRouter<'a> {
    pub db: &'a mut Db,
    edges: Option<Edges>,
}

impl<'a> DbRouter<'a> {
    pub fn new(db: &'a mut Db) -> DbRouter<'a> {
        DbRouter { db, edges: None }
    }
}

thread_local! {
    static LUT: vyges_stt::flute::lut::Lut = vyges_stt::flute::lut::load_tables(vyges_stt::flute::lut::MAX_LUT_DEGREE).expect("flute tables");
}

fn to_rsmt(t: &vyges_stt::Tree) -> vyges_grt::RsmtTree {
    vyges_grt::RsmtTree { deg: t.deg, length: t.length, branch: t.branch.iter().map(|b| vyges_grt::Branch { x: b.x, y: b.y, n: b.n }).collect() }
}

/// `SteinerTreeBuilder::makeSteinerTree(x, y, drvr, alpha)`.
fn stt(x: &[i32], y: &[i32], drvr: usize, alpha: f32) -> vyges_grt::RsmtTree {
    LUT.with(|lut| to_rsmt(&vyges_stt::make_steiner_tree(lut, x, y, drvr, alpha).0.expect("a Steiner tree")))
}

/// FastRoute's pre-sorted FLUTE.
fn flutes(xs: &[i32], ys: &[i32], s: &[usize], acc: i32) -> vyges_grt::RsmtTree {
    LUT.with(|lut| to_rsmt(&vyges_stt::flute::medium_degree::flutes_all_degree_acc(lut, xs.len(), xs, ys, s, acc).expect("flute")))
}

fn e(e: impl std::fmt::Display) -> crate::log::Stop {
    crate::log::Stop::Refused(e.to_string())
}

impl Router for DbRouter<'_> {
    fn write_locations(&mut self, _log: &mut Log, locations: &[(usize, i32, i32)]) -> Res<()> {
        let names = self.db.block_get_insts();
        for &(di, x, y) in locations {
            self.db.inst_set_placement_status(&names[di], "PLACED").map_err(e)?;
            self.db.set_inst_location(&names[di], x, y).map_err(e)?;
        }
        Ok(())
    }

    /// `reportHpwlMetric`: the signal nets' terminal boxes (no supply or special net), summed.
    fn report_hpwl(&mut self, log: &mut Log) -> Res<()> {
        let mut hpwl = 0i64;
        for net in self.db.block_get_nets() {
            let sig = self.db.net_get_sig_type(&net);
            if matches!(sig.as_str(), "POWER" | "GROUND") || self.db.net_is_special(&net) {
                continue;
            }
            let (x0, y0, x1, y1) = (self.db.net_get_term_b_box_x_min(&net), self.db.net_get_term_b_box_y_min(&net), self.db.net_get_term_b_box_x_max(&net), self.db.net_get_term_b_box_y_max(&net));
            if x0 > x1 || y0 > y1 {
                continue;
            }
            hpwl += i64::from(x1 - x0) + i64::from(y1 - y0);
        }
        let dbu = f64::from(self.db.tech_get_db_units_per_micron());
        log.info("GPL", 1018, format!("Final HPWL (um): {:.2}", hpwl as f64 / dbu));
        Ok(())
    }

    /// `Rudy::Rudy` on an uninitialized router: `getMinMaxLayer`, `initFastRoute(min, max, false)`
    /// up to its adjustments (the nets it builds after do not touch the edges), the grid and the
    /// capacity-reduction data; the routing layers' pitches for the wire width.
    fn rudy_inputs(&mut self, _log: &mut Log) -> Res<RudyInputs> {
        let opts = RouteOptions::new();
        let t = setup_tech(self.db, &opts).map_err(e)?;
        let mut grt_log = Vec::new();
        let adjusted = setup_adjust(self.db, &t, &opts, &mut grt_log).map_err(e)?;
        // an uninitialized router only: a routed one keeps its edges
        if self.edges.is_none() {
            let horizontal: Vec<bool> = t.fast_route.directions.iter().map(|d| *d == Some(Direction::Horizontal)).collect();
            let a = adjusted.edges;
            self.edges = Some(Edges { x_grid: a.x_grid, y_grid: a.y_grid, horizontal, h3: a.h3, v3: a.v3 });
        }
        let mut pitches = Vec::new();
        for level in t.min_routing_layer..=t.max_routing_layer {
            let Some((_, layer)) = t.routing_layers.iter().find(|(l, _)| *l == level) else {
                return refuse(format!("routing level {level}: no layer"));
            };
            let mut pitch = self.db.layer_get_pitch(&layer.name);
            if pitch == 0 {
                pitch = self.db.layer_get_width(&layer.name) as i32 + self.db.layer_get_spacing(&layer.name);
            }
            pitches.push(pitch);
        }
        let die = (self.db.block_get_die_area_x_min(), self.db.block_get_die_area_y_min(), self.db.block_get_die_area_x_max(), self.db.block_get_die_area_y_max());
        Ok(RudyInputs { die, x_grids: t.core.x_grids, y_grids: t.core.y_grids, tile_size: t.core.tile_size, pitches })
    }

    fn capacity_reduction(&mut self) -> Res<Vec<(u8, u8)>> {
        let Some(e) = &self.edges else { return refuse("the router's capacities before it is initialized") };
        Ok(capacity_reduction_data(e.x_grid, e.y_grid, &e.horizontal, &e.h3, &e.v3))
    }

    /// `getGrtResult`: the instances written, `globalRoute` (congestion allowed, no congestion
    /// iterations, no critical nets), then the gcell grid `updateDbCongestion` leaves — each layer
    /// in the router's range — and `getBlockage` per layer.
    fn global_route(&mut self, log: &mut Log, locations: &[(usize, i32, i32)]) -> Res<GrtCongestion> {
        self.write_locations(log, locations)?;
        let mut opts = RouteOptions::new();
        opts.allow_congestion = true;
        opts.congestion_iterations = 0;
        opts.critical_nets_percentage = 0.0;
        let t = setup_tech(self.db, &opts).map_err(e)?;
        let res = route_design(self.db, &opts, &stt, &flutes).map_err(e)?;
        let a = &res.after;
        let Some(g3) = &a.final_3d else { return refuse("the global route left no 3D edges") };
        let ((lx, cnt_x, tile_x), (ly, cnt_y, tile_y)) = res.gcell_grid;
        let cells = (cnt_x * cnt_y) as usize;
        let n_router = g3.num_layers;
        // the routed edges: capacity and usage from the route, reductions from its adjustments
        let edge = |l: usize, i: usize, h: bool| {
            let red = if h { a.edges_3d.0[l * cells + i].red } else { a.edges_3d.1[l * cells + i].red };
            let (cap, usage) = if h { (g3.h_cap[l][i], g3.h_usage[l][i]) } else { (g3.v_cap[l][i], g3.v_usage[l][i]) };
            CongestionEdge { cap, red, usage }
        };
        let horizontal: Vec<bool> = t.fast_route.directions.iter().map(|d| *d == Some(Direction::Horizontal)).collect();
        let mut h3 = Vec::with_capacity(n_router * cells);
        let mut v3 = Vec::with_capacity(n_router * cells);
        for l in 0..n_router {
            for i in 0..cells {
                let (h, v) = (edge(l, i, true), edge(l, i, false));
                h3.push(EdgeState { cap: h.cap, red: h.red, real_cap: 0 });
                v3.push(EdgeState { cap: v.cap, red: v.red, real_cap: 0 });
            }
        }
        let mut layers = Vec::new();
        for level in 1..=t.tech.routing_layer_count {
            let horizontal_layer = t.tech.routing_layers.iter().find(|l| l.routing_level == level).and_then(|l| l.direction) == Some(Direction::Horizontal);
            let mut gl = GrtLayer { horizontal: horizontal_layer, cap: vec![0; cells], usage: vec![0; cells], block_h: vec![0; cells], block_v: vec![0; cells] };
            let k = (level - 1) as usize;
            if level >= t.min_routing_layer && level <= t.max_routing_layer && k < n_router {
                let hs: Vec<CongestionEdge> = (0..cells).map(|i| edge(k, i, true)).collect();
                let vs: Vec<CongestionEdge> = (0..cells).map(|i| edge(k, i, false)).collect();
                let got = update_db_congestion(cnt_x, cnt_y, &DbCongestionLayer { horizontal: horizontal[k], h: &hs, v: &vs });
                for (i, (cap, usage)) in got.into_iter().enumerate() {
                    // read back into a `uint8_t`
                    gl.cap[i] = cap as i64 as u8;
                    gl.usage[i] = usage as i64 as u8;
                }
            }
            // `getBlockage`: at or below the router's top layer, the cell's reductions (the
            // top-right cell's from its left and lower edges)
            if level <= t.max_routing_layer && k < n_router {
                for y in 0..cnt_y {
                    for x in 0..cnt_x {
                        let i = (y * cnt_x + x) as usize;
                        let (hi, vi) = if x == cnt_x - 1 && y == cnt_y - 1 && cnt_x > 1 && cnt_y > 1 { ((y * cnt_x + x - 1) as usize, ((y - 1) * cnt_x + x) as usize) } else { (i, i) };
                        gl.block_h[i] = edge(k, hi, true).red as u8;
                        gl.block_v[i] = edge(k, vi, false).red as u8;
                    }
                }
            }
            layers.push(gl);
        }
        self.edges = Some(Edges { x_grid: cnt_x, y_grid: cnt_y, horizontal, h3, v3 });
        Ok(GrtCongestion { lx, ly, tile_x, tile_y, cnt_x, cnt_y, min_layer: t.min_routing_layer, max_layer: t.max_routing_layer, layers })
    }

    fn net_term_boxes(&mut self) -> Res<Vec<(usize, crate::route_base::Box4)>> {
        let mut out = Vec::new();
        for (k, net) in self.db.block_get_nets().into_iter().enumerate() {
            // `isSupply`
            if matches!(self.db.net_get_sig_type(&net).as_str(), "POWER" | "GROUND") {
                continue;
            }
            out.push((k, (self.db.net_get_term_b_box_x_min(&net), self.db.net_get_term_b_box_y_min(&net), self.db.net_get_term_b_box_x_max(&net), self.db.net_get_term_b_box_y_max(&net))));
        }
        Ok(out)
    }
}

/// `FastRouteCore::getCapacityReductionData`, x-major: per layer, each tile's capacity is the
/// capacity plus reduction of its edge in the layer's direction (the last tile in that direction,
/// which has no edge, repeats the previous one), and its reduction that of both its edges (the
/// top-right tile borrows its left and lower neighbours'). Every sum is the reference's `uint8_t`:
/// it wraps.
pub fn capacity_reduction_data(x_grid: i32, y_grid: i32, horizontal: &[bool], h3: &[EdgeState], v3: &[EdgeState]) -> Vec<(u8, u8)> {
    let at = |k: i32, x: i32, y: i32| ((k * y_grid + y) * x_grid + x) as usize;
    let mut out = vec![(0u8, 0u8); (x_grid * y_grid) as usize];
    let tile = |x: i32, y: i32| (x * y_grid + y) as usize;
    for (k, &is_h) in horizontal.iter().enumerate() {
        let k = k as i32;
        if is_h {
            let mut last: u8 = 0;
            for y in 0..y_grid {
                for x in 0..x_grid {
                    let cap = if x == x_grid - 1 { last } else { (i32::from(h3[at(k, x, y)].cap) + i32::from(h3[at(k, x, y)].red)) as u8 };
                    let t = &mut out[tile(x, y)];
                    t.0 = t.0.wrapping_add(cap);
                    last = cap;
                }
            }
        } else {
            let mut last: u8 = 0;
            for x in 0..x_grid {
                for y in 0..y_grid {
                    let cap = if y == y_grid - 1 { last } else { (i32::from(v3[at(k, x, y)].cap) + i32::from(v3[at(k, x, y)].red)) as u8 };
                    let t = &mut out[tile(x, y)];
                    t.0 = t.0.wrapping_add(cap);
                    last = cap;
                }
            }
        }
        for x in 0..x_grid {
            for y in 0..y_grid {
                let (bh, bv) = if x == x_grid - 1 && y == y_grid - 1 && x_grid > 1 && y_grid > 1 {
                    (h3[at(k, x - 1, y)].red as u8, v3[at(k, x, y - 1)].red as u8)
                } else {
                    (h3[at(k, x, y)].red as u8, v3[at(k, x, y)].red as u8)
                };
                let t = &mut out[tile(x, y)];
                t.1 = (i32::from(t.1) + i32::from(bh) + i32::from(bv)) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(cap: u16, red: u16) -> EdgeState {
        EdgeState { cap, red, real_cap: 0 }
    }

    /// Rule (`getCapacityReductionData`): a horizontal layer's last column repeats the column
    /// before it; the top-right tile's reduction is its left and lower neighbours'; sums wrap at
    /// 256 (`uint8_t`).
    #[test]
    fn capacity_reduction_wraps_and_repeats_the_edge() {
        // one horizontal layer, 2 x 2
        let h3 = vec![st(200, 1), st(9, 9), st(100, 2), st(9, 9)];
        let v3 = vec![st(0, 0), st(0, 0), st(0, 0), st(0, 0)];
        let cr = capacity_reduction_data(2, 2, &[true], &h3, &v3);
        // x-major: (0,0) (0,1) (1,0) (1,1)
        assert_eq!(cr, vec![(201, 1), (102, 2), (201, 9), (102, 2)]);
        // a second horizontal layer with the same edges: 201 + 201 wraps to 146
        let h3b = [h3.clone(), h3].concat();
        let v3b = [v3.clone(), v3].concat();
        let cr = capacity_reduction_data(2, 2, &[true, true], &h3b, &v3b);
        assert_eq!(cr[0].0, 146);
    }
}
