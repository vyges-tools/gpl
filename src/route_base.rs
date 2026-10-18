//! `RouteBase` — routability-driven placement's congestion measurement — and `Rudy`, the
//! rectangular-uniform-wire-density estimate it reads.
//!
//! A routability pass measures congestion on the router's tile grid: RUDY by default (every
//! signal net's terminal box spreads `hpwl · wire_width / area` over the tiles it covers), or the
//! global router's own usage. Below the target the pass ends routability; at or above it the
//! placer inflates the cells in congested tiles (`Nesterov::routability`) and places again.

use crate::log::{Log, Res};

/// `RouteBaseVars`.
#[derive(Debug, Clone, Copy)]
pub struct RouteVars {
    pub use_rudy: bool,
    pub target_rc: f32,
    pub inflation_ratio_coef: f32,
    pub max_inflation_ratio: f32,
    pub max_density: f32,
    /// `ignoreEdgeRatio`: fixed at 0.8.
    pub ignore_edge_ratio: f32,
    pub min_congestion_for_inflation: f32,
    pub max_inflation_total: f32,
    pub net_weight_max: f32,
    pub congested_nets_percentage: f32,
    pub rc_k: [f32; 4],
}

impl Default for RouteVars {
    /// `PlaceOptions`' defaults.
    fn default() -> Self {
        RouteVars {
            use_rudy: true,
            target_rc: 1.01,
            inflation_ratio_coef: 2.0,
            max_inflation_ratio: 3.0,
            max_density: 0.99,
            ignore_edge_ratio: 0.8,
            min_congestion_for_inflation: 0.95,
            max_inflation_total: 0.30,
            net_weight_max: 1.0,
            congested_nets_percentage: 1.0,
            rc_k: [1.0, 1.0, 0.0, 0.0],
        }
    }
}

/// What the router supplies when RUDY is built: the die, its tile grid and the routing layers'
/// pitches (`getPitch`, or width plus spacing when that is 0, minimum to maximum routing layer).
#[derive(Debug, Clone, Default)]
pub struct RudyInputs {
    pub die: (i32, i32, i32, i32),
    pub x_grids: i32,
    pub y_grids: i32,
    pub tile_size: i32,
    pub pitches: Vec<i32>,
}

/// One routing layer of the gcell grid the global router leaves (`updateDbCongestion`), `[y][x]`:
/// its capacity and usage as `getCapacity` / `getUsage` read back into a `uint8_t`, and
/// `getBlockage`'s horizontal and vertical reductions (0 above the router's top layer).
#[derive(Debug, Clone, Default)]
pub struct GrtLayer {
    pub horizontal: bool,
    pub cap: Vec<u8>,
    pub usage: Vec<u8>,
    pub block_h: Vec<u8>,
    pub block_v: Vec<u8>,
}

/// The gcell grid after a global route: its first gridline, tile size and count per axis, the
/// technology's routing layer count, the router's layer range, and every routing level's layer
/// (`layers[level - 1]`).
#[derive(Debug, Clone, Default)]
pub struct GrtCongestion {
    pub lx: i32,
    pub ly: i32,
    pub tile_x: i32,
    pub tile_y: i32,
    pub cnt_x: i32,
    pub cnt_y: i32,
    pub min_layer: i32,
    pub max_layer: i32,
    pub layers: Vec<GrtLayer>,
}

/// A box `(xmin, ymin, xmax, ymax)` in DBU.
pub type Box4 = (i32, i32, i32, i32);

/// The routing services a routability pass calls. The database lives with the caller.
pub trait Router {
    /// `updateDbGCells`: each instance at its location.
    fn write_locations(&mut self, log: &mut Log, locations: &[(usize, i32, i32)]) -> Res<()>;
    /// `reportHpwlMetric`: GPL-1018, the design's HPWL as the database has it.
    fn report_hpwl(&mut self, log: &mut Log) -> Res<()>;
    /// `GlobalRouter::getRudy`'s grid — the router initialized (`initFastRoute`) when it is not.
    fn rudy_inputs(&mut self, log: &mut Log) -> Res<RudyInputs>;
    /// `getCapacityReductionData`, x-major, from the router's edges as they stand — after a
    /// route, the routed ones.
    fn capacity_reduction(&mut self) -> Res<Vec<(u8, u8)>>;
    /// `getGrtResult`'s route: the database updated, `globalRoute` with congestion allowed, no
    /// congestion iterations and no critical nets — and the gcell grid it leaves.
    fn global_route(&mut self, log: &mut Log, locations: &[(usize, i32, i32)]) -> Res<GrtCongestion>;
    /// `Resizer::resizeSlackPreamble` (timing-driven placement, after GPL-0084): its warnings
    /// (RSZ-0088) and its error (RSZ-0089, no corner with a wire resistance) as it logs them.
    fn resize_slack_preamble(&mut self, _log: &mut Log) -> Res<()> {
        crate::log::refuse("timing-driven placement needs the design database's resizer")
    }
    /// `Resizer::findResizeSlacks` (timing-driven placement's reweight): the instances written,
    /// the estimate, repair_design at the preamble's wire length, the placement-time rebuffering
    /// (and, asked, one setup repair pass), then each driver's slack; the design re-read.
    fn find_resize_slacks(&mut self, _log: &mut Log, _locations: &[(usize, i32, i32)], _repair_timing: bool, _tns_end_percent: f32) -> Res<crate::timing_driven::ResizeReport> {
        crate::log::refuse("timing-driven placement needs the design database's resizer")
    }
    /// `dbNet::getTermBBox` of every net that is not POWER or GROUND, in block order, with its
    /// index among the block's nets — an inverted box for a net with no placed terminal.
    fn net_term_boxes(&mut self) -> Res<Vec<(usize, Box4)>>;
}

/// `Rudy`: the grid of tiles (x-major), each with a demand and a blockage.
#[derive(Debug, Clone)]
pub struct Rudy {
    die: (i32, i32, i32, i32),
    cnt_x: i32,
    cnt_y: i32,
    tile_size: i32,
    wire_width: i32,
    rects: Vec<(i32, i32, i32, i32)>,
    demand: Vec<f32>,
    blockage: Vec<f32>,
}

impl Rudy {
    /// `Rudy::Rudy`: the wire width — the harmonic mean pitch over the routing layers times 1.25,
    /// truncated to an integer (100 with no pitch) — then `makeGrid`.
    pub fn new(inp: &RudyInputs) -> Rudy {
        let mut pitch_terms = 0.0f64;
        for &p in &inp.pitches {
            pitch_terms += 1.0 / f64::from(p);
        }
        let wire_width = if pitch_terms != 0.0 { ((1.0 / pitch_terms) * 1.25) as i32 } else { 100 };
        let mut r = Rudy { die: inp.die, cnt_x: inp.x_grids, cnt_y: inp.y_grids, tile_size: inp.tile_size, wire_width, rects: vec![], demand: vec![], blockage: vec![] };
        r.make_grid();
        r
    }

    /// `makeGrid`: tiles of the router's size from the die's lower left; the last column and row
    /// take what the grid leaves of the die (measured from the origin, not the die's corner).
    fn make_grid(&mut self) {
        let (glx, gly, gux, guy) = self.die;
        let x_extra = gux - self.cnt_x * self.tile_size;
        let y_extra = guy - self.cnt_y * self.tile_size;
        let n = (self.cnt_x * self.cnt_y) as usize;
        self.rects = Vec::with_capacity(n);
        let mut cur_x = glx;
        for x in 0..self.cnt_x {
            let mut cur_y = gly;
            for y in 0..self.cnt_y {
                let x_ext = if x == self.cnt_x - 1 { x_extra } else { 0 };
                let y_ext = if y == self.cnt_y - 1 { y_extra } else { 0 };
                self.rects.push((cur_x, cur_y, cur_x + self.tile_size + x_ext, cur_y + self.tile_size + y_ext));
                cur_y += self.tile_size;
            }
            cur_x += self.tile_size;
        }
        self.demand = vec![0.0; n];
        self.blockage = vec![0.0; n];
    }

    fn at(&self, x: i32, y: i32) -> usize {
        (x * self.cnt_y + y) as usize
    }

    /// `getResourceReductions`: a tile's blockage is its reduction over its capacity, at most 1 —
    /// and 1 when it has no capacity at all.
    fn resource_reductions(&mut self, cap_red: &[(u8, u8)]) {
        for (i, &(cap, red)) in cap_red.iter().enumerate().take(self.blockage.len()) {
            self.blockage[i] = if cap == 0 { 1.0 } else { (f32::from(red) / f32::from(cap)).min(1.0) };
        }
    }

    /// `calculateRudy`: every tile cleared, the reductions, then each net's demand.
    pub fn calculate(&mut self, cap_red: &[(u8, u8)], nets: &[(i32, i32, i32, i32)]) {
        self.demand.iter_mut().for_each(|d| *d = 0.0);
        self.blockage.iter_mut().for_each(|b| *b = 0.0);
        self.resource_reductions(cap_red);
        for &net in nets {
            self.visit_net_tiles(net, |r, i, demand| r.demand[i] += demand);
        }
    }

    /// `getNetDemandInTiles`: each net's demand summed over the selected tiles (x-major), in
    /// the nets' order; only nets with some.
    fn net_demand_in_tiles(&mut self, selected: &[bool], nets: &[(usize, Box4)]) -> Vec<(usize, f32)> {
        let mut out = Vec::new();
        if self.cnt_y == 0 {
            return out;
        }
        for &(net, rect) in nets {
            let mut demand = 0.0f32;
            self.visit_net_tiles(rect, |_, i, d| {
                if i < selected.len() && selected[i] {
                    demand += d;
                }
            });
            if demand > 0.0 {
                out.push((net, demand));
            }
        }
        out
    }

    /// `visitNetTiles`: a net's box (skipped when inverted or of no area) deposits
    /// `hpwl · wire_width / area` (float, the area converted) in each tile it strictly overlaps,
    /// scaled by the share of the tile it covers, times 100.
    fn visit_net_tiles(&mut self, (xl, yl, xh, yh): (i32, i32, i32, i32), mut visit: impl FnMut(&mut Rudy, usize, f32)) {
        if xl > xh || yl > yh {
            return;
        }
        let net_area = i64::from(xh - xl) * i64::from(yh - yl);
        if net_area == 0 {
            return;
        }
        let hpwl = ((xh - xl) + (yh - yl)) as f32;
        let net_congestion = (hpwl * self.wire_width as f32) / net_area as f32;
        let (glx, gly) = (self.die.0, self.die.1);
        let min_x = 0.max((xl - glx) / self.tile_size);
        let max_x = (self.cnt_x - 1).min((xh - glx) / self.tile_size);
        let min_y = 0.max((yl - gly) / self.tile_size);
        let max_y = (self.cnt_y - 1).min((yh - gly) / self.tile_size);
        for x in min_x..=max_x {
            for y in min_y..=max_y {
                let i = self.at(x, y);
                let (tl, tb, tr, tt) = self.rects[i];
                // `Rect::overlaps`: strict on every side
                if !(tr > xl && tl < xh && tt > yl && tb < yh) {
                    continue;
                }
                let inter = i64::from(xh.min(tr) - xl.max(tl)) * i64::from(yh.min(tt) - yl.max(tb));
                let tile_area = i64::from(tr - tl) * i64::from(tt - tb);
                let ratio = inter as f32 / tile_area as f32;
                visit(self, i, net_congestion * ratio * 100.0);
            }
        }
    }

    /// `getTileCongestion` (gpl's): demand over the capacity left after blockage — infinite when
    /// nothing is left.
    fn congestion(&self, x: i32, y: i32) -> f32 {
        let i = self.at(x, y);
        let available = 1.0f32 - self.blockage[i];
        if available <= 0.0 {
            return f32::INFINITY;
        }
        (self.demand[i] / 100.0f32) / available
    }
}

/// gpl's `Tile`.
#[derive(Debug, Clone)]
struct Tile {
    x: i32,
    y: i32,
    inflation_ratio: f32,
    /// `inflatedRatio_`: 0 until a pass sets it.
    inflated_ratio: f32,
    actionable: bool,
}

/// `RouteBase`, its measured path.
#[derive(Debug, Clone)]
pub struct RouteBase {
    pub vars: RouteVars,
    rudy: Option<Rudy>,
    /// `tg_`: y-major tiles, from the origin.
    tiles: Vec<Tile>,
    cnt_x: i32,
    cnt_y: i32,
    size: i32,
    size_y: i32,
    /// `tg_->lx(), ly()`: 0 on the RUDY grid, the first gridline on the router's.
    grid_origin: (i32, i32),
    revert_count: i32,
    final_average_rc: f32,
    /// `minRc_` (1e30 until a pass measures), `min_RC_violated_cnt_`.
    pub min_rc: f32,
    pub min_rc_violated: i32,
    /// Per region: `minRcTargetDensity_`, `inflatedAreaDelta_`, `accumulatedInflatedAreaDelta_`,
    /// `minRcInflatedAreaDelta_`.
    pub min_rc_target_density: Vec<f32>,
    pub inflated_delta: Vec<i64>,
    pub accumulated: Vec<i64>,
    pub min_rc_inflated: Vec<i64>,
    /// `original_movable_area_`: the regions' instance area when the route base was built.
    pub original_movable_area: i64,
    /// `netWeights_` / `minRcNetWeights_`: by design net, the nets routability reweighted.
    pub net_weights: std::collections::HashMap<usize, f32>,
    pub min_rc_net_weights: std::collections::HashMap<usize, f32>,
}

impl RouteBase {
    pub fn new(vars: RouteVars) -> RouteBase {
        RouteBase {
            vars,
            rudy: None,
            tiles: vec![],
            cnt_x: 0,
            cnt_y: 0,
            size: 0,
            size_y: 0,
            grid_origin: (0, 0),
            revert_count: 0,
            final_average_rc: 0.0,
            min_rc: 1e30,
            min_rc_violated: 0,
            min_rc_target_density: vec![],
            inflated_delta: vec![],
            accumulated: vec![],
            min_rc_inflated: vec![],
            original_movable_area: 0,
            net_weights: std::collections::HashMap::new(),
            min_rc_net_weights: std::collections::HashMap::new(),
        }
    }

    pub fn rudy_average(&self) -> f32 {
        self.final_average_rc
    }

    /// `RouteBase::init` / its constructor: one slot per region, the regions' instance area.
    pub fn init(&mut self, regions: usize, original_movable_area: i64) {
        self.min_rc_target_density = vec![0.0; regions];
        self.inflated_delta = vec![0; regions];
        self.accumulated = vec![0; regions];
        self.min_rc_inflated = vec![0; regions];
        self.original_movable_area = original_movable_area;
    }

    /// `getTotalInflation`.
    pub fn total_inflation(&self) -> i64 {
        self.accumulated.iter().sum()
    }

    /// `resetRoutabilityResources`.
    pub fn reset_resources(&mut self) {
        self.inflated_delta.iter_mut().for_each(|a| *a = 0);
    }

    /// The inflated ratio of the tile under a density centre (`getTile`), if there is one.
    pub fn inflated_ratio_at(&self, x: i32, y: i32) -> Option<f32> {
        self.tile_at(x, y).map(|i| self.tiles[i].inflated_ratio)
    }

    /// `set inflated ratio`: each tile's inflation ratio when above 1, else 1.
    pub fn set_inflated_ratios(&mut self) {
        for t in self.tiles.iter_mut() {
            t.inflated_ratio = if t.inflation_ratio > 1.0 { t.inflation_ratio } else { 1.0 };
        }
    }

    /// `scaleInflationToBudget`'s trim: every inflated tile's excess over 1 scaled (in double).
    pub fn scale_inflated(&mut self, scale: f64) {
        for t in self.tiles.iter_mut() {
            if t.inflated_ratio > 1.0 {
                t.inflated_ratio = (1.0 + scale * (f64::from(t.inflated_ratio) - 1.0)) as f32;
            }
        }
    }

    /// `weightCongestedNets` up to `applyNetWeights`: on the RUDY grid, with a weight above 1 and
    /// a share above 0, the nets depositing the most demand in the tiles about to be inflated —
    /// ranked by that demand (libc++'s unstable sort: ties are values), the top share (at least
    /// one) weighted from the maximum down to 1 at the cutoff. `Some(depositors)` when nets were
    /// weighted; the map is cleared either way.
    pub fn weight_congested_nets(&mut self, router: &mut dyn Router) -> Res<Option<usize>> {
        self.net_weights.clear();
        let v = self.vars;
        if !v.use_rudy || v.net_weight_max <= 1.0 || v.congested_nets_percentage <= 0.0 {
            return Ok(None);
        }
        let Some(rudy) = self.rudy.as_mut() else { return Ok(None) };
        if rudy.cnt_x != self.cnt_x || rudy.cnt_y != self.cnt_y {
            return Ok(None);
        }
        // Rudy indexes x-major, the tile grid y-major
        let mut congested = vec![false; self.tiles.len()];
        for t in &self.tiles {
            if t.inflation_ratio > 1.0 {
                congested[(t.x * rudy.cnt_y + t.y) as usize] = true;
            }
        }
        let nets = router.net_term_boxes()?;
        let mut demand = rudy.net_demand_in_tiles(&congested, &nets);
        if demand.is_empty() {
            return Ok(None);
        }
        if crate::order::libcxx_sort_by(&mut demand, |a, b| a.1 > b.1).is_err() {
            return crate::log::refuse("ranking the congested nets reached the sort's heap fallback, which is not modelled");
        }
        let n = demand.len();
        let share = (f64::from(n as f32 * v.congested_nets_percentage) / 100.0).ceil() as usize;
        let count = n.min(share.max(1));
        let (worst, cutoff) = (demand[0].1, demand[count - 1].1);
        for &(net, d) in &demand[..count] {
            let w = if worst > cutoff { 1.0f32 + (v.net_weight_max - 1.0f32) * (d - cutoff) / (worst - cutoff) } else { v.net_weight_max };
            self.net_weights.insert(net, w);
        }
        Ok(Some(n))
    }

    /// `getTile`: the tile under a point, clamped to the last column and row; none left of or
    /// below the grid.
    fn tile_at(&self, x: i32, y: i32) -> Option<usize> {
        let ix = ((x - self.grid_origin.0) / self.size).min(self.cnt_x - 1);
        let iy = ((y - self.grid_origin.1) / self.size_y).min(self.cnt_y - 1);
        if ix < 0 || iy < 0 {
            return None;
        }
        let i = iy as usize * self.cnt_x as usize + ix as usize;
        (i < self.tiles.len()).then_some(i)
    }

    /// `calculateRudyTiles`: the database updated, RUDY computed (the router initialized on the
    /// first call), the tile grid rebuilt from the origin at the router's tile size; the tiles
    /// holding a standard cell's density centre marked actionable, then a tile mostly blocked
    /// unmarked; an actionable tile's inflation ratio from its congestion.
    pub fn calculate_rudy_tiles(&mut self, log: &mut Log, router: &mut dyn Router, locations: &[(usize, i32, i32)], std_centres: &[(i32, i32)]) -> Res<()> {
        router.write_locations(log, locations)?;
        if self.rudy.is_none() {
            let inp = router.rudy_inputs(log)?;
            self.rudy = Some(Rudy::new(&inp));
        }
        let cap_red = router.capacity_reduction()?;
        let nets: Vec<(i32, i32, i32, i32)> = router.net_term_boxes()?.into_iter().map(|(_, r)| r).collect();
        let rudy = self.rudy.as_mut().expect("initialized above");
        rudy.calculate(&cap_red, &nets);
        let (cnt_x, cnt_y, size) = (rudy.cnt_x, rudy.cnt_y, rudy.tile_size);
        // `initTiles(use_rudy)`: no layer count to report on the RUDY grid
        if !self.vars.use_rudy {
            log.info("GPL", 39, "Number of routing layers: 0");
        }
        self.cnt_x = cnt_x;
        self.cnt_y = cnt_y;
        self.size = size;
        self.size_y = size;
        self.grid_origin = (0, 0);
        self.tiles = (0..cnt_y).flat_map(|y| (0..cnt_x).map(move |x| Tile { x, y, inflation_ratio: 1.0, inflated_ratio: 0.0, actionable: true })).collect();
        // `markTilesWithMovableArea`
        self.tiles.iter_mut().for_each(|t| t.actionable = false);
        for &(cx, cy) in std_centres {
            if let Some(i) = self.tile_at(cx, cy) {
                self.tiles[i].actionable = true;
            }
        }
        let rudy = self.rudy.as_ref().expect("initialized above");
        for t in self.tiles.iter_mut() {
            if rudy.blockage[rudy.at(t.x, t.y)] >= self.vars.ignore_edge_ratio {
                t.actionable = false;
            }
            if !t.actionable {
                continue;
            }
            let ratio = rudy.congestion(t.x, t.y);
            // `updateTileInflationRatio`
            let v = &self.vars;
            if v.min_congestion_for_inflation > 0.0 && ratio >= v.min_congestion_for_inflation {
                let r = (ratio / v.min_congestion_for_inflation).powf(v.inflation_ratio_coef);
                t.inflation_ratio = r.min(v.max_inflation_ratio);
            }
        }
        Ok(())
    }

    /// `updateRudyAverage`: over the actionable tiles with a finite, non-negative congestion, the
    /// total overflow, the overflowed count and the weighted mean of the top 0.5/1/2/5% (each
    /// mean over `ceil(share · n)` tiles, in double).
    pub fn update_rudy_average(&mut self, log: &mut Log, verbose: bool) {
        let rudy = self.rudy.as_ref().expect("RUDY computed before its average");
        let mut total_overflow = 0.0f64;
        let mut overflowed = 0usize;
        let mut cong: Vec<f64> = Vec::new();
        for t in self.tiles.iter().filter(|t| t.actionable) {
            let ratio = rudy.congestion(t.x, t.y);
            if ratio.is_finite() && ratio >= 0.0 {
                total_overflow += 0.0f64.max(-1.0 + f64::from(ratio));
                cong.push(f64::from(ratio));
                if ratio > 1.0 {
                    overflowed += 1;
                }
            }
        }
        let actionable = cong.len();
        if verbose {
            log.info("GPL", 88, format!("Tiles the placer can act on: {} of {} ({:.2}%)", actionable, self.tiles.len(), actionable as f64 / self.tiles.len() as f64 * 100.0));
            log.info("GPL", 41, format!("Total routing overflow: {total_overflow:.4}"));
            let pct = if actionable > 0 { overflowed as f64 / actionable as f64 * 100.0 } else { 0.0 };
            log.info("GPL", 42, format!("Number of overflowed tiles: {overflowed} ({pct:.2}%)"));
        }
        if cong.is_empty() {
            self.final_average_rc = 0.0;
            return;
        }
        cong.sort_by(|a, b| b.total_cmp(a));
        let [a005, a010, a020, a050] = top_averages(&cong);
        let k = self.vars.rc_k;
        let weighted = (f64::from(k[0]) * a005 + f64::from(k[1]) * a010 + f64::from(k[2]) * a020 + f64::from(k[3]) * a050) / f64::from(k[0] + k[1] + k[2] + k[3]);
        self.final_average_rc = weighted as f32;
        if verbose {
            log.info("GPL", 43, format!("Average top 0.5% routing congestion: {a005:.4}"));
            log.info("GPL", 44, format!("Average top 1.0% routing congestion: {a010:.4}"));
            log.info("GPL", 45, format!("Average top 2.0% routing congestion: {a020:.4}"));
            log.info("GPL", 46, format!("Average top 5.0% routing congestion: {a050:.4}"));
            log.info("GPL", 47, format!("Routability iteration weighted routing congestion: {:.4}", self.final_average_rc));
        }
    }

    /// `RouteBase::routability`'s first half: the pass counted (GPL-0040) and the congestion
    /// measured (`curRc`) — RUDY, or the global router's own.
    pub fn measure(&mut self, log: &mut Log, router: &mut dyn Router, locations: &[(usize, i32, i32)], std_centres: &[(i32, i32)]) -> Res<f32> {
        self.revert_count += 1;
        log.info("GPL", 40, format!("Routability iteration: {}", self.revert_count));
        let cur_rc = if self.vars.use_rudy {
            self.calculate_rudy_tiles(log, router, locations, std_centres)?;
            self.update_rudy_average(log, true);
            self.final_average_rc
        } else {
            let g = router.global_route(log, locations)?;
            self.update_grt_route(log, &g, std_centres);
            self.grt_rc(log, &g)
        };
        Ok(cur_rc)
    }
}

impl RouteBase {
    /// `updateGrtRoute`: the tile grid on the gcell grid (GPL-0039: the technology's routing
    /// layers), the standard cells' tiles actionable, and each actionable tile's inflation ratio
    /// from its worst layer in the router's range — a horizontal layer also reads the tile to its
    /// left, a vertical one the tile below.
    fn update_grt_route(&mut self, log: &mut Log, g: &GrtCongestion, std_centres: &[(i32, i32)]) {
        let num_layers = g.layers.len() as i32;
        log.info("GPL", 39, format!("Number of routing layers: {num_layers}"));
        self.cnt_x = g.cnt_x;
        self.cnt_y = g.cnt_y;
        self.size = g.tile_x;
        self.grid_origin = (g.lx, g.ly);
        self.size_y = g.tile_y;
        self.tiles = (0..g.cnt_y).flat_map(|y| (0..g.cnt_x).map(move |x| Tile { x, y, inflation_ratio: 1.0, inflated_ratio: 0.0, actionable: true })).collect();
        self.tiles.iter_mut().for_each(|t| t.actionable = false);
        for &(cx, cy) in std_centres {
            if let Some(i) = self.tile_at(cx, cy) {
                self.tiles[i].actionable = true;
            }
        }
        for level in 1..=num_layers {
            let l = &g.layers[(level - 1) as usize];
            for ti in 0..self.tiles.len() {
                if !self.tiles[ti].actionable {
                    continue;
                }
                let (x, y) = (self.tiles[ti].x, self.tiles[ti].y);
                let ratio = if level >= g.min_layer && level <= g.max_layer {
                    let mut r = usage_capacity_ratio(l, g.cnt_x, x, y, self.vars.ignore_edge_ratio);
                    if l.horizontal && x >= 1 {
                        r = r.max(usage_capacity_ratio(l, g.cnt_x, x - 1, y, self.vars.ignore_edge_ratio));
                    }
                    if !l.horizontal && y >= 1 {
                        r = r.max(usage_capacity_ratio(l, g.cnt_x, x, y - 1, self.vars.ignore_edge_ratio));
                    }
                    r.max(0.0)
                } else {
                    0.0
                };
                let v = self.vars;
                let t = &mut self.tiles[ti];
                if v.min_congestion_for_inflation > 0.0 && ratio >= v.min_congestion_for_inflation {
                    t.inflation_ratio = ((ratio / v.min_congestion_for_inflation).powf(v.inflation_ratio_coef)).min(v.max_inflation_ratio);
                }
            }
        }
    }

    /// `getGrtRC`: per actionable tile and every routing layer (0 outside the router's range), the
    /// usage ratio joins its direction's list; the top 0.5/1/2/5% means of each list, the larger
    /// direction's weighted (GPL-0064..0075).
    fn grt_rc(&self, log: &mut Log, g: &GrtCongestion) -> f32 {
        let (mut over_h, mut over_v, mut over_cnt) = (0.0f64, 0.0f64, 0i32);
        let (mut hor, mut ver): (Vec<f64>, Vec<f64>) = (vec![], vec![]);
        for t in self.tiles.iter().filter(|t| t.actionable) {
            for (k, l) in g.layers.iter().enumerate() {
                let level = k as i32 + 1;
                let mut ratio = usage_capacity_ratio(l, g.cnt_x, t.x, t.y, self.vars.ignore_edge_ratio);
                if level < g.min_layer || level > g.max_layer {
                    ratio = 0.0;
                }
                if ratio >= 0.0 {
                    if l.horizontal {
                        over_h += 0.0f64.max(-1.0 + f64::from(ratio));
                        hor.push(f64::from(ratio));
                    } else {
                        over_v += 0.0f64.max(-1.0 + f64::from(ratio));
                        ver.push(f64::from(ratio));
                    }
                    if ratio > 1.0 {
                        over_cnt += 1;
                    }
                }
            }
        }
        log.info("GPL", 64, format!("TotalRouteOverflowH2: {over_h:.4}"));
        log.info("GPL", 65, format!("TotalRouteOverflowV2: {over_v:.4}"));
        log.info("GPL", 66, format!("OverflowTileCnt2: {over_cnt}"));
        hor.sort_by(|a, b| b.total_cmp(a));
        ver.sort_by(|a, b| b.total_cmp(a));
        let (h, v) = (top_averages(&hor), top_averages(&ver));
        // `std::fmax`: a NaN mean (an empty list) yields the other
        let m: Vec<f64> = (0..4).map(|i| h[i].max(v[i])).collect();
        log.info("GPL", 67, format!("0.5%RC: {:.4}", m[0]));
        log.info("GPL", 68, format!("1.0%RC: {:.4}", m[1]));
        log.info("GPL", 69, format!("2.0%RC: {:.4}", m[2]));
        log.info("GPL", 70, format!("5.0%RC: {:.4}", m[3]));
        let k = self.vars.rc_k;
        log.info("GPL", 71, format!("0.5rcK: {:.2}", k[0]));
        log.info("GPL", 72, format!("1.0rcK: {:.2}", k[1]));
        log.info("GPL", 73, format!("2.0rcK: {:.2}", k[2]));
        log.info("GPL", 74, format!("5.0rcK: {:.2}", k[3]));
        let rc = ((f64::from(k[0]) * m[0] + f64::from(k[1]) * m[1] + f64::from(k[2]) * m[2] + f64::from(k[3]) * m[3]) / f64::from(k[0] + k[1] + k[2] + k[3])) as f32;
        log.info("GPL", 75, format!("Final routing congestion: {rc}"));
        rc
    }
}

/// `getUsageCapacityRatio`: usage over capacity — `lowest` (never counted) where the tile has no
/// capacity or its blockage in the layer's direction is at least `ignore_edge_ratio` of it.
fn usage_capacity_ratio(l: &GrtLayer, cnt_x: i32, x: i32, y: i32, ignore_edge_ratio: f32) -> f32 {
    let i = (y * cnt_x + x) as usize;
    let (cap, used) = (l.cap[i], l.usage[i]);
    let blockage = if l.horizontal { l.block_h[i] } else { l.block_v[i] };
    if cap == 0 {
        return f32::MIN;
    }
    if f32::from(blockage) / f32::from(cap) >= ignore_edge_ratio {
        return f32::MIN;
    }
    f32::from(used) / f32::from(cap)
}

/// The means of the top 0.5%, 1%, 2% and 5% of a descending list: element `i` joins a share when
/// `i < share · n`, and each sum is divided by `ceil(share · n)`.
fn top_averages(desc: &[f64]) -> [f64; 4] {
    let n = desc.len() as f64;
    let shares = [0.005, 0.01, 0.02, 0.05];
    let mut sums = [0.0f64; 4];
    for (i, &v) in desc.iter().enumerate() {
        for (s, sum) in shares.iter().zip(sums.iter_mut()) {
            if (i as f64) < s * n {
                *sum += v;
            }
        }
    }
    // the divisors as the reference spells them: 0.005, 0.010, 0.020, 0.050
    [sums[0] / (0.005 * n).ceil(), sums[1] / (0.010 * n).ceil(), sums[2] / (0.020 * n).ceil(), sums[3] / (0.050 * n).ceil()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> RudyInputs {
        RudyInputs { die: (0, 0, 250, 200), x_grids: 2, y_grids: 2, tile_size: 100, pitches: vec![100, 100] }
    }

    /// Rule (`Rudy::Rudy`): wire width = (1 / Σ 1/pitch) · 1.25, truncated to an int; 100 with no
    /// layers.
    #[test]
    fn wire_width_is_the_truncated_harmonic_pitch() {
        assert_eq!(Rudy::new(&inputs()).wire_width, 62);
        assert_eq!(Rudy::new(&RudyInputs { pitches: vec![], ..inputs() }).wire_width, 100);
    }

    /// Rule (`makeGrid`): the last column/row is stretched by what the grid leaves of the die,
    /// measured from the origin; x-major.
    #[test]
    fn last_column_takes_the_die_remainder() {
        let r = Rudy::new(&inputs());
        assert_eq!(r.rects, vec![(0, 0, 100, 100), (0, 100, 100, 200), (100, 0, 250, 100), (100, 100, 250, 200)]);
    }

    /// Rule (`visitNetTiles`): a tile only touched by the box gets nothing (strict overlap); a box
    /// of no area deposits nothing; an inverted box is skipped.
    #[test]
    fn rudy_deposits_on_strict_overlap_only() {
        let mut r = Rudy::new(&inputs());
        r.calculate(&[(10, 0); 4], &[(0, 0, 100, 50), (10, 10, 10, 90), (i32::MAX, i32::MAX, i32::MIN, i32::MIN)]);
        // hpwl 150 · 62 / 5000 = 1.86, half the tile, times 100
        assert_eq!(r.demand[0], 1.86f32 * 0.5 * 100.0);
        assert_eq!(&r.demand[1..], &[0.0, 0.0, 0.0]);
    }

    /// Rule (`getResourceReductions`): blockage = min(red / cap, 1), and 1 at zero capacity; the
    /// congestion divides by what is left, infinite when nothing is.
    #[test]
    fn blockage_derates_the_tile() {
        let mut r = Rudy::new(&inputs());
        r.calculate(&[(10, 5), (0, 0), (4, 9), (10, 0)], &[]);
        assert_eq!(r.blockage, vec![0.5, 1.0, 1.0, 0.0]);
        r.demand[0] = 50.0;
        assert_eq!(r.congestion(0, 0), 1.0);
        assert!(r.congestion(0, 1).is_infinite());
    }

    /// Rule (`updateRudyAverage`): element i joins a share when i < share · n; each mean divides
    /// by ceil(share · n) — with 10 tiles every share is the top one.
    #[test]
    fn top_shares_round_up() {
        let v: Vec<f64> = (0..10).map(|i| f64::from(10 - i)).collect();
        assert_eq!(top_averages(&v), [10.0, 10.0, 10.0, 10.0]);
        let v: Vec<f64> = (0..100).map(|i| f64::from(100 - i)).collect();
        assert_eq!(top_averages(&v), [100.0, 100.0, 99.5, 98.0]);
    }
}
