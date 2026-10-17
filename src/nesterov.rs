// SPDX-License-Identifier: Apache-2.0
//! Nesterov global placement (the ePlace electrostatic model): the placer's cells (`GCell`), nets
//! (`GNet`) and pins (`GPin`) over the bin grid, filler cells, the density field from a 2-D DCT
//! Poisson solve, the weighted-average (WA) wirelength gradient, and the accelerated descent with
//! back-tracking — in the reference's call sequence, each stage its own function.
//!
//! Scope: the plain mode over the top level and every region group (`NesterovBaseCommon` shared,
//! one `NesterovBase` per region, `NesterovPlace` looping over them) — no clusters, no IO-pin
//! cells, no timing-driven or routability modes, no virtual CTS. Anything else is refused by the
//! caller.
//!
//! ⛔ **Every value is the reference's type, and every expression its evaluation order.** The
//! cells' coordinates, gradients, step length and penalty are `float`; several expressions mix in
//! a `double` literal (`1.0 - coef · x`, `stepLength · 0.95`, the phi coefficient's `pow`) and
//! are computed in `f64` here exactly where they are there; the wavenumbers and the √2 density
//! threshold are `long double` ([`crate::ld`]). Sums run in the reference's order (bins in index
//! order, cells in placement order, a net's pins in the net's order). The iteration count is a
//! value: one ulp early moves the convergence iteration and every row after it.

// Index loops over the parallel per-cell arrays read as the reference's loops read.
#![allow(clippy::needless_range_loop)]

use crate::ld::{self, Ld};
use crate::log::{refuse, Log, Res};
use crate::normal::{normal, Mt19937};
use crate::ooura;
use crate::placer_base::{Base, Common, Design};
use crate::initial_place::Ip;

/// `PlaceOptions` as the plain Nesterov mode reads them.
#[derive(Debug, Clone)]
pub struct Opts {
    pub density: f32,
    pub uniform_density: bool,
    pub bin_cnt: Option<i32>,
    pub init_density_penalty: f32,
    pub init_wirelength_coef: f32,
    pub min_phi_coef: f32,
    pub max_phi_coef: f32,
    pub overflow: f32,
    pub reference_hpwl: f32,
    pub max_iter: i32,
    pub seed: i32,
    pub perturb_dist: f32,
    pub disable_revert_if_diverge: bool,
    pub pad_left: i32,
}

impl Default for Opts {
    fn default() -> Opts {
        Opts {
            density: 0.7,
            uniform_density: false,
            bin_cnt: None,
            init_density_penalty: 0.00008,
            init_wirelength_coef: 0.25,
            min_phi_coef: 0.95,
            max_phi_coef: 1.05,
            overflow: 0.1,
            reference_hpwl: 0.0,
            max_iter: 5000,
            seed: 1,
            perturb_dist: -1.0,
            disable_revert_if_diverge: false,
            pad_left: 0,
        }
    }
}

const MIN_WIRELENGTH_FORCE_BAR: f32 = -300.0;
const MAX_BACK_TRACK: i32 = 10;
const MIN_PRECONDITIONER: f32 = 1.0;
const MAX_RECURSION_WL_COEF: i32 = 10;
const MAX_RECURSION_INIT_SLP_COEF: i32 = 10;
const REFERENCE_HPWL_FLOOR: f32 = 446_000_000.0;
const REFERENCE_HPWL_FRACTION: f32 = 1.0;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FloatPoint {
    pub x: f32,
    pub y: f32,
}

impl FloatPoint {
    fn new(x: f32, y: f32) -> FloatPoint {
        FloatPoint { x, y }
    }
}

/// `GCell`: a placed instance (`inst`, its index in [`Common::insts`]) or a filler.
#[derive(Debug, Clone, Default)]
pub struct GCell {
    pub inst: Option<usize>,
    pub macro_: bool,
    pub lx: i32,
    pub ly: i32,
    pub ux: i32,
    pub uy: i32,
    pub dlx: i32,
    pub dly: i32,
    pub dux: i32,
    pub duy: i32,
    pub density_scale: f32,
    pub pins: Vec<usize>,
    pub locked: bool,
    /// A placement cluster's instances, in its order (`inst` is the first); empty for a cell of
    /// one instance or a filler.
    pub members: Vec<usize>,
}

impl GCell {
    /// `GCell(cx, cy, dx, dy)`: a filler, halves truncated.
    fn filler(cx: i32, cy: i32, dx: i32, dy: i32) -> GCell {
        let (lx, ly, ux, uy) = (cx - dx / 2, cy - dy / 2, cx + dx / 2, cy + dy / 2);
        GCell { lx, ly, ux, uy, dlx: lx, dly: ly, dux: ux, duy: uy, ..GCell::default() }
    }
    fn dx(&self) -> i32 {
        self.ux - self.lx
    }
    fn dy(&self) -> i32 {
        self.uy - self.ly
    }
    fn d_cx(&self) -> i32 {
        (self.dux + self.dlx) / 2
    }
    fn d_cy(&self) -> i32 {
        (self.duy + self.dly) / 2
    }
    fn d_dx(&self) -> i32 {
        self.dux - self.dlx
    }
    fn d_dy(&self) -> i32 {
        self.duy - self.dly
    }
    fn is_instance(&self) -> bool {
        self.inst.is_some()
    }
    fn is_macro(&self) -> bool {
        self.inst.is_some() && self.macro_
    }
    fn is_std(&self) -> bool {
        self.inst.is_some() && !self.macro_
    }
    /// `setDensitySize`: about the density centre, halves truncated.
    fn set_density_size(&mut self, ddx: i32, ddy: i32) {
        let (cx, cy) = (self.d_cx(), self.d_cy());
        self.dlx = cx - ddx / 2;
        self.dly = cy - ddy / 2;
        self.dux = cx + ddx / 2;
        self.duy = cy + ddy / 2;
    }
}

/// `GPin`.
#[derive(Debug, Clone, Default)]
pub struct GPin {
    pub cx: i32,
    pub cy: i32,
    pub off_x: i32,
    pub off_y: i32,
    pub gcell: Option<usize>,
    has_min_x: bool,
    has_max_x: bool,
    has_min_y: bool,
    has_max_y: bool,
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
}

/// `GNet`.
#[derive(Debug, Clone, Default)]
pub struct GNet {
    pub pins: Vec<usize>,
    lx: i32,
    ly: i32,
    ux: i32,
    uy: i32,
    min_x: f32,
    x_min_x: f32,
    max_x: f32,
    x_max_x: f32,
    min_y: f32,
    y_min_y: f32,
    max_y: f32,
    y_max_y: f32,
}

impl GNet {
    /// `GNet::getTotalWeight`: timing × custom, both 1 in the plain mode.
    fn total_weight(&self) -> f32 {
        1.0
    }
    fn hpwl(&self) -> i64 {
        if self.ux < self.lx {
            return 0;
        }
        (i64::from(self.ux) - i64::from(self.lx)) + (i64::from(self.uy) - i64::from(self.ly))
    }
}

/// `Bin`.
#[derive(Debug, Clone, Default)]
pub struct Bin {
    pub lx: i32,
    pub ly: i32,
    pub ux: i32,
    pub uy: i32,
    pub target_density: f32,
    pub density: f32,
    pub field_x: f32,
    pub field_y: f32,
    pub phi: f32,
    pub non_place: i64,
    pub non_place_unscaled: i64,
    pub inst_placed: i64,
    pub inst_placed_unscaled: i64,
    pub filler: i64,
}

impl Bin {
    fn area(&self) -> i64 {
        i64::from(self.ux - self.lx) * i64::from(self.uy - self.ly)
    }
}

/// `BinGrid`.
#[derive(Debug, Clone, Default)]
pub struct BinGrid {
    pub lx: i32,
    pub ly: i32,
    pub ux: i32,
    pub uy: i32,
    pub cnt_x: i32,
    pub cnt_y: i32,
    pub size_x: f64,
    pub size_y: f64,
    pub bins: Vec<Bin>,
    pub overflow: i64,
    pub overflow_unscaled: i64,
}

impl BinGrid {
    fn density_idx_x(&self, g: &GCell) -> (i32, i32) {
        let lower = (f64::from(g.dlx - self.lx) / self.size_x) as i32;
        let upper = (f64::from(g.dux - self.lx) / self.size_x).ceil() as i32;
        (lower.max(0), upper.min(self.cnt_x))
    }
    fn density_idx_y(&self, g: &GCell) -> (i32, i32) {
        let lower = (f64::from(g.dly - self.ly) / self.size_y) as i32;
        let upper = (f64::from(g.duy - self.ly) / self.size_y).ceil() as i32;
        (lower.max(0), upper.min(self.cnt_y))
    }
    fn idx_x(&self, lx: i32, ux: i32) -> (i32, i32) {
        let lower = (f64::from(lx - self.lx) / self.size_x) as i32;
        let upper = (f64::from(ux - self.lx) / self.size_x).ceil() as i32;
        (lower.max(0), upper.min(self.cnt_x))
    }
    fn idx_y(&self, ly: i32, uy: i32) -> (i32, i32) {
        let lower = (f64::from(ly - self.ly) / self.size_y) as i32;
        let upper = (f64::from(uy - self.ly) / self.size_y).ceil() as i32;
        (lower.max(0), upper.min(self.cnt_y))
    }
}

/// `getOverlapDensityArea`: the overlap of a bin with a cell's density box, as a float product.
fn overlap_density_area(b: &Bin, c: &GCell) -> f32 {
    let (lx, ly) = (b.lx.max(c.dlx), b.ly.max(c.dly));
    let (ux, uy) = (b.ux.min(c.dux), b.uy.min(c.duy));
    if lx >= ux || ly >= uy {
        return 0.0;
    }
    (ux - lx) as f32 * (uy - ly) as f32
}

/// `fastExp`: `(1 + x/1024)^1024` by ten squarings.
pub fn fast_exp(x: f32) -> f32 {
    let mut e = 1.0f32 + x / 1024.0f32;
    for _ in 0..10 {
        e *= e;
    }
    e
}

/// `getDistance` (no IO pins): a float sum of squared differences in cell order, then
/// `sqrt(sum / (2.0 · n))` in double, returned as float.
fn distance(a: &[FloatPoint], b: &[FloatPoint]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += (a[i].x - b[i].x) * (a[i].x - b[i].x);
        sum += (a[i].y - b[i].y) * (a[i].y - b[i].y);
    }
    if a.is_empty() {
        return 0.0;
    }
    (f64::from(sum) / (2.0 * a.len() as f64)).sqrt() as f32
}

extern "C" {
    fn erff(x: f32) -> f32;
}

/// `calculateBiVariateNormalCDF` (float `erf`, the `√2` a double).
#[allow(clippy::too_many_arguments)]
fn bivariate_normal_cdf(mean_x: f32, mean_y: f32, sigma_x: f32, sigma_y: f32, lx: f32, ly: f32, ux: f32, uy: f32) -> f32 {
    let s2 = 2f64.sqrt();
    let x1 = (f64::from(mean_x - lx) / (s2 * f64::from(sigma_x))) as f32;
    let x2 = (f64::from(mean_x - ux) / (s2 * f64::from(sigma_x))) as f32;
    let y1 = (f64::from(mean_y - ly) / (s2 * f64::from(sigma_y))) as f32;
    let y2 = (f64::from(mean_y - uy) / (s2 * f64::from(sigma_y))) as f32;
    // SAFETY: `erff` is the C library's pure function of one float.
    let erf = |v: f32| unsafe { erff(v) };
    let s = erf(x1) * erf(y1) + erf(x2) * erf(y2) - erf(x1) * erf(y2) - erf(x2) * erf(y1);
    (0.25 * f64::from(s)) as f32
}

/// The FFT context: the bin grid's density, potential and field, and the solver's tables.
#[derive(Debug, Clone)]
struct Fft {
    nx: usize,
    ny: usize,
    density: Vec<f32>,
    phi: Vec<f32>,
    field_x: Vec<f32>,
    field_y: Vec<f32>,
    cs: Vec<f32>,
    ip: Vec<i32>,
    wx: Vec<f32>,
    wx2: Vec<f32>,
    wy: Vec<f32>,
    wy2: Vec<f32>,
}

impl Fft {
    /// The CPU backend's construction: the wavenumbers in extended precision.
    fn new(nx: i32, ny: i32, bin_x: f32, bin_y: f32) -> Fft {
        let (nxu, nyu) = (nx as usize, ny as usize);
        let mx = nx.max(ny);
        let mut f = Fft {
            nx: nxu,
            ny: nyu,
            density: vec![0.0; nxu * nyu],
            phi: vec![0.0; nxu * nyu],
            field_x: vec![0.0; nxu * nyu],
            field_y: vec![0.0; nxu * nyu],
            cs: vec![0.0; (mx * 3 / 2) as usize],
            ip: vec![0; (f64::from(mx).sqrt().round() as usize) + 2],
            wx: vec![0.0; nxu],
            wx2: vec![0.0; nxu],
            wy: vec![0.0; nyu],
            wy2: vec![0.0; nyu],
        };
        for i in 0..nxu {
            f.wx[i] = ld::PI.times(Ld::from_f32(i as f32)).over(Ld::from_f32(nx as f32)).to_f32();
            f.wx2[i] = f.wx[i] * f.wx[i];
        }
        for i in 0..nyu {
            f.wy[i] = ld::PI.times(Ld::from_f32(i as f32)).over(Ld::from_f32(ny as f32)).times(Ld::from_f32(bin_y)).over(Ld::from_f32(bin_x)).to_f32();
            f.wy2[i] = f.wy[i] * f.wy[i];
        }
        f
    }

    /// `CpuFftBackend::solve`: forward DCT, normalisation, the Poisson solve per wavenumber, then
    /// the inverse transforms of the potential and both field components.
    fn solve(&mut self) {
        let (nx, ny) = (self.nx, self.ny);
        ooura::ddct2d(nx, ny, -1, &mut self.density, &mut self.ip, &mut self.cs);
        for i in 1..nx {
            self.density[i * ny] = (f64::from(self.density[i * ny]) * 0.5) as f32;
        }
        for j in 1..ny {
            self.density[j] = (f64::from(self.density[j]) * 0.5) as f32;
        }
        let scale = 4.0 / nx as f64 / ny as f64;
        for v in self.density.iter_mut() {
            *v = (f64::from(*v) * scale) as f32;
        }
        for i in 0..nx {
            let (wx, wx2) = (self.wx[i], self.wx2[i]);
            for j in 0..ny {
                let (wy, wy2) = (self.wy[j], self.wy2[j]);
                let k = i * ny + j;
                let (p, ex, ey) = if i == 0 && j == 0 {
                    (0.0, 0.0, 0.0)
                } else {
                    let p = self.density[k] / (wx2 + wy2);
                    (p, p * wx, p * wy)
                };
                self.phi[k] = p;
                self.field_x[k] = ex;
                self.field_y[k] = ey;
            }
        }
        ooura::ddct2d(nx, ny, 1, &mut self.phi, &mut self.ip, &mut self.cs);
        ooura::ddsct2d(nx, ny, 1, &mut self.field_x, &mut self.ip, &mut self.cs);
        ooura::ddcst2d(nx, ny, 1, &mut self.field_y, &mut self.ip, &mut self.cs);
    }
}

/// One region's `NesterovBase`: its cells (indices into [`Nesterov::gcells`] — its placed
/// instances in its placer base's order, then its own fillers), its bin grid and density field,
/// and its descent state, every per-cell vector in `cells` order.
#[derive(Debug, Clone)]
pub struct Nb {
    pub cells: Vec<usize>,
    /// The region group's name; `None` for the top level.
    pub group: Option<String>,
    pub bg: BinGrid,
    fft: Fft,
    pub filler_dx: i32,
    pub filler_dy: i32,
    pub fillers: usize,
    pub target_density: f32,
    uniform_target_density: f32,
    white_space_area: i64,
    std_insts_area: i64,
    macro_insts_area: i64,
    non_place_insts_area: i64,
    max_phi_coef: f32,
    max_phi_coef_changed: bool,
    cur_slp: Vec<FloatPoint>,
    cur_slp_wl: Vec<FloatPoint>,
    cur_slp_den: Vec<FloatPoint>,
    cur_slp_sum: Vec<FloatPoint>,
    next_slp: Vec<FloatPoint>,
    next_slp_wl: Vec<FloatPoint>,
    next_slp_den: Vec<FloatPoint>,
    next_slp_sum: Vec<FloatPoint>,
    prev_slp: Vec<FloatPoint>,
    prev_slp_wl: Vec<FloatPoint>,
    prev_slp_den: Vec<FloatPoint>,
    prev_slp_sum: Vec<FloatPoint>,
    cur: Vec<FloatPoint>,
    next: Vec<FloatPoint>,
    snapshot: Vec<FloatPoint>,
    snapshot_slp: Vec<FloatPoint>,
    snapshot_slp_sum: Vec<FloatPoint>,
    snapshot_prev_slp_sum: Vec<FloatPoint>,
    snapshot_density_penalty: f32,
    snapshot_step_length: f32,
    pub step_length: f32,
    pub density_penalty: f32,
    wl_grad_sum: f32,
    density_grad_sum: f32,
    pub sum_overflow: f32,
    pub sum_overflow_unscaled: f32,
    base_wl_coef_nb: f32,
    prev_hpwl: i64,
    prev_reported_hpwl: i64,
    prev_reported_overflow_unscaled: f32,
    min_sum_overflow: f32,
    hpwl_with_min_sum_overflow: f32,
    coordi_distance: f32,
    is_converged: bool,
    is_diverged: bool,
}

impl Nb {
    fn new(base: &Base, group: Option<String>, o: &Opts) -> Nb {
        Nb {
            cells: vec![],
            group,
            bg: BinGrid::default(),
            fft: Fft::new(1, 1, 1.0, 1.0),
            filler_dx: 0,
            filler_dy: 0,
            fillers: 0,
            target_density: 0.0,
            uniform_target_density: 0.0,
            white_space_area: 0,
            std_insts_area: base.std_insts_area,
            macro_insts_area: base.macro_insts_area,
            non_place_insts_area: base.non_place_insts_area,
            max_phi_coef: o.max_phi_coef,
            max_phi_coef_changed: false,
            cur_slp: vec![],
            cur_slp_wl: vec![],
            cur_slp_den: vec![],
            cur_slp_sum: vec![],
            next_slp: vec![],
            next_slp_wl: vec![],
            next_slp_den: vec![],
            next_slp_sum: vec![],
            prev_slp: vec![],
            prev_slp_wl: vec![],
            prev_slp_den: vec![],
            prev_slp_sum: vec![],
            cur: vec![],
            next: vec![],
            snapshot: vec![],
            snapshot_slp: vec![],
            snapshot_slp_sum: vec![],
            snapshot_prev_slp_sum: vec![],
            snapshot_density_penalty: 0.0,
            snapshot_step_length: 0.0,
            step_length: 0.0,
            density_penalty: 0.0,
            wl_grad_sum: 0.0,
            density_grad_sum: 0.0,
            sum_overflow: 0.0,
            sum_overflow_unscaled: 0.0,
            base_wl_coef_nb: 0.0,
            prev_hpwl: 0,
            prev_reported_hpwl: 0,
            prev_reported_overflow_unscaled: 0.0,
            min_sum_overflow: 1e30,
            hpwl_with_min_sum_overflow: 1e30,
            coordi_distance: 0.0,
            is_converged: false,
            is_diverged: false,
        }
    }
}

/// The placer's whole Nesterov state: `NesterovBaseCommon` (every placed instance's cell, the pins
/// and nets — then each region's fillers, appended), the regions' `NesterovBase`s, and
/// `NesterovPlace`'s loop variables.
#[derive(Debug, Clone)]
pub struct Nesterov {
    pub o: Opts,
    pub gcells: Vec<GCell>,
    pub gpins: Vec<GPin>,
    pub gnets: Vec<GNet>,
    /// Each pin's net (`GPin::getGNet`).
    pin_net: Vec<usize>,
    pub nbs: Vec<Nb>,
    // NesterovPlace.
    wl_coef_x: f32,
    wl_coef_y: f32,
    base_wl_coef: f32,
    recursion_wl: i32,
    recursion_init_slp: i32,
    initial_prev_coordi_update_coef: f32,
    reference_hpwl: f32,
    max_iter: i32,
    average_overflow: f32,
    average_overflow_unscaled: f32,
    min_hpwl: i64,
    is_min_hpwl: bool,
    diverge_snapshot_iter: i32,
    diverge_snapshot_overflow: f32,
    num_region_diverged: i32,
    diverge: Option<(u32, &'static str)>,
}

/// A float point vector's FNV-1a (x then y, each little-endian) — the reference probe's.
fn fnv_points(v: &[FloatPoint]) -> u64 {
    let mut h = FNV0;
    for p in v {
        h = fnv(&p.x.to_le_bytes(), h);
        h = fnv(&p.y.to_le_bytes(), h);
    }
    h
}

const FNV0: u64 = 1469598103934665603;

fn fnv(bytes: &[u8], mut h: u64) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(1099511628211);
    }
    h
}

/// `roundDownToPowerOfTwo`.
fn round_down_to_power_of_two(mut x: u32) -> u32 {
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x ^ (x >> 1)
}

fn dbu_area(d: &Design, a: f64) -> f64 {
    a / (f64::from(d.dbu) * f64::from(d.dbu))
}

/// `NesterovBaseCommon` then each placer base's `NesterovBase` (`Replace::initNesterovPlace`, up
/// to the placer): one cell per placed instance of the whole design, pins and nets; then per
/// region its fillers, its perturbation, its bin grid, its FFT, its density sizes.
pub fn init_nesterov_place(log: &mut Log, d: &Design, c: &mut Common, ip: &mut Ip, bases: &[Base], o: &Opts) -> Res<Nesterov> {
    // NesterovBaseCommon: the placement clusters first, each ONE cell of its instances; then one
    // cell per other placed instance; pins and nets as the placer holds them.
    let mut gcells: Vec<GCell> = Vec::new();
    let mut inst_cell = vec![None; c.insts.len()];
    let mut pb_of = vec![None; d.insts.len()];
    for (i, inst) in c.insts.iter().enumerate() {
        if let Some(k) = inst.db {
            pb_of[k] = Some(i);
        }
    }
    for cl in &d.clusters {
        let mut members = Vec::with_capacity(cl.len());
        for &k in cl {
            match pb_of[k] {
                Some(i) if !c.insts[i].fixed => members.push(i),
                _ => return refuse("a placement cluster holding an instance the placer does not move is not modelled"),
            }
        }
        if members.iter().any(|&m| inst_cell[m].is_some()) {
            return refuse("an instance in two placement clusters is not modelled");
        }
        let gi = gcells.len();
        for &m in &members {
            inst_cell[m] = Some(gi);
        }
        let first = &c.insts[members[0]];
        let (lx, ly, ux, uy) = cluster_box(c, &members, d);
        let locked = members.iter().any(|&m| c.insts[m].locked);
        let pins = members.iter().flat_map(|&m| ip.inst_pins[m].iter().copied()).collect();
        gcells.push(GCell { inst: Some(members[0]), macro_: first.macro_, lx, ly, ux, uy, dlx: lx, dly: ly, dux: ux, duy: uy, locked, pins, members, ..GCell::default() });
    }
    for &k in &ip.place {
        if inst_cell[k].is_some() {
            continue;
        }
        let i = &c.insts[k];
        inst_cell[k] = Some(gcells.len());
        gcells.push(GCell { inst: Some(k), macro_: i.macro_, lx: i.lx, ly: i.ly, ux: i.ux, uy: i.uy, dlx: i.lx, dly: i.ly, dux: i.ux, duy: i.uy, locked: i.locked, ..GCell::default() });
    }
    let gpins: Vec<GPin> = ip.pins.iter().map(|p| GPin { cx: p.cx, cy: p.cy, off_x: p.off_x, off_y: p.off_y, gcell: p.inst.and_then(|k| inst_cell[k]), ..GPin::default() }).collect();
    for (k, pins) in ip.inst_pins.iter().enumerate() {
        if let Some(g) = inst_cell[k] {
            if gcells[g].members.is_empty() {
                gcells[g].pins = pins.clone();
            }
        }
    }
    let gnets: Vec<GNet> = ip.net_pins.iter().map(|p| GNet { pins: p.clone(), ..GNet::default() }).collect();
    let mut pin_net = vec![0usize; ip.pins.len()];
    for (n, pins) in ip.net_pins.iter().enumerate() {
        for &p in pins {
            pin_net[p] = n;
        }
    }
    // Every base's placed instances are cells of the common set (`pbToNb`).
    if bases.iter().any(|b| b.place_insts.iter().any(|&k| inst_cell[k].is_none())) {
        return refuse("a placer region holding an instance outside the common placed set is not modelled");
    }
    let mut np = Nesterov {
        o: o.clone(),
        gcells,
        gpins,
        gnets,
        pin_net,
        nbs: Vec::new(),
        wl_coef_x: 0.0,
        wl_coef_y: 0.0,
        base_wl_coef: 0.0,
        recursion_wl: 0,
        recursion_init_slp: 0,
        initial_prev_coordi_update_coef: 100.0,
        reference_hpwl: 0.0,
        max_iter: o.max_iter,
        average_overflow: 0.0,
        average_overflow_unscaled: 0.0,
        min_hpwl: i64::MAX,
        is_min_hpwl: false,
        diverge_snapshot_iter: 0,
        diverge_snapshot_overflow: 0.0,
        num_region_diverged: 0,
        diverge: None,
    };
    for base in bases {
        let nb = nesterov_base(log, d, c, ip, base, &inst_cell, &mut np)?;
        np.nbs.push(nb);
    }
    Ok(np)
}

/// `NesterovBase::NesterovBase`: the fillers first (their size from this region's instances),
/// then the region's instances perturbed in its own order with its own generator, its cells (the
/// instances, then the fillers), the bin grid over the region, the FFT, the density sizes.
fn nesterov_base(log: &mut Log, d: &Design, c: &mut Common, ip: &mut Ip, base: &Base, inst_cell: &[Option<usize>], np: &mut Nesterov) -> Res<Nb> {
    let group = base.group.map(|g| d.groups[g].name.clone());
    log.info("GPL", 33, format!("---- Initialize Nesterov Region: {}", group.as_deref().unwrap_or("Top-level")));
    let mut nb = Nb::new(base, group, &np.o);
    let fillers = init_filler_gcells(log, d, c, base, &mut nb, &np.o)?;
    perturb_place_insts(d, c, ip, base, inst_cell, &mut np.gcells, &np.o);
    nb.fillers = fillers.len();
    // a cluster is added where its first instance comes in the region's order
    nb.cells = base.place_insts.iter().filter_map(|&k| {
        let gi = inst_cell[k].expect("checked");
        np.gcells[gi].members.first().is_none_or(|&m| m == k).then_some(gi)
    }).collect();
    for f in fillers {
        nb.cells.push(np.gcells.len());
        np.gcells.push(f);
    }
    init_bins(log, d, c, base, &mut nb, &np.o);
    nb.fft = Fft::new(nb.bg.cnt_x, nb.bg.cnt_y, nb.bg.size_x as f32, nb.bg.size_y as f32);
    update_density_size(&mut np.gcells, &nb);
    probe_init(log, &np.gcells, &nb);
    Ok(nb)
}

/// `GCell::updateLocations` for a cluster: centred on its instances' bounding box (integer
/// centre), as wide and high as their total area at the core's aspect ratio (double, truncated
/// when the box is made).
fn cluster_box(c: &Common, members: &[usize], d: &Design) -> (i32, i32, i32, i32) {
    let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    let mut area = 0i64;
    for &m in members {
        let i = &c.insts[m];
        area += i64::from(i.dx()) * i64::from(i.dy());
        x0 = x0.min(i.lx);
        y0 = y0.min(i.ly);
        x1 = x1.max(i.ux);
        y1 = y1.max(i.uy);
    }
    let (cx, cy) = ((x0 + x1) / 2, (y0 + y1) / 2);
    let aspect = f64::from(d.core.dx()) / f64::from(d.core.dy());
    let height = (area as f64 / aspect).sqrt();
    let width = height * aspect;
    let (ax, ay) = ((f64::from(cx) - width / 2.0) as i32, (f64::from(cy) - height / 2.0) as i32);
    let (bx, by) = ((f64::from(cx) + width / 2.0) as i32, (f64::from(cy) + height / 2.0) as i32);
    (ax.min(bx), ay.min(by), ax.max(bx), ay.max(by))
}

/// `NesterovBase::getNesterovInstsArea`: standard cells, plus macros scaled by the target density
/// (`round` of a float).
fn nesterov_insts_area(nb: &Nb) -> i64 {
    nb.std_insts_area + (nb.macro_insts_area as f32 * nb.target_density).round() as i64
}

/// `initFillerGCells`: the filler size (the mean instance size between the 5th and 95th
/// percentiles of each dimension), the target density, the filler area and count, and the
/// fillers themselves at `mt19937(0)` positions inside the region.
fn init_filler_gcells(log: &mut Log, d: &Design, c: &Common, base: &Base, nb: &mut Nb, o: &Opts) -> Res<Vec<GCell>> {
    let mut dxs: Vec<i32> = base.place_insts.iter().map(|&k| c.insts[k].dx()).collect();
    let mut dys: Vec<i32> = base.place_insts.iter().map(|&k| c.insts[k].dy()).collect();
    dxs.sort_unstable();
    dys.sort_unstable();
    let n = dxs.len();
    let (mut lo, mut hi) = ((n as f64 * 0.05) as i32, (n as f64 * 0.95) as i32);
    if lo == hi {
        lo = 0;
        hi = n as i32;
    }
    if lo == hi {
        return refuse("a region with no placeable instance (no filler size) is not modelled");
    }
    let (mut sx, mut sy) = (0i64, 0i64);
    for i in lo..hi {
        sx += i64::from(dxs[i as usize]);
        sy += i64::from(dys[i as usize]);
    }
    nb.filler_dx = (sx / i64::from(hi - lo)) as i32;
    nb.filler_dy = (sy / i64::from(hi - lo)) as i32;
    nb.white_space_area = base.region_area - base.non_place_insts_area;
    let tmp_target = (f64::from(nb.std_insts_area as f32 / (nb.white_space_area - nb.macro_insts_area) as f32) + 0.01) as f32;
    nb.target_density = if o.uniform_density { tmp_target } else { o.density };
    let nia = nesterov_insts_area(nb);
    let mut movable = (nb.white_space_area as f32 * nb.target_density) as i64;
    let mut total_filler = movable - nia;
    nb.uniform_target_density = nia as f32 / nb.white_space_area as f32;
    nb.uniform_target_density = (nb.uniform_target_density * 100.0).ceil() / 100.0;
    if total_filler < 0 {
        log.warn(
            "GPL",
            302,
            format!("Target density {:.4} is too low for the available free area.\nAutomatically adjusting to uniform density {:.4}.", nb.target_density, nb.uniform_target_density),
        );
        nb.target_density = nb.uniform_target_density;
        movable = (nb.white_space_area as f32 * nb.target_density) as i64;
        total_filler = movable - nia;
    }
    // Limit the filler count: grow the filler size when there would be too many.
    let scale = (total_filler as f64 / (10.0 * nia as f64)).sqrt();
    if scale > 1.0 {
        let max_x = ((f64::from(d.core.dx()) / 1024.0) as i32).max(nb.filler_dx);
        let max_y = ((f64::from(d.core.dy()) / 1024.0) as i32).max(nb.filler_dy);
        nb.filler_dx = (f64::from(nb.filler_dx) * scale) as i32;
        nb.filler_dy = (f64::from(nb.filler_dy) * scale) as i32;
        nb.filler_dx = nb.filler_dx.min(max_x);
        nb.filler_dy = nb.filler_dy.min(max_y);
    }
    let count = (total_filler / i64::from(nb.filler_dx.wrapping_mul(nb.filler_dy))) as i32;
    let mut rng = Mt19937::new(0);
    let r = base.region_bbox;
    let mut fillers = Vec::with_capacity(count.max(0) as usize);
    for _ in 0..count {
        let (rx, ry) = (rng.next_u32(), rng.next_u32());
        // Unsigned arithmetic, as the reference's `randX % region_dx + region_lx`.
        let cx = ((rx % r.dx() as u32).wrapping_add(r.lx as u32)) as i32;
        let cy = ((ry % r.dy() as u32).wrapping_add(r.ly as u32)) as i32;
        fillers.push(GCell::filler(cx, cy, nb.filler_dx, nb.filler_dy));
    }
    Ok(fillers)
}

/// The constructor's perturbation (`calculatePlacementPerturbationOffset` per placed instance of
/// the region, in its order, with the region's own generator): two draws of a normal with
/// σ = 0.3072 · r, r the given distance (scaled by DBU/1000) or min(half a micron, a row); each
/// rounded; the instance and its pins moved, its cell rebuilt. A distance of exactly 0, or no row
/// height, draws nothing.
fn perturb_place_insts(d: &Design, c: &mut Common, ip: &mut Ip, base: &Base, inst_cell: &[Option<usize>], gcells: &mut [GCell], o: &Opts) {
    let dist = o.perturb_dist;
    let draws = c.site_y > 0 && dist != 0.0;
    let r = if dist > 0.0 { f64::from(dist) * (f64::from(d.dbu) / 1000.0) } else { (0.5 * f64::from(d.dbu)).min(f64::from(c.site_y)) };
    let sigma = r * 0.3071963263271184;
    let mut rng = Mt19937::new(o.seed as u32);
    for &k in &base.place_insts {
        let (xo, yo) = if draws {
            let dx = normal(&mut rng, 0.0, sigma);
            let dy = normal(&mut rng, 0.0, sigma);
            (dx.round() as i32, dy.round() as i32)
        } else {
            (0, 0)
        };
        let gi = inst_cell[k].expect("a placed instance");
        // A cluster is moved once, from its first instance; the others only drew.
        if gcells[gi].members.first().is_some_and(|&m| m != k) {
            continue;
        }
        let members = if gcells[gi].members.is_empty() { vec![k] } else { gcells[gi].members.clone() };
        // `Instance::setLocation(pb_inst->lx() + x_offset, ...)` for each instance of the cell, then
        // its pins (`Pin::updateLocation`) — `pb_inst->lx()` read anew each time, as the reference
        // reads it (the first instance has already moved when the next is set); a locked instance
        // stays where it is, though its offset was drawn (the stream stays in step).
        for &m in &members {
            if c.insts[m].locked {
                continue;
            }
            let (nx, ny) = (c.insts[k].lx + xo, c.insts[k].ly + yo);
            let i = &mut c.insts[m];
            let (w, h) = (i.dx(), i.dy());
            i.lx = nx;
            i.ly = ny;
            i.ux = i.lx + w;
            i.uy = i.ly + h;
            let (icx, icy) = (i.cx(), i.cy());
            for &p in &ip.inst_pins[m] {
                ip.pins[p].cx = icx + ip.pins[p].off_x;
                ip.pins[p].cy = icy + ip.pins[p].off_y;
            }
        }
        if !gcells[gi].members.is_empty() {
            let (lx, ly, ux, uy) = cluster_box(c, &members, d);
            let cell = &mut gcells[gi];
            cell.lx = lx;
            cell.ly = ly;
            cell.ux = ux;
            cell.uy = uy;
            cell.dlx = lx;
            cell.dly = ly;
            cell.dux = ux;
            cell.duy = uy;
            continue;
        }
        // `GCell::updateLocations` (one instance): the box, density box the same.
        let i = &c.insts[k];
        let cell = &mut gcells[inst_cell[k].expect("a placed instance")];
        cell.lx = i.lx;
        cell.ly = i.ly;
        cell.ux = i.ux;
        cell.uy = i.uy;
        cell.dlx = i.lx;
        cell.dly = i.ly;
        cell.dux = i.ux;
        cell.duy = i.uy;
    }
}

/// `BinGrid::initBins`: the bin count (a power of two per side, scaled by the aspect ratio, so that
/// a bin holds about one average instance at the target density), the bins, their fixed area.
fn init_bins(log: &mut Log, d: &Design, c: &Common, base: &Base, nb: &mut Nb, o: &Opts) {
    let r = base.region_bbox;
    let mut bg = BinGrid { lx: r.lx, ly: r.ly, ux: r.ux, uy: r.uy, ..BinGrid::default() };
    let total_bin_area = i64::from(r.ux - r.lx) * i64::from(r.uy - r.ly);
    let avg = if base.place_insts.is_empty() {
        log.warn("GPL", 306, "GPL component has no placed instances.");
        0
    } else {
        base.place_insts_area / base.place_insts.len() as i64
    };
    let ideal_bin_area = if nb.target_density != 0.0 { (avg as f32 / nb.target_density).round() as i64 } else { 0 };
    let ideal_cnt = if ideal_bin_area != 0 { (total_bin_area / ideal_bin_area) as i32 } else { 0 }.max(4);
    log.info("GPL", 23, format!("Placement target density:   {:10.4}", nb.target_density));
    log.info("GPL", 24, format!("Movable insts average area: {:10.3} um^2", dbu_area(d, avg as f64)));
    log.info("GPL", 25, format!("Ideal bin area:             {:10.3} um^2", dbu_area(d, ideal_bin_area as f64)));
    log.info("GPL", 26, format!("Ideal bin count:            {ideal_cnt:10}"));
    log.info("GPL", 27, format!("Total bin area:             {:10.3} um^2", dbu_area(d, total_bin_area as f64)));
    match o.bin_cnt {
        Some(n) => {
            bg.cnt_x = n;
            bg.cnt_y = n;
        }
        None => {
            let (w, h) = (r.ux - r.lx, r.uy - r.ly);
            let ratio = round_down_to_power_of_two((w.max(h) / w.min(h)) as u32) as i32;
            let mut found = 2;
            while found <= 1024 {
                if (found == 2 || found * (found * ratio) <= ideal_cnt) && 4 * found * (found * ratio) > ideal_cnt {
                    break;
                }
                found *= 2;
            }
            if w > h {
                bg.cnt_x = found * ratio;
                bg.cnt_y = found;
            } else {
                bg.cnt_x = found;
                bg.cnt_y = found * ratio;
            }
        }
    }
    log.info("GPL", 28, format!("{:21} {:7} , {:6}", "Bin count (X, Y):", bg.cnt_x, bg.cnt_y));
    bg.size_x = f64::from(r.ux - r.lx) / f64::from(bg.cnt_x);
    bg.size_y = f64::from(r.uy - r.ly) / f64::from(bg.cnt_y);
    log.info("GPL", 29, format!("{:21} {:7.3} * {:6.3} um", "Bin size (W * H):", bg.size_x / f64::from(d.dbu), bg.size_y / f64::from(d.dbu)));
    bg.bins = Vec::with_capacity((bg.cnt_x * bg.cnt_y) as usize);
    for y in 0..bg.cnt_y {
        for x in 0..bg.cnt_x {
            let b = Bin {
                lx: r.lx + (f64::from(x) * bg.size_x).round() as i32,
                ly: r.ly + (f64::from(y) * bg.size_y).round() as i32,
                ux: r.lx + (f64::from(x + 1) * bg.size_x).round() as i32,
                uy: r.ly + (f64::from(y + 1) * bg.size_y).round() as i32,
                target_density: nb.target_density,
                ..Bin::default()
            };
            if b.ux - b.lx < 0 || b.uy - b.ly < 0 {
                log.warn("GPL", 34, format!("Bin (center: {},{}, index: {}) has negative size: {}, {}", (b.lx + b.ux) / 2, (b.ly + b.uy) / 2, y * bg.cnt_x + x, b.ux - b.lx, b.uy - b.ly));
            }
            bg.bins.push(b);
        }
    }
    log.info("GPL", 30, format!("Number of bins:             {:10}", bg.bins.len()));
    nb.bg = bg;
    // Only initialised once.
    update_bins_non_place_area(c, base, nb, d.dbu);
}

/// The area a union of rectangles covers (coordinate compression; exact).
fn union_area(rects: &[(i32, i32, i32, i32)]) -> i64 {
    let mut xs: Vec<i32> = rects.iter().flat_map(|r| [r.0, r.2]).collect();
    xs.sort_unstable();
    xs.dedup();
    let mut area = 0i64;
    for w in xs.windows(2) {
        let (x0, x1) = (w[0], w[1]);
        let mut spans: Vec<(i32, i32)> = rects.iter().filter(|r| r.0 <= x0 && r.2 >= x1).map(|r| (r.1, r.3)).collect();
        spans.sort_unstable();
        let mut covered = 0i64;
        let mut cur: Option<(i32, i32)> = None;
        for (a, b) in spans {
            match cur {
                Some((ca, cb)) if a <= cb => cur = Some((ca, cb.max(b))),
                Some((ca, cb)) => {
                    covered += i64::from(cb - ca);
                    cur = Some((a, b));
                }
                None => cur = Some((a, b)),
            }
        }
        if let Some((ca, cb)) = cur {
            covered += i64::from(cb - ca);
        }
        area += covered * i64::from(x1 - x0);
    }
    area
}

/// `getOverlapArea`: a fixed instance's area in a bin — exact for a cell; for a macro, a bivariate
/// normal around its centre (σ = a sixth of the half size), never below the plain overlap and
/// never above 1.10 times it.
fn overlap_area(b: &Bin, i: &crate::placer_base::Instance, dbu: i32) -> i64 {
    let (lx, ly, ux, uy) = (b.lx.max(i.lx), b.ly.max(i.ly), b.ux.min(i.ux), b.uy.min(i.uy));
    if lx >= ux || ly >= uy {
        return 0;
    }
    if i.macro_ {
        let f = dbu as f32;
        let mean_x = (i.cx() - i.lx) as f32 / f;
        let mean_y = (i.cy() - i.ly) as f32 / f;
        let cdf = bivariate_normal_cdf(mean_x, mean_y, mean_x / 6.0, mean_y / 6.0, (lx - i.lx) as f32 / f, (ly - i.ly) as f32 / f, (ux - i.lx) as f32 / f, (uy - i.ly) as f32 / f);
        let original = (ux - lx) as f32 * (uy - ly) as f32;
        let scaled = cdf * (i.ux - i.lx) as f32 * (i.uy - i.ly) as f32;
        if scaled >= original {
            return scaled.min((f64::from(original) * 1.10) as f32) as i64;
        }
        return original as i64;
    }
    i64::from(ux - lx) * i64::from(uy - ly)
}

/// `BinGrid::updateBinsNonPlaceArea`: each bin's fixed area — the union of the fixed instances in
/// it (unscaled), and their per-instance areas scaled by the target density, capped at 1.10 times
/// the scaled union.
fn update_bins_non_place_area(c: &Common, base: &Base, nb: &mut Nb, dbu: i32) {
    let fixed: Vec<crate::placer_base::Instance> = base.fixed_insts.iter().map(|&k| c.insts[k].clone()).chain(base.dummies.iter().cloned()).collect();
    let bg = &mut nb.bg;
    for b in bg.bins.iter_mut() {
        b.non_place = 0;
        b.non_place_unscaled = 0;
    }
    let n = bg.bins.len();
    let mut touching: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (k, i) in fixed.iter().enumerate() {
        if i.lx >= i.ux || i.ly >= i.uy {
            continue;
        }
        let (x0, x1) = bg.idx_x(i.lx, i.ux);
        let (y0, y1) = bg.idx_y(i.ly, i.uy);
        for y in y0..y1 {
            for x in x0..x1 {
                touching[(y * bg.cnt_x + x) as usize].push(k);
            }
        }
    }
    let mut union = vec![0i64; n];
    for (b, t) in touching.iter().enumerate() {
        if t.is_empty() {
            continue;
        }
        let bin = &bg.bins[b];
        let rects: Vec<(i32, i32, i32, i32)> = t
            .iter()
            .map(|&k| (fixed[k].lx.max(bin.lx), fixed[k].ly.max(bin.ly), fixed[k].ux.min(bin.ux), fixed[k].uy.min(bin.uy)))
            .filter(|r| r.0 < r.2 && r.1 < r.3)
            .collect();
        union[b] = union_area(&rects);
        let td = bg.bins[b].target_density;
        bg.bins[b].non_place_unscaled = (union[b] as f32 * td) as i64;
    }
    let mut raw = vec![0i64; n];
    for i in &fixed {
        let (x0, x1) = bg.idx_x(i.lx, i.ux);
        let (y0, y1) = bg.idx_y(i.ly, i.uy);
        for y in y0..y1 {
            for x in x0..x1 {
                let b = (y * bg.cnt_x + x) as usize;
                raw[b] += overlap_area(&bg.bins[b], i, dbu);
            }
        }
    }
    for b in 0..n {
        if raw[b] != 0 {
            let td = bg.bins[b].target_density;
            bg.bins[b].non_place += (raw[b] as f32 * td) as i64;
        }
    }
    for b in 0..n {
        if touching[b].is_empty() {
            continue;
        }
        let cap = (union[b] as f32 * bg.bins[b].target_density * 1.10f32) as i64;
        if bg.bins[b].non_place > cap {
            bg.bins[b].non_place = cap;
        }
    }
}

/// `NesterovBase::updateDensitySize`: a cell of the region narrower than √2 bins is spread to √2
/// bins with its density scaled down to match (the comparison and the size in extended precision).
fn update_density_size(gcells: &mut [GCell], nb: &Nb) {
    let (bsx, bsy) = (nb.bg.size_x, nb.bg.size_y);
    let thr_x = ld::SQRT2.times(Ld::from_f64(bsx));
    let thr_y = ld::SQRT2.times(Ld::from_f64(bsy));
    let size_x = ld::SQRT2.times(Ld::from_f32(bsx as f32)).to_f32();
    let size_y = ld::SQRT2.times(Ld::from_f32(bsy as f32)).to_f32();
    for &gi in &nb.cells {
        let g = &mut gcells[gi];
        let (dx, dy) = (g.dx(), g.dy());
        let (scale_x, dsx) = if Ld::from_u64(dx as u64) < thr_x { (dx as f32 / thr_x.to_f32(), size_x) } else { (1.0, dx as f32) };
        let (scale_y, dsy) = if Ld::from_u64(dy as u64) < thr_y { (dy as f32 / thr_y.to_f32(), size_y) } else { (1.0, dy as f32) };
        g.set_density_size(dsx as i32, dsy as i32);
        g.density_scale = scale_x * scale_y;
    }
}

/// `vygDump("INIT")`: the bin grid, the fillers and every cell as the constructor leaves them.
fn probe_init(log: &mut Log, gcells: &[GCell], nb: &Nb) {
    let (mut hb, mut hd, mut hs, mut hg, mut hn) = (FNV0, FNV0, FNV0, FNV0, FNV0);
    for g in nb.cells.iter().map(|&gi| &gcells[gi]) {
        for v in [g.lx, g.ly, g.ux, g.uy] {
            hb = fnv(&v.to_le_bytes(), hb);
        }
        for v in [g.dlx, g.dly, g.dux, g.duy] {
            hd = fnv(&v.to_le_bytes(), hd);
        }
        hs = fnv(&g.density_scale.to_le_bytes(), hs);
    }
    for b in &nb.bg.bins {
        for v in [b.lx, b.ly, b.ux, b.uy] {
            hg = fnv(&v.to_le_bytes(), hg);
        }
        for v in [b.non_place, b.non_place_unscaled] {
            hn = fnv(&v.to_le_bytes(), hn);
        }
    }
    log.trace(format!(
        "VYGNB|INIT|{}|{}|{:016x}|{:016x}|{}|{}|{}|{:08x}|{:016x}|{:016x}|{:016x}|{:016x}|{:016x}",
        nb.bg.cnt_x,
        nb.bg.cnt_y,
        nb.bg.size_x.to_bits(),
        nb.bg.size_y.to_bits(),
        nb.filler_dx,
        nb.filler_dy,
        nb.fillers,
        nb.target_density.to_bits(),
        hb,
        hd,
        hs,
        hg,
        hn
    ));
}

/// Which slot a gradient update fills.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Slot {
    Prev,
    Cur,
    Next,
}

/// fmt's `{:W.Pe}`: a mantissa with P decimals, `e`, a sign and at least two exponent digits.
fn fmt_e(v: f64, prec: usize, width: usize) -> String {
    let s = format!("{v:.prec$e}");
    let (m, e) = s.split_once('e').unwrap_or((&s, "0"));
    let e: i32 = e.parse().unwrap_or(0);
    let t = format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
    format!("{t:>width$}")
}

/// fmt's `{:g}`: six significant digits, the shorter of fixed and exponent form, no trailing zeros.
fn fmt_g(v: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    let exp = v.abs().log10().floor() as i32;
    let strip = |s: String| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    };
    if !(-4..6).contains(&exp) {
        let s = format!("{v:.5e}");
        let (m, e) = s.split_once('e').unwrap_or((&s, "0"));
        let e: i32 = e.parse().unwrap_or(0);
        format!("{}e{}{:02}", strip(m.to_string()), if e < 0 { '-' } else { '+' }, e.abs())
    } else {
        let decimals = (5 - exp).max(0) as usize;
        strip(format!("{v:.decimals$}"))
    }
}
impl Nesterov {
    /// `GCell::setDensityCenterLocation` with its pins (`GPin::updateDensityLocation`) — region
    /// `b`'s cell `i`.
    fn set_density_center(&mut self, b: usize, i: usize, cx: i32, cy: i32) {
        let gi = self.nbs[b].cells[i];
        let g = &mut self.gcells[gi];
        let (hx, hy) = (g.d_dx() / 2, g.d_dy() / 2);
        g.dlx = cx - hx;
        g.dly = cy - hy;
        g.dux = cx + hx;
        g.duy = cy + hy;
        let (dcx, dcy) = (g.d_cx(), g.d_cy());
        for k in 0..self.gcells[gi].pins.len() {
            let p = self.gcells[gi].pins[k];
            self.gpins[p].cx = dcx + self.gpins[p].off_x;
            self.gpins[p].cy = dcy + self.gpins[p].off_y;
        }
    }

    /// `GCell::setDensityLocation` with its pins.
    fn set_density_location(&mut self, b: usize, i: usize, lx: i32, ly: i32) {
        let gi = self.nbs[b].cells[i];
        let g = &mut self.gcells[gi];
        g.dux = lx + (g.dux - g.dlx);
        g.duy = ly + (g.duy - g.dly);
        g.dlx = lx;
        g.dly = ly;
        let (dcx, dcy) = (g.d_cx(), g.d_cy());
        for k in 0..self.gcells[gi].pins.len() {
            let p = self.gcells[gi].pins[k];
            self.gpins[p].cx = dcx + self.gpins[p].off_x;
            self.gpins[p].cy = dcy + self.gpins[p].off_y;
        }
    }

    /// `updateDensityCoordiLayoutInside`: the density box moved inside the region's grid (float
    /// limits, truncated back to the integer box).
    fn update_density_coordi_layout_inside(&mut self, b: usize, i: usize) {
        let bg = &self.nbs[b].bg;
        let g = &self.gcells[self.nbs[b].cells[i]];
        let mut tlx = (g.dlx as f32).max(bg.lx as f32);
        let mut tly = (g.dly as f32).max(bg.ly as f32);
        if tlx + g.d_dx() as f32 > bg.ux as f32 {
            tlx = (bg.ux - g.d_dx()) as f32;
        }
        if tly + g.d_dy() as f32 > bg.uy as f32 {
            tly = (bg.uy - g.d_dy()) as f32;
        }
        self.set_density_location(b, i, tlx as i32, tly as i32);
    }

    /// `getDensityCoordiLayoutInsideX`.
    fn inside_x(&self, b: usize, i: usize, cx: f32) -> f32 {
        let bg = &self.nbs[b].bg;
        let half = self.gcells[self.nbs[b].cells[i]].d_dx() as f32 / 2.0f32;
        let mut adj = cx;
        if cx - half < bg.lx as f32 {
            adj = bg.lx as f32 + half;
        }
        if cx + half > bg.ux as f32 {
            adj = bg.ux as f32 - half;
        }
        adj
    }

    /// `getDensityCoordiLayoutInsideY`.
    fn inside_y(&self, b: usize, i: usize, cy: f32) -> f32 {
        let bg = &self.nbs[b].bg;
        let half = self.gcells[self.nbs[b].cells[i]].d_dy() as f32 / 2.0f32;
        let mut adj = cy;
        if cy - half < bg.ly as f32 {
            adj = bg.ly as f32 + half;
        }
        if cy + half > bg.uy as f32 {
            adj = bg.uy as f32 - half;
        }
        adj
    }

    /// `BinGrid::updateBinsGCellDensityArea`: each of the region's cells' density box scattered
    /// into its bins (integer accumulators, truncated addends), then per bin, in order, the density
    /// and the overflow — accumulated into an integer through a float, as the reference's `+=` does.
    fn update_bins_gcell_density_area(&mut self, b: usize) {
        let gcells = &self.gcells;
        let Nb { cells, bg, .. } = &mut self.nbs[b];
        for bin in bg.bins.iter_mut() {
            bin.inst_placed_unscaled = 0;
            bin.filler = 0;
        }
        for g in cells.iter().map(|&gi| &gcells[gi]) {
            let (x0, x1) = bg.density_idx_x(g);
            let (y0, y1) = bg.density_idx_y(g);
            if g.is_instance() {
                let macro_ = g.is_macro();
                if !macro_ && !g.is_std() {
                    continue;
                }
                for y in y0..y1 {
                    for x in x0..x1 {
                        let bin = &mut bg.bins[(y * bg.cnt_x + x) as usize];
                        let mut a = overlap_density_area(bin, g) * g.density_scale;
                        if macro_ {
                            a *= bin.target_density;
                        }
                        bin.inst_placed_unscaled += a as i64;
                    }
                }
            } else {
                for y in y0..y1 {
                    for x in x0..x1 {
                        let bin = &mut bg.bins[(y * bg.cnt_x + x) as usize];
                        bin.filler += (overlap_density_area(bin, g) * g.density_scale) as i64;
                    }
                }
            }
        }
        bg.overflow = 0;
        bg.overflow_unscaled = 0;
        for bin in bg.bins.iter_mut() {
            bin.inst_placed = bin.inst_placed_unscaled;
            let scaled = bin.area() as f32 * bin.target_density;
            bin.density = (bin.inst_placed as f32 + bin.filler as f32 + bin.non_place as f32) / scaled;
            let o = 0.0f32.max(bin.inst_placed as f32 + bin.non_place as f32 - scaled);
            bg.overflow = (bg.overflow as f32 + o) as i64;
            let ou = 0.0f32.max(bin.inst_placed_unscaled as f32 + bin.non_place_unscaled as f32 - scaled);
            bg.overflow_unscaled = (bg.overflow_unscaled as f32 + ou) as i64;
        }
    }

    /// `updateGCellDensityCenterLocation`: every cell's density centre (truncated), then the bins.
    fn update_gcell_density_center_location(&mut self, b: usize, coords: &[FloatPoint]) {
        for (i, c) in coords.iter().enumerate() {
            self.set_density_center(b, i, c.x as i32, c.y as i32);
        }
        self.update_bins_gcell_density_area(b);
    }

    /// `updateDensityFieldBin`: the bins' densities through the solver, back as potential and field.
    fn update_density_field_bin(&mut self, b: usize) {
        let Nb { bg, fft, .. } = &mut self.nbs[b];
        let ny = bg.cnt_y as usize;
        let cnt_x = bg.cnt_x as usize;
        for (k, bin) in bg.bins.iter().enumerate() {
            let (x, y) = (k % cnt_x, k / cnt_x);
            fft.density[x * ny + y] = bin.density;
        }
        fft.solve();
        for (k, bin) in bg.bins.iter_mut().enumerate() {
            let (x, y) = (k % cnt_x, k / cnt_x);
            bin.field_x = fft.field_x[x * ny + y];
            bin.field_y = fft.field_y[x * ny + y];
            bin.phi = fft.phi[x * ny + y];
        }
    }

    /// `NesterovBaseCommon::getHpwl`: every net's box from its pins, summed.
    fn hpwl(&mut self) -> i64 {
        let mut total = 0i64;
        for n in 0..self.gnets.len() {
            self.update_box(n);
            total += self.gnets[n].hpwl();
        }
        total
    }

    /// `GNet::updateBox`.
    fn update_box(&mut self, n: usize) {
        let (mut lx, mut ly, mut ux, mut uy) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        for &p in &self.gnets[n].pins {
            let q = &self.gpins[p];
            lx = lx.min(q.cx);
            ly = ly.min(q.cy);
            ux = ux.max(q.cx);
            uy = uy.max(q.cy);
        }
        let g = &mut self.gnets[n];
        g.lx = lx;
        g.ly = ly;
        g.ux = ux;
        g.uy = uy;
    }

    /// `updateWireLengthForceWA`: per net, per pin in the net's order, the shifted exponentials of
    /// the WA model (`fastExp`) where above the force bar, accumulated into the net.
    fn update_wirelength_force_wa(&mut self) {
        let (cx, cy) = (self.wl_coef_x, self.wl_coef_y);
        for p in self.gpins.iter_mut() {
            p.has_min_x = false;
            p.has_max_x = false;
            p.has_min_y = false;
            p.has_max_y = false;
            p.min_x = 0.0;
            p.max_x = 0.0;
            p.min_y = 0.0;
            p.max_y = 0.0;
        }
        for n in 0..self.gnets.len() {
            {
                let g = &mut self.gnets[n];
                g.min_x = 0.0;
                g.x_min_x = 0.0;
                g.max_x = 0.0;
                g.x_max_x = 0.0;
                g.min_y = 0.0;
                g.y_min_y = 0.0;
                g.max_y = 0.0;
                g.y_max_y = 0.0;
            }
            self.update_box(n);
            let (nlx, nly, nux, nuy) = (self.gnets[n].lx, self.gnets[n].ly, self.gnets[n].ux, self.gnets[n].uy);
            for k in 0..self.gnets[n].pins.len() {
                let pi = self.gnets[n].pins[k];
                let (px, py) = (self.gpins[pi].cx, self.gpins[pi].cy);
                let emin_x = (nlx - px) as f32 * cx;
                let emax_x = (px - nux) as f32 * cx;
                let emin_y = (nly - py) as f32 * cy;
                let emax_y = (py - nuy) as f32 * cy;
                if emin_x > MIN_WIRELENGTH_FORCE_BAR {
                    let v = fast_exp(emin_x);
                    self.gpins[pi].has_min_x = true;
                    self.gpins[pi].min_x = v;
                    self.gnets[n].min_x += v;
                    self.gnets[n].x_min_x += px as f32 * v;
                }
                if emax_x > MIN_WIRELENGTH_FORCE_BAR {
                    let v = fast_exp(emax_x);
                    self.gpins[pi].has_max_x = true;
                    self.gpins[pi].max_x = v;
                    self.gnets[n].max_x += v;
                    self.gnets[n].x_max_x += px as f32 * v;
                }
                if emin_y > MIN_WIRELENGTH_FORCE_BAR {
                    let v = fast_exp(emin_y);
                    self.gpins[pi].has_min_y = true;
                    self.gpins[pi].min_y = v;
                    self.gnets[n].min_y += v;
                    self.gnets[n].y_min_y += py as f32 * v;
                }
                if emax_y > MIN_WIRELENGTH_FORCE_BAR {
                    let v = fast_exp(emax_y);
                    self.gpins[pi].has_max_y = true;
                    self.gpins[pi].max_y = v;
                    self.gnets[n].max_y += v;
                    self.gnets[n].y_max_y += py as f32 * v;
                }
            }
        }
    }

    /// `getWireLengthGradientPinWA` (JingWei's thesis eq. 4.13): the `1.0 ± coef · x` terms and
    /// the whole quotient in double, the result a float.
    fn wl_gradient_pin(&self, p: usize, net: usize) -> FloatPoint {
        let q = &self.gpins[p];
        let g = &self.gnets[net];
        let (cx, cy) = (self.wl_coef_x, self.wl_coef_y);
        let (mut gmin_x, mut gmax_x, mut gmin_y, mut gmax_y) = (0f32, 0f32, 0f32, 0f32);
        if q.has_min_x {
            let num = f64::from(g.min_x) * (f64::from(q.min_x) * (1.0 - f64::from(cx * q.cx as f32))) + f64::from(cx * q.min_x * g.x_min_x);
            gmin_x = (num / f64::from(g.min_x * g.min_x)) as f32;
        }
        if q.has_max_x {
            let num = f64::from(g.max_x) * (f64::from(q.max_x) * (1.0 + f64::from(cx * q.cx as f32))) - f64::from(cx * q.max_x * g.x_max_x);
            gmax_x = (num / f64::from(g.max_x * g.max_x)) as f32;
        }
        if q.has_min_y {
            let num = f64::from(g.min_y) * (f64::from(q.min_y) * (1.0 - f64::from(cy * q.cy as f32))) + f64::from(cy * q.min_y * g.y_min_y);
            gmin_y = (num / f64::from(g.min_y * g.min_y)) as f32;
        }
        if q.has_max_y {
            let num = f64::from(g.max_y) * (f64::from(q.max_y) * (1.0 + f64::from(cy * q.cy as f32))) - f64::from(cy * q.max_y * g.y_max_y);
            gmax_y = (num / f64::from(g.max_y * g.max_y)) as f32;
        }
        FloatPoint::new(gmin_x - gmax_x, gmin_y - gmax_y)
    }

    /// `getWireLengthGradientWA`: the cell's pins in order, each weighted by its net (`gi`: the
    /// cell in the common store).
    fn wl_gradient(&self, gi: usize, pin_net: &[usize]) -> FloatPoint {
        let mut s = FloatPoint::default();
        for &p in &self.gcells[gi].pins {
            let n = pin_net[p];
            let mut t = self.wl_gradient_pin(p, n);
            let w = self.gnets[n].total_weight();
            t.x *= w;
            t.y *= w;
            s.x += t.x;
            s.y += t.y;
        }
        s
    }

    /// `getDensityGradient`: the field over the region's bins the density box covers (x outer,
    /// y inner), weighted by the scaled overlap.
    fn density_gradient(&self, b: usize, i: usize) -> FloatPoint {
        let bg = &self.nbs[b].bg;
        let g = &self.gcells[self.nbs[b].cells[i]];
        let (x0, x1) = bg.density_idx_x(g);
        let (y0, y1) = bg.density_idx_y(g);
        let mut f = FloatPoint::default();
        for x in x0..x1 {
            for y in y0..y1 {
                let bin = &bg.bins[(y * bg.cnt_x + x) as usize];
                let a = overlap_density_area(bin, g) * g.density_scale;
                f.x += a * bin.field_x;
                f.y += a * bin.field_y;
            }
        }
        f
    }
}

impl Nesterov {
    /// `NesterovBase::updateGradients` into one slot: every cell's WA and density gradient, the
    /// preconditioned sum, then (serially, in cell order) the two absolute sums.
    fn update_gradients(&mut self, b: usize, slot: Slot) {
        if self.nbs[b].is_converged {
            return;
        }
        let n = self.nbs[b].cells.len();
        let mut wl = vec![FloatPoint::default(); n];
        let mut den = vec![FloatPoint::default(); n];
        let mut sum = vec![FloatPoint::default(); n];
        for i in 0..n {
            wl[i] = self.wl_gradient(self.nbs[b].cells[i], &self.pin_net);
        }
        for i in 0..n {
            den[i] = self.density_gradient(b, i);
        }
        let p = self.nbs[b].density_penalty;
        for i in 0..n {
            let g = &self.gcells[self.nbs[b].cells[i]];
            sum[i].x = wl[i].x + p * den[i].x;
            sum[i].y = wl[i].y + p * den[i].y;
            let wl_pre = g.pins.len() as f32;
            let area = g.dx() as f32 * g.dy() as f32;
            let pre_x = (wl_pre + p * area).max(MIN_PRECONDITIONER);
            let pre_y = (wl_pre + p * area).max(MIN_PRECONDITIONER);
            sum[i].x /= pre_x;
            sum[i].y /= pre_y;
        }
        let nb = &mut self.nbs[b];
        nb.wl_grad_sum = 0.0;
        nb.density_grad_sum = 0.0;
        for i in 0..n {
            nb.wl_grad_sum += wl[i].x.abs();
            nb.wl_grad_sum += wl[i].y.abs();
            nb.density_grad_sum += den[i].x.abs();
            nb.density_grad_sum += den[i].y.abs();
        }
        match slot {
            Slot::Prev => {
                nb.prev_slp_wl = wl;
                nb.prev_slp_den = den;
                nb.prev_slp_sum = sum;
            }
            Slot::Cur => {
                nb.cur_slp_wl = wl;
                nb.cur_slp_den = den;
                nb.cur_slp_sum = sum;
            }
            Slot::Next => {
                nb.next_slp_wl = wl;
                nb.next_slp_den = den;
                nb.next_slp_sum = sum;
            }
        }
    }

    /// `NesterovPlace::npUpdate*Gradient(nb)`: a zero WA sum halves the coefficients and retries
    /// (up to ten times, a count shared by every region); a NaN or infinite sum is a divergence
    /// (`checkInvalidValues`, GPL-0306).
    fn np_update_gradient(&mut self, b: usize, slot: Slot) {
        self.update_gradients(b, slot);
        if self.nbs[b].wl_grad_sum == 0.0 && self.recursion_wl < MAX_RECURSION_WL_COEF {
            self.wl_coef_x = (f64::from(self.wl_coef_x) * 0.5) as f32;
            self.wl_coef_y = (f64::from(self.wl_coef_y) * 0.5) as f32;
            self.base_wl_coef = (f64::from(self.base_wl_coef) * 0.5) as f32;
            self.update_wirelength_force_wa();
            self.recursion_wl += 1;
            self.np_update_gradient(b, slot);
            return;
        }
        let bad = |v: f32| v.is_nan() || v.is_infinite();
        if bad(self.nbs[b].wl_grad_sum) || bad(self.nbs[b].density_grad_sum) {
            self.diverge = Some((306, "RePlAce diverged at wire/density gradient Sum. An internal value is NaN or Inf."));
            self.num_region_diverged = 1;
        }
    }

    /// `NesterovPlace::updateWireLengthCoef`: from the overflow (double arithmetic, `pow`), times
    /// the base coefficient.
    fn update_wirelength_coef(&mut self, overflow: f32) {
        let c = if overflow > 1.0 {
            0.1f64 as f32
        } else if f64::from(overflow) < 0.1 {
            10.0
        } else {
            (1.0 / 10f64.powf((f64::from(overflow) - 0.1) * 20.0 / 9.0 - 1.0)) as f32
        };
        self.wl_coef_x = c * self.base_wl_coef;
        self.wl_coef_y = c * self.base_wl_coef;
    }

    /// `initDensity1`: every cell inside the grid, the four coordinate slots at its density
    /// centre, the bins, the first HPWL, the field, the base WA coefficient and the overflow.
    fn init_density1(&mut self, b: usize) {
        let n = self.nbs[b].cells.len();
        {
            let nb = &mut self.nbs[b];
            for v in [&mut nb.cur_slp, &mut nb.next_slp, &mut nb.prev_slp, &mut nb.cur, &mut nb.next] {
                v.resize(n, FloatPoint::default());
            }
            for v in [&mut nb.cur_slp_wl, &mut nb.cur_slp_den, &mut nb.cur_slp_sum, &mut nb.next_slp_wl, &mut nb.next_slp_den, &mut nb.next_slp_sum, &mut nb.prev_slp_wl, &mut nb.prev_slp_den, &mut nb.prev_slp_sum] {
                v.resize(n, FloatPoint::default());
            }
            for v in [&mut nb.snapshot, &mut nb.snapshot_slp, &mut nb.snapshot_slp_sum, &mut nb.snapshot_prev_slp_sum] {
                v.resize(n, FloatPoint::default());
            }
        }
        for i in 0..n {
            self.update_density_coordi_layout_inside(b, i);
            let g = &self.gcells[self.nbs[b].cells[i]];
            let p = FloatPoint::new(g.d_cx() as f32, g.d_cy() as f32);
            let nb = &mut self.nbs[b];
            nb.cur_slp[i] = p;
            nb.prev_slp[i] = p;
            nb.cur[i] = p;
        }
        let cs = self.nbs[b].cur_slp.clone();
        self.update_gcell_density_center_location(b, &cs);
        let hpwl = self.hpwl();
        self.nbs[b].prev_hpwl = hpwl;
        self.update_density_field_bin(b);
        let init_wl = self.o.init_wirelength_coef;
        let nb = &mut self.nbs[b];
        nb.base_wl_coef_nb = (f64::from(init_wl) / (f64::from((nb.bg.size_x + nb.bg.size_y) as f32) * 0.5)) as f32;
        let nia = nesterov_insts_area(nb) as f32;
        nb.sum_overflow = nb.bg.overflow as f32 / nia;
        nb.sum_overflow_unscaled = nb.bg.overflow_unscaled as f32 / nia;
    }

    /// `updateInitialPrevSLPCoordi`: a step back along the current gradient, kept inside.
    fn update_initial_prev_slp_coordi(&mut self, b: usize) {
        let c = self.initial_prev_coordi_update_coef;
        for i in 0..self.nbs[b].cells.len() {
            // a locked cell's previous point is its current one
            if self.gcells[self.nbs[b].cells[i]].locked {
                self.nbs[b].prev_slp[i] = self.nbs[b].cur_slp[i];
                continue;
            }
            let px = self.nbs[b].cur_slp[i].x - c * self.nbs[b].cur_slp_sum[i].x;
            let py = self.nbs[b].cur_slp[i].y - c * self.nbs[b].cur_slp_sum[i].y;
            let q = FloatPoint::new(self.inside_x(b, i, px), self.inside_y(b, i, py));
            self.nbs[b].prev_slp[i] = q;
        }
    }

    /// `initDensity2`: the first penalty (WA sum over density sum, times the initial factor), the
    /// overflow, and the first step length (`getStepLength`: coordinate distance over gradient
    /// distance).
    fn init_density2(&mut self, b: usize) -> f32 {
        if self.nbs[b].wl_grad_sum == 0.0 {
            self.nbs[b].density_penalty = self.o.init_density_penalty;
            self.update_gradients(b, Slot::Prev);
        }
        let idp = self.o.init_density_penalty;
        let nb = &mut self.nbs[b];
        if nb.wl_grad_sum != 0.0 {
            nb.density_penalty = (nb.wl_grad_sum / nb.density_grad_sum) * idp;
        }
        let nia = nesterov_insts_area(nb) as f32;
        nb.sum_overflow = nb.bg.overflow as f32 / nia;
        nb.sum_overflow_unscaled = nb.bg.overflow_unscaled as f32 / nia;
        nb.coordi_distance = distance(&nb.prev_slp, &nb.cur_slp);
        let grad = distance(&nb.prev_slp_sum, &nb.cur_slp_sum);
        nb.step_length = nb.coordi_distance / grad;
        nb.step_length
    }

    /// A state probe (`vygDump`) of region `b`, in the instrumented reference's format.
    fn probe(&self, log: &mut Log, b: usize, tag: &str, iter: i32, a: f32, bb: f32) {
        let nb = &self.nbs[b];
        if tag == "NPINIT" && std::env::var_os("VYG_NP_CELLS").is_some() {
            for (i, g) in nb.cells.iter().map(|&gi| &self.gcells[gi]).enumerate() {
                let v = [nb.cur_slp_wl[i].x, nb.cur_slp_wl[i].y, nb.cur_slp_den[i].x, nb.cur_slp_den[i].y, nb.cur_slp_sum[i].x, nb.cur_slp_sum[i].y, nb.prev_slp_wl[i].x, nb.prev_slp_wl[i].y, nb.prev_slp_sum[i].x, nb.prev_slp_sum[i].y];
                let hex: Vec<String> = v.iter().map(|f| format!("{:08x}", f.to_bits())).collect();
                eprintln!("VYGNBC|{i}|{}|{}|{}|{}", g.pins.len(), g.dx(), g.dy(), hex.join("|"));
            }
        }
        let (mut hd, mut hf, mut hp) = (FNV0, FNV0, FNV0);
        for bin in &nb.bg.bins {
            hd = fnv(&bin.density.to_le_bytes(), hd);
            hf = fnv(&bin.field_x.to_le_bytes(), hf);
            hf = fnv(&bin.field_y.to_le_bytes(), hf);
            hp = fnv(&bin.phi.to_le_bytes(), hp);
        }
        log.trace(format!(
            "VYGNB|{tag}|{iter}|{:08x}|{:08x}|{:08x}|{:08x}|{:08x}|{:08x}|{:08x}|{:08x}|{}|{}|{}|{:016x}|{:016x}|{:016x}|{:016x}|{:016x}|{:016x}|{:016x}|{:016x}",
            a.to_bits(),
            bb.to_bits(),
            nb.step_length.to_bits(),
            nb.density_penalty.to_bits(),
            nb.sum_overflow.to_bits(),
            nb.sum_overflow_unscaled.to_bits(),
            nb.wl_grad_sum.to_bits(),
            nb.density_grad_sum.to_bits(),
            nb.prev_hpwl,
            nb.bg.overflow,
            nb.bg.overflow_unscaled,
            fnv_points(&nb.cur_slp),
            fnv_points(&nb.cur),
            fnv_points(&nb.prev_slp),
            fnv_points(&nb.cur_slp_sum),
            fnv_points(&nb.prev_slp_sum),
            hd,
            hf,
            hp
        ));
    }

    /// `NesterovPlace::init`: every region's density init, the WA coefficient from their mean
    /// overflow and mean base coefficient, per region the current gradient, an approximate
    /// previous point, its bins and field; the WA forces there; per region the previous gradient;
    /// then each region's first step length — a NaN or infinite one re-runs the whole init with a
    /// ten times larger back-step (and the loop stops: the outer init finishes after the inner).
    fn np_init(&mut self, log: &mut Log) -> Res<()> {
        let mut total_sum_overflow = 0.0f32;
        let mut total_base_wl_coef = 0.0f32;
        for b in 0..self.nbs.len() {
            self.init_density1(b);
            total_sum_overflow += self.nbs[b].sum_overflow;
            total_base_wl_coef += self.nbs[b].base_wl_coef_nb;
        }
        let n = self.nbs.len() as f32;
        self.average_overflow = total_sum_overflow / n;
        self.base_wl_coef = total_base_wl_coef / n;
        self.update_wirelength_coef(self.average_overflow);
        self.update_wirelength_force_wa();
        for b in 0..self.nbs.len() {
            self.np_update_gradient(b, Slot::Cur);
            self.update_initial_prev_slp_coordi(b);
            let p = self.nbs[b].prev_slp.clone();
            self.update_gcell_density_center_location(b, &p);
            self.update_density_field_bin(b);
        }
        self.update_wirelength_force_wa();
        for b in 0..self.nbs.len() {
            self.np_update_gradient(b, Slot::Prev);
        }
        for b in 0..self.nbs.len() {
            let step = self.init_density2(b);
            if (step.is_nan() || step.is_infinite()) && self.recursion_init_slp < MAX_RECURSION_INIT_SLP_COEF {
                self.initial_prev_coordi_update_coef *= 10.0;
                self.recursion_init_slp += 1;
                self.np_init(log)?;
                break;
            }
            if step.is_nan() || step.is_infinite() {
                return log.error("GPL", 304, format!("RePlAce diverged at initial iteration with steplength being {step}. Re-run with a smaller init_density_penalty value."));
            }
        }
        for b in 0..self.nbs.len() {
            self.probe(log, b, "NPINIT", -1, self.wl_coef_x, self.base_wl_coef);
        }
        Ok(())
    }

    /// `nesterovUpdateCoordinates`: the next point and the next look-ahead point, kept inside; the
    /// bins and the field at the look-ahead point.
    fn nesterov_update_coordinates(&mut self, b: usize, coeff: f32) {
        if self.nbs[b].is_converged {
            return;
        }
        let s = self.nbs[b].step_length;
        for k in 0..self.nbs[b].cells.len() {
            // a locked cell stays where it is
            if self.gcells[self.nbs[b].cells[k]].locked {
                let nb = &mut self.nbs[b];
                nb.next[k] = nb.cur[k];
                nb.next_slp[k] = nb.cur_slp[k];
                continue;
            }
            let nb = &self.nbs[b];
            let nx = nb.cur_slp[k].x + s * nb.cur_slp_sum[k].x;
            let ny = nb.cur_slp[k].y + s * nb.cur_slp_sum[k].y;
            let sx = nx + coeff * (nx - nb.cur[k].x);
            let sy = ny + coeff * (ny - nb.cur[k].y);
            let next = FloatPoint::new(self.inside_x(b, k, nx), self.inside_y(b, k, ny));
            let next_slp = FloatPoint::new(self.inside_x(b, k, sx), self.inside_y(b, k, sy));
            self.nbs[b].next[k] = next;
            self.nbs[b].next_slp[k] = next_slp;
        }
        let ns = self.nbs[b].next_slp.clone();
        self.update_gcell_density_center_location(b, &ns);
        self.update_density_field_bin(b);
    }

    /// `nesterovUpdateStepLength`: accept the new step when it is at most 0.95 of the last (in
    /// double); a step under 0.01 is set to 0.01 and not accepted.
    fn nesterov_update_step_length(&mut self, b: usize) -> bool {
        let nb = &mut self.nbs[b];
        if nb.is_converged {
            return true;
        }
        nb.coordi_distance = distance(&nb.cur_slp, &nb.next_slp);
        let grad = distance(&nb.cur_slp_sum, &nb.next_slp_sum);
        let new = nb.coordi_distance / grad;
        if new.is_nan() || new.is_infinite() {
            nb.is_diverged = true;
            return false;
        }
        if f64::from(new) > f64::from(nb.step_length) * 0.95 {
            nb.step_length = new;
            return false;
        }
        if f64::from(new) < 0.01 {
            nb.step_length = 0.01f64 as f32;
            return false;
        }
        nb.step_length = new;
        true
    }

    /// `NesterovPlace::doBackTracking`: up to ten trial steps, every region each time, until every
    /// region's step length settles.
    fn do_back_tracking(&mut self, log: &mut Log, coeff: f32) {
        for nbt in 0..MAX_BACK_TRACK {
            for b in 0..self.nbs.len() {
                self.nesterov_update_coordinates(b, coeff);
            }
            self.update_wirelength_force_wa();
            self.num_region_diverged = 0;
            for b in 0..self.nbs.len() {
                self.np_update_gradient(b, Slot::Next);
                self.num_region_diverged += i32::from(self.nbs[b].is_diverged);
            }
            if self.num_region_diverged > 0 {
                self.diverge = Some((306, "RePlAce diverged at wire/density gradient Sum."));
                break;
            }
            let mut ok = 0usize;
            self.num_region_diverged = 0;
            for b in 0..self.nbs.len() {
                ok += usize::from(self.nesterov_update_step_length(b));
                self.num_region_diverged += i32::from(self.nbs[b].is_diverged);
            }
            if self.num_region_diverged > 0 {
                self.diverge = Some((305, "RePlAce diverged during gradient descent calculation, resulting in an invalid step length (Inf or NaN). This is often caused by numerical instability or high placement density. Consider reducing placement density to potentially resolve the issue."));
                break;
            }
            for b in 0..self.nbs.len() {
                self.probe(log, b, "BT", nbt, self.wl_coef_x, coeff);
            }
            if ok != self.nbs.len() {
                break;
            }
        }
    }

    /// `nesterovAdjustPhi`: the maximum phi coefficient drops 1% once overflow is below 0.35, and
    /// never to 1.0 or below.
    fn nesterov_adjust_phi(&mut self, b: usize) {
        let nb = &mut self.nbs[b];
        if nb.is_converged {
            return;
        }
        if !nb.max_phi_coef_changed && nb.sum_overflow_unscaled < 0.35f32 {
            nb.max_phi_coef_changed = true;
            nb.max_phi_coef = (f64::from(nb.max_phi_coef) * 0.99) as f32;
        }
        if nb.max_phi_coef <= 1.0f32 {
            nb.max_phi_coef = 1.01f32;
        }
    }

    /// `getPhiCoef`: `max · max^(-Δ)` in double for a growing HPWL, `max` otherwise; at least `min`.
    fn phi_coef(&self, b: usize, scaled_diff_hpwl: f32) -> f32 {
        let m = self.nbs[b].max_phi_coef;
        let r = if scaled_diff_hpwl < 0.0 { f64::from(m) } else { f64::from(m) * f64::from(m).powf(-f64::from(scaled_diff_hpwl)) } as f32;
        self.o.min_phi_coef.max(r)
    }

    /// `NesterovBase::updateNextIter`: rotate the slots, the overflow over a floor that grows with
    /// the iteration, the progress row every tenth iteration (the header only from the top level;
    /// a region's row names it), the penalty times the phi coefficient, and the divergence
    /// watermark.
    fn nb_update_next_iter(&mut self, log: &mut Log, d: &Design, b: usize, iter: i32) {
        if self.nbs[b].is_converged {
            return;
        }
        let max_iter = self.max_iter;
        {
            let nb = &mut self.nbs[b];
            std::mem::swap(&mut nb.prev_slp, &mut nb.cur_slp);
            std::mem::swap(&mut nb.prev_slp_wl, &mut nb.cur_slp_wl);
            std::mem::swap(&mut nb.prev_slp_den, &mut nb.cur_slp_den);
            std::mem::swap(&mut nb.prev_slp_sum, &mut nb.cur_slp_sum);
            // "Prevent locked instances from moving": a locked instance cell's next state is the
            // (now) current one — between the two rotations, as the reference does it.
            for k in 0..nb.cells.len() {
                let g = &self.gcells[nb.cells[k]];
                if g.is_instance() && g.locked {
                    nb.next_slp[k] = nb.cur_slp[k];
                    nb.next_slp_wl[k] = nb.cur_slp_wl[k];
                    nb.next_slp_den[k] = nb.cur_slp_den[k];
                    nb.next_slp_sum[k] = nb.cur_slp_sum[k];
                    nb.next[k] = nb.cur[k];
                }
            }
            std::mem::swap(&mut nb.cur_slp, &mut nb.next_slp);
            std::mem::swap(&mut nb.cur_slp_wl, &mut nb.next_slp_wl);
            std::mem::swap(&mut nb.cur_slp_den, &mut nb.next_slp_den);
            std::mem::swap(&mut nb.cur_slp_sum, &mut nb.next_slp_sum);
            std::mem::swap(&mut nb.cur, &mut nb.next);
            let fraction = iter as f32 / max_iter as f32;
            let denom = (nesterov_insts_area(nb) as f32).max(fraction * nb.non_place_insts_area as f32 * 0.05f32);
            nb.sum_overflow = nb.bg.overflow as f32 / denom;
            nb.sum_overflow_unscaled = nb.bg.overflow_unscaled as f32 / denom;
        }
        let hpwl = self.hpwl();
        let nb = &mut self.nbs[b];
        if iter == 0 || iter % 10 == 0 {
            let mut pct = 0.0f32;
            if nb.prev_reported_hpwl != 0 {
                pct = ((hpwl - nb.prev_reported_hpwl) as f64 / nb.prev_reported_hpwl as f64 * 100.0) as f32;
            }
            nb.prev_reported_hpwl = hpwl;
            nb.prev_reported_overflow_unscaled = nb.sum_overflow_unscaled;
            let group = nb.group.as_ref().map(|g| format!(" ({g})")).unwrap_or_default();
            if iter == 0 && nb.group.is_none() {
                log.info("GPL", 31, "HPWL: Half-Perimeter Wirelength");
                log.report(format!("{:>9} | {:>8} | {:>13} | {:>8} | {:>9} | {:>5}", "Iteration", "Overflow", "HPWL (um)", "HPWL(%)", "Penalty", "Group"));
                log.report("---------------------------------------------------------------");
            }
            log.report(format!("{iter:9} | {:8.4} | {} | {:+7.2}% | {} | {:>5}", nb.sum_overflow_unscaled, fmt_e(hpwl as f64 / f64::from(d.dbu), 6, 13), pct, fmt_e(f64::from(nb.density_penalty), 2, 9), group));
        }
        let scaled = (hpwl - self.nbs[b].prev_hpwl) as f32 / self.reference_hpwl;
        let phi = self.phi_coef(b, scaled);
        let nb = &mut self.nbs[b];
        nb.density_penalty *= phi;
        nb.prev_hpwl = hpwl;
        if iter > 50 && nb.min_sum_overflow > nb.sum_overflow_unscaled {
            nb.min_sum_overflow = nb.sum_overflow_unscaled;
            nb.hpwl_with_min_sum_overflow = nb.prev_hpwl as f32;
        }
    }

    /// `NesterovPlace::updateNextIter`: every region's update, the WA coefficient from their mean
    /// overflow, and the minimum-HPWL watermark the divergence revert returns to.
    fn np_update_next_iter(&mut self, log: &mut Log, d: &Design, iter: i32) {
        let mut total = 0.0f32;
        let mut total_unscaled = 0.0f32;
        for b in 0..self.nbs.len() {
            self.nb_update_next_iter(log, d, b, iter);
            total += self.nbs[b].sum_overflow;
            total_unscaled += self.nbs[b].sum_overflow_unscaled;
        }
        let n = self.nbs.len() as f32;
        self.average_overflow = total / n;
        self.average_overflow_unscaled = total_unscaled / n;
        self.update_wirelength_coef(self.average_overflow);
        if !self.o.disable_revert_if_diverge {
            let hpwl = self.hpwl();
            if hpwl < self.min_hpwl && f64::from(self.average_overflow_unscaled) <= 0.25 {
                self.min_hpwl = hpwl;
                self.diverge_snapshot_overflow = self.average_overflow_unscaled;
                self.diverge_snapshot_iter = iter + 1;
                self.is_min_hpwl = true;
            } else {
                self.is_min_hpwl = false;
            }
        }
    }

    /// `saveSnapshot`.
    fn save_snapshot(&mut self, b: usize) {
        let nb = &mut self.nbs[b];
        if nb.is_converged {
            return;
        }
        nb.snapshot = nb.cur.clone();
        nb.snapshot_slp = nb.cur_slp.clone();
        nb.snapshot_slp_sum = nb.cur_slp_sum.clone();
        nb.snapshot_prev_slp_sum = nb.prev_slp_sum.clone();
        nb.snapshot_density_penalty = nb.density_penalty;
        nb.snapshot_step_length = nb.step_length;
    }

    /// `revertToSnapshot`.
    fn revert_to_snapshot(&mut self, b: usize) {
        if self.nbs[b].is_converged {
            return;
        }
        {
            let nb = &mut self.nbs[b];
            nb.cur = nb.snapshot.clone();
            nb.cur_slp = nb.snapshot_slp.clone();
            nb.cur_slp_sum = nb.snapshot_slp_sum.clone();
            nb.prev_slp_sum = nb.snapshot_prev_slp_sum.clone();
            nb.density_penalty = nb.snapshot_density_penalty;
            nb.step_length = nb.snapshot_step_length;
        }
        let c = self.nbs[b].cur.clone();
        self.update_gcell_density_center_location(b, &c);
        self.update_density_field_bin(b);
        let nb = &mut self.nbs[b];
        nb.is_diverged = false;
        nb.prev_reported_hpwl = 0;
        nb.prev_reported_overflow_unscaled = 0.0;
    }

    /// `NesterovBase::checkDivergence`.
    fn check_divergence(&mut self, log: &mut Log, b: usize) -> bool {
        let nb = &mut self.nbs[b];
        if nb.sum_overflow_unscaled < 0.2f32 && nb.sum_overflow_unscaled - nb.min_sum_overflow >= 0.02f32 && nb.hpwl_with_min_sum_overflow * 1.2f32 < nb.prev_hpwl as f32 {
            nb.is_diverged = true;
            log.warn("GPL", 323, "Divergence detected between consecutive iterations");
        }
        if nb.prev_reported_overflow_unscaled > 0.0 && nb.prev_reported_hpwl > 0 {
            let change = nb.sum_overflow_unscaled - nb.prev_reported_overflow_unscaled;
            let increase = (nb.prev_hpwl - nb.prev_reported_hpwl) as f32 / nb.prev_reported_hpwl as f32;
            if change >= 0.05f32 && increase >= 0.25f32 {
                nb.is_diverged = true;
                log.warn("GPL", 324, format!("Divergence detected between reported values. Overflow change: {}, HPWL increase: {}%.", fmt_g(f64::from(change)), fmt_g(f64::from(increase * 100.0f32))));
            }
        }
        nb.is_diverged
    }

    /// `NesterovPlace::isDiverged`: the snapshot of every region at each new minimum HPWL; on a
    /// divergence in any region, every region back to it (GPL-0998/0999) when one was saved, else
    /// an error at the end (GPL-0307).
    fn np_is_diverged(&mut self, log: &mut Log, saved: &mut bool, snap: &mut (f32, f32)) -> bool {
        self.num_region_diverged = 0;
        for b in 0..self.nbs.len() {
            self.num_region_diverged += i32::from(self.check_divergence(log, b));
        }
        if !self.o.disable_revert_if_diverge && self.num_region_diverged == 0 && self.is_min_hpwl {
            *snap = (self.wl_coef_x, self.wl_coef_y);
            for b in 0..self.nbs.len() {
                self.save_snapshot(b);
            }
            *saved = true;
        }
        if self.num_region_diverged > 0 {
            log.report(format!("Divergence occured in {} regions.", self.num_region_diverged));
            if !self.o.disable_revert_if_diverge && *saved {
                log.warn("GPL", 998, "Divergence detected, reverting to snapshot with min hpwl.");
                log.warn("GPL", 999, format!("Revert to iter: {:4} overflow: {:.3} HPWL: {}", self.diverge_snapshot_iter, self.diverge_snapshot_overflow, self.min_hpwl));
                self.wl_coef_x = snap.0;
                self.wl_coef_y = snap.1;
                self.update_wirelength_force_wa();
                for b in 0..self.nbs.len() {
                    self.revert_to_snapshot(b);
                }
                self.num_region_diverged = 0;
            } else {
                self.diverge = Some((307, "RePlAce divergence detected: Current overflow is low and increasing relative to minimum,and the HPWL has significantly worsened. Consider re-running with a smaller max_phi_cof value."));
            }
            return true;
        }
        false
    }

    /// `NesterovBase::checkConvergence`: at the target overflow, the closing row (a region's named),
    /// GPL-1001 (the top level) or GPL-1016 (a region), GPL-1002..1009, the region's instance cells
    /// locked.
    fn check_convergence(&mut self, log: &mut Log, d: &Design, b: usize, iter: i32) -> bool {
        if self.nbs[b].is_converged {
            return true;
        }
        if self.nbs[b].sum_overflow_unscaled > self.o.overflow {
            return false;
        }
        let hpwl = self.hpwl();
        let nb = &self.nbs[b];
        let group = nb.group.clone().unwrap_or_default();
        log.report(format!("{iter:9} | {:8.4} | {} | {:>8} | {} | {:>5}", nb.sum_overflow_unscaled, fmt_e(hpwl as f64 / f64::from(d.dbu), 6, 13), "", fmt_e(f64::from(nb.density_penalty), 2, 9), group));
        log.report("---------------------------------------------------------------");
        match &nb.group {
            Some(g) => log.info("GPL", 1016, format!("Region '{g}' placement finished at iteration {iter}")),
            None => log.info("GPL", 1001, format!("Global placement finished at iteration {iter}")),
        }
        let nia = nesterov_insts_area(nb);
        log.info("GPL", 1002, format!("Placed Cell Area            {:10.4}", dbu_area(d, nia as f64)));
        log.info("GPL", 1003, format!("Available Free Area         {:10.4}", dbu_area(d, nb.white_space_area as f64)));
        log.info("GPL", 1004, format!("Minimum Feasible Density        {:.4} (cell_area / free_area)", nb.uniform_target_density));
        log.info("GPL", 1006, "  Suggested Target Densities:");
        let w = nb.white_space_area as f64;
        log.info("GPL", 1007, format!("    - For 90% usage of free space: {:.4}", nia as f64 / (w * 0.90)));
        log.info("GPL", 1008, format!("    - For 80% usage of free space: {:.4}", nia as f64 / (w * 0.80)));
        if nia as f64 / (w * 0.50) <= 1.0 {
            log.info("GPL", 1009, format!("    - For 50% usage of free space: {:.4}", nia as f64 / (w * 0.50)));
        }
        if nb.uniform_target_density > 0.95f32 {
            log.warn("GPL", 1015, format!("High uniform density (>{:.2}) may cause congestion or legalization issues.", nb.uniform_target_density));
        }
        for k in 0..self.nbs[b].cells.len() {
            let gi = self.nbs[b].cells[k];
            if self.gcells[gi].is_instance() {
                self.gcells[gi].locked = true;
            }
        }
        self.nbs[b].is_converged = true;
        true
    }

    /// `NesterovPlace::isConverged`: every region asked, in order; converged when all are.
    fn np_is_converged(&mut self, log: &mut Log, d: &Design, iter: i32) -> bool {
        let mut converged = 0;
        for b in 0..self.nbs.len() {
            converged += usize::from(self.check_convergence(log, d, b, iter));
        }
        converged == self.nbs.len()
    }
}

/// Where Nesterov placement put each placed instance: (design instance, lower-left), as
/// `updateDbGCells` writes it — the density centre less half the (padded) instance, plus the
/// left padding.
pub fn placed_locations(c: &Common, nb: &Nesterov) -> Vec<(usize, i32, i32)> {
    let mut out = Vec::new();
    for g in nb.gcells.iter().filter(|g| g.inst.is_some()) {
        // every instance of a cluster at the cluster's centre
        let members = if g.members.is_empty() { vec![g.inst.expect("an instance cell")] } else { g.members.clone() };
        for k in members {
            let i = &c.insts[k];
            out.push((i.db.expect("a placed instance is a design instance"), g.d_cx() - i.dx() / 2 + c.site_x * nb.o.pad_left, g.d_cy() - i.dy() / 2));
        }
    }
    out
}

/// `Replace::doNesterovPlace` from GPL-0084 (the placer is built): the reference HPWL, the
/// placer's init, then the loop — `NesterovPlace::doNesterovPlace` — and its report. Returns the
/// final iteration.
pub fn do_nesterov_place(log: &mut Log, d: &Design, np: &mut Nesterov) -> Res<i32> {
    // `NesterovPlaceVars`: the reference HPWL, from the design's own (initial) HPWL when unset.
    let design_hpwl = np.hpwl();
    np.reference_hpwl = if np.o.reference_hpwl > 0.0 { np.o.reference_hpwl } else { REFERENCE_HPWL_FLOOR.max(REFERENCE_HPWL_FRACTION * design_hpwl as f32) };
    np.np_init(log)?;
    resume_nesterov_place(log, d, np, 0)
}

/// `Replace::doNesterovPlace` again on the placer that already exists (`initNesterovPlace` finds
/// it: no new placer, no init — only the target overflow and the iteration limit are set anew),
/// the loop from `start_iter`. Incremental placement's second pass.
pub fn continue_nesterov_place(log: &mut Log, d: &Design, np: &mut Nesterov, overflow: f32, max_iter: i32, start_iter: i32) -> Res<i32> {
    np.o.overflow = overflow;
    np.max_iter = max_iter;
    resume_nesterov_place(log, d, np, start_iter)
}

/// `PlacerBase::unlockAll` for every region: no cell (and no instance) locked.
pub fn unlock_all(np: &mut Nesterov, c: &mut Common) {
    for g in np.gcells.iter_mut().filter(|g| g.is_instance()) {
        g.locked = false;
    }
    for i in c.insts.iter_mut() {
        i.locked = false;
    }
}

/// `NesterovPlace::doNesterovPlace(start_iter)` with GPL-0084 before it.
fn resume_nesterov_place(log: &mut Log, d: &Design, np: &mut Nesterov, start_iter: i32) -> Res<i32> {
    log.info("GPL", 84, "---- Execute Nesterov Global Placement.");
    if np.num_region_diverged > 0 {
        let (code, msg) = np.diverge.unwrap_or((0, ""));
        return log.error("GPL", code, msg);
    }
    let mut original_area = 0i64;
    for nb in np.nbs.iter_mut() {
        nb.max_phi_coef_changed = false;
        nb.min_sum_overflow = 1e30;
        nb.hpwl_with_min_sum_overflow = 1e30;
        nb.is_converged = false;
        original_area += nesterov_insts_area(nb);
    }
    let (mut saved, mut snap) = (false, (0f32, 0f32));
    let mut cur_a = 1.0f32;
    let mut iter = start_iter;
    while iter < np.max_iter {
        let prev_a = cur_a;
        cur_a = ((1.0 + (4.0 * f64::from(prev_a) * f64::from(prev_a) + 1.0).sqrt()) * 0.5) as f32;
        let coeff = ((f64::from(prev_a) - 1.0) / f64::from(cur_a)) as f32;
        np.do_back_tracking(log, coeff);
        for b in 0..np.nbs.len() {
            np.nesterov_adjust_phi(b);
        }
        if np.num_region_diverged > 0 {
            break;
        }
        np.np_update_next_iter(log, d, iter);
        for b in 0..np.nbs.len() {
            np.probe(log, b, "IT", iter, np.wl_coef_x, cur_a);
        }
        if np.np_is_diverged(log, &mut saved, &mut snap) {
            break;
        }
        if np.np_is_converged(log, d, iter) {
            break;
        }
        iter += 1;
    }
    // `reportResults`.
    if iter >= np.max_iter {
        log.warn("GPL", 1010, format!("GPL reached the maximum number of iterations for nesterov {}. Placement may have failed to converge.", np.max_iter));
    }
    let new_area: i64 = np.nbs.iter().map(nesterov_insts_area).sum();
    let diff = (100.0 * (new_area - original_area) as f64 / original_area as f64) as f32;
    log.info("GPL", 1014, format!("Final placement area: {:.2} ({:+.2}%)", dbu_area(d, new_area as f64), diff));
    if np.num_region_diverged > 0 {
        let (code, msg) = np.diverge.unwrap_or((0, ""));
        return log.error("GPL", code, msg);
    }
    Ok(iter)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cluster cell is not its instances' bounding box: it is centred on that box's integer
    /// centre and holds their total area at the core's aspect ratio, each side truncated.
    #[test]
    fn cluster_box_is_the_total_area_at_the_core_aspect() {
        use crate::placer_base::{Instance, Rect};
        let inst = |lx, ly, ux, uy| Instance { db: None, lx, ly, ux, uy, fixed: false, macro_: false, locked: false };
        let c = Common { site_x: 1, site_y: 1, insts: vec![inst(0, 0, 10, 10), inst(31, 0, 41, 10)], nets: vec![], pins: 0 };
        let d = Design { core: Rect { lx: 0, ly: 0, ux: 200, uy: 100 }, ..Design::default() };
        // bbox (0,0)-(41,10): centre (20, 5); area 200 at aspect 2 is 20 x 10 — not 41 x 10
        assert_eq!(cluster_box(&c, &[0, 1], &d), (10, 0, 30, 10));
        // area 3: height sqrt(1.5) = 1.2247, width 2.4495 about (0, 0): each end truncated toward 0
        let c = Common { insts: vec![inst(-1, -1, 2, 0)], ..c };
        assert_eq!(cluster_box(&c, &[0], &d), (-1, 0, 1, 0));
    }

    /// Rule (`fastExp`): `(1 + x/1024)` squared ten times, in float — not `exp`.
    #[test]
    fn fast_exp_is_ten_squarings() {
        assert_eq!(fast_exp(0.0), 1.0);
        let mut e = 1.0f32 - 1.0 / 1024.0;
        for _ in 0..10 {
            e *= e;
        }
        assert_eq!(fast_exp(-1.0), e);
        assert_ne!(fast_exp(-1.0), (-1.0f32).exp(), "an approximation, and the bits are the approximation's");
    }

    /// Rule (the progress table, fmt `{:13.6e}` / `{:9.2e}`): a sign and two exponent digits.
    #[test]
    fn exponent_format_is_fmts() {
        assert_eq!(fmt_e(2124.479, 6, 13), " 2.124479e+03");
        assert_eq!(fmt_e(2.36e-11, 2, 9), " 2.36e-11");
        assert_eq!(fmt_g(0.0512), "0.0512");
        assert_eq!(fmt_g(1.5e-7), "1.5e-07");
    }

    /// Rule (`roundDownToPowerOfTwo`, the bin aspect ratio).
    #[test]
    fn aspect_ratio_rounds_down_to_a_power_of_two() {
        assert_eq!(round_down_to_power_of_two(1), 1);
        assert_eq!(round_down_to_power_of_two(3), 2);
        assert_eq!(round_down_to_power_of_two(1023), 512);
    }

    /// Rule (`updateBinsNonPlaceArea`): overlapping fixed cells count once in the union.
    #[test]
    fn union_area_counts_overlap_once() {
        assert_eq!(union_area(&[(0, 0, 10, 10), (5, 5, 15, 15)]), 175);
        assert_eq!(union_area(&[(0, 0, 10, 10), (0, 0, 10, 10)]), 100);
        assert_eq!(union_area(&[]), 0);
    }

    /// Rule (`getDistance`): a float sum, then `sqrt(sum / (2.0 · n))` in double.
    #[test]
    fn distance_is_rms_per_coordinate() {
        let a = [FloatPoint::new(0.0, 0.0), FloatPoint::new(3.0, 4.0)];
        let b = [FloatPoint::new(0.0, 0.0), FloatPoint::new(0.0, 0.0)];
        assert_eq!(distance(&a, &b), (25.0f64 / 4.0).sqrt() as f32);
    }
}
