// SPDX-License-Identifier: Apache-2.0
//! Initial placement: `InitialPlace::doBicgstabPlace` — every movable instance at a start point,
//! then up to `maxIter` rounds of the bound-to-bound net model solved by BiCGSTAB, x then y.
//!
//! One function per reference function, in call order:
//! - [`pins`] — the placer's pins as `PlacerBaseCommon` makes them (`Pin::updateCoordi`: an
//!   instance terminal at its shapes' centre, rotated with the instance about the master centre);
//! - [`place_insts_initial_positions`] — region centre, database location, or core centre;
//! - per round: [`update_pin_info`] (each net's min/max pins), [`create_sparse_matrix`] (the B2B
//!   triplets and right-hand sides), [`crate::bicgstab::bicgstab`] for x then y, the
//!   `[InitialPlace]` line (residual, HPWL of the DATABASE as it stands), [`update_coordi`].

use crate::bicgstab::{self, Csr};
use crate::log::Log;
use crate::placer_base::{Common, Design, Rect, Vars};

/// `InitialPlaceVars` (with the options' defaults).
#[derive(Debug, Clone, Copy)]
pub struct IpVars {
    pub max_iter: usize,
    pub min_diff_length: i32,
    pub max_solver_iter: usize,
    pub max_fanout: usize,
    pub net_weight_scale: f32,
    pub force_center: bool,
}

impl Default for IpVars {
    fn default() -> IpVars {
        IpVars { max_iter: 20, min_diff_length: 1500, max_solver_iter: 100, max_fanout: 200, net_weight_scale: 800.0, force_center: false }
    }
}

/// A placer pin: its instance in [`Common::insts`] (none for a port or an instance the placer
/// does not hold), its offset from the instance centre, its location, and its B2B marks.
#[derive(Debug, Clone, Default)]
pub struct Pin {
    pub inst: Option<usize>,
    pub port: bool,
    pub off_x: i32,
    pub off_y: i32,
    pub cx: i32,
    pub cy: i32,
    pub min_x: bool,
    pub max_x: bool,
    pub min_y: bool,
    pub max_y: bool,
}

/// The state initial placement works on.
#[derive(Debug, Clone)]
pub struct Ip {
    pub pins: Vec<Pin>,
    /// Per placer net (in [`Common::nets`] order), its pins: terminals, then ports.
    pub net_pins: Vec<Vec<usize>>,
    /// Each design instance's database box as it stands (moved by [`update_coordi`]).
    pub db_bbox: Vec<Rect>,
    /// `placeInsts()`: indices into [`Common::insts`] (their `extId` is the position here).
    pub place: Vec<usize>,
    /// Each placer instance's pins (`Instance::pins_`).
    pub inst_pins: Vec<Vec<usize>>,
}

/// `dbTransform` for an orientation applied to a point.
fn orient_point(orient: &str, x: i32, y: i32) -> (i32, i32) {
    match orient {
        "R90" => (-y, x),
        "R180" => (-x, -y),
        "R270" => (y, -x),
        "MY" => (-x, y),
        "MYR90" => (-y, -x),
        "MX" => (x, -y),
        "MXR90" => (y, x),
        _ => (x, y),
    }
}

/// `Pin::updateCoordi(dbITerm)`: the shapes' box rotated about the master centre with the
/// instance; the offset is its centre less the master centre; the pin sits at the instance's
/// database lower-left plus the master centre plus the offset.
fn iterm_pin(d: &Design, inst: usize, shape: Option<Rect>) -> (i32, i32, i32, i32) {
    let di = &d.insts[inst];
    let (icx, icy) = (di.master_w / 2, di.master_h / 2);
    let (ox, oy) = match shape {
        None => (0, 0),
        Some(r) => {
            let (ax, ay) = orient_point(&di.orient, r.lx - icx, r.ly - icy);
            let (bx, by) = orient_point(&di.orient, r.ux - icx, r.uy - icy);
            let (lx, ux) = (ax.min(bx) + icx, ax.max(bx) + icx);
            let (ly, uy) = (ay.min(by) + icy, ay.max(by) + icy);
            ((lx + ux) / 2 - icx, (ly + uy) / 2 - icy)
        }
    };
    (ox, oy, di.bbox.lx + icx + ox, di.bbox.ly + icy + oy)
}

/// The placer's pins and nets, as `PlacerBaseCommon::init` builds them.
pub fn pins(d: &Design, c: &Common, v: &Vars) -> Ip {
    let mut inst_map = vec![None; d.insts.len()];
    for (k, i) in c.insts.iter().enumerate() {
        if let Some(di) = i.db {
            inst_map[di] = Some(k);
        }
    }
    let mut ip = Ip { pins: Vec::new(), net_pins: Vec::new(), db_bbox: d.insts.iter().map(|i| i.bbox).collect(), place: Vec::new(), inst_pins: vec![Vec::new(); c.insts.len()] };
    for &(net, _) in &c.nets {
        let n = &d.nets[net];
        let mut list = Vec::new();
        for t in &n.iterms {
            let (off_x, off_y, cx, cy) = iterm_pin(d, t.inst, t.shape);
            list.push(ip.pins.len());
            if let Some(k) = inst_map[t.inst] {
                ip.inst_pins[k].push(ip.pins.len());
            }
            ip.pins.push(Pin { inst: inst_map[t.inst], off_x, off_y, cx, cy, ..Pin::default() });
        }
        if !v.skip_io {
            for p in &n.bterms {
                // updateCoordi(dbBTerm): the box centre (an unplaced port is refused earlier).
                let b = p.bbox.unwrap_or_default();
                list.push(ip.pins.len());
                ip.pins.push(Pin { port: true, cx: (b.lx + b.ux) / 2, cy: (b.ly + b.uy) / 2, ..Pin::default() });
            }
        }
        ip.net_pins.push(list);
    }
    // `Instance::pins_` follows the instance's own terminals (`dbInst::getITerms`), not the nets.
    let mut order = vec![0usize; ip.pins.len()];
    let mut k = 0;
    for &(net, _) in &c.nets {
        let n = &d.nets[net];
        for t in &n.iterms {
            order[k] = t.order;
            k += 1;
        }
        if !v.skip_io {
            k += n.bterms.len();
        }
    }
    for pins in ip.inst_pins.iter_mut() {
        pins.sort_by_key(|&p| order[p]);
    }
    ip.place = c.insts.iter().enumerate().filter(|(_, i)| !i.fixed).map(|(k, _)| k).collect();
    ip
}

/// `Instance::setCenterLocation` with `Pin::updateLocation` for its pins.
fn set_center_location(c: &mut Common, ip: &mut Ip, k: usize, x: i32, y: i32) {
    let i = &mut c.insts[k];
    let (hx, hy) = ((i.ux - i.lx) / 2, (i.uy - i.ly) / 2);
    i.lx = x - hx;
    i.ly = y - hy;
    i.ux = x + hx;
    i.uy = y + hy;
    let (cx, cy) = (i.cx(), i.cy());
    for &pk in &ip.inst_pins[k] {
        let p = &mut ip.pins[pk];
        p.cx = cx + p.off_x;
        p.cy = cy + p.off_y;
    }
}

/// `placeInstsInitialPositions`: a grouped instance at its region's centre, a placed one (unless
/// forced) at its database centre, the rest at the core centre; GPL-0051.
pub fn place_insts_initial_positions(log: &mut Log, d: &Design, c: &mut Common, ip: &mut Ip, vars: &IpVars) {
    let (ccx, ccy) = ((d.core.lx + d.core.ux) / 2, (d.core.ly + d.core.uy) / 2);
    let (mut region, mut db, mut core) = (0, 0, 0);
    for &k in &ip.place.clone() {
        // a locked instance keeps its place and is counted nowhere
        if c.insts[k].locked {
            continue;
        }
        let di = &d.insts[c.insts[k].db.expect("a placed instance is a design instance")];
        let reg = di.group.and_then(|g| d.groups[g].region);
        if let Some(r) = reg {
            let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
            for b in &d.regions[r].boundaries {
                x0 = x0.min(b.lx);
                y0 = y0.min(b.ly);
                x1 = x1.max(b.ux);
                y1 = y1.max(b.uy);
            }
            set_center_location(c, ip, k, x1 - (x1 - x0) / 2, y1 - (y1 - y0) / 2);
            region += 1;
        } else if !vars.force_center && di.placed {
            let b = di.bbox;
            set_center_location(c, ip, k, (b.lx + b.ux) / 2, (b.ly + b.uy) / 2);
            db += 1;
        } else {
            set_center_location(c, ip, k, ccx, ccy);
            core += 1;
        }
    }
    log.info("GPL", 51, format!("Source of initial instance position counters:\n\tOdb location = {db}\tCore center = {core}\tRegion center = {region}"));
}

/// `updatePinInfo`: per net, the first pin at the least and greatest x and y (strictly better
/// replaces), marked.
pub fn update_pin_info(ip: &mut Ip) {
    for p in ip.pins.iter_mut() {
        p.min_x = false;
        p.max_x = false;
        p.min_y = false;
        p.max_y = false;
    }
    for net in &ip.net_pins {
        let (mut lx, mut ly, mut ux, mut uy) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        let (mut pmin_x, mut pmin_y, mut pmax_x, mut pmax_y): (Option<usize>, Option<usize>, Option<usize>, Option<usize>) = (None, None, None, None);
        for &k in net {
            let (cx, cy) = (ip.pins[k].cx, ip.pins[k].cy);
            if lx > cx {
                if let Some(o) = pmin_x {
                    ip.pins[o].min_x = false;
                }
                lx = cx;
                pmin_x = Some(k);
                ip.pins[k].min_x = true;
            }
            if ux < cx {
                if let Some(o) = pmax_x {
                    ip.pins[o].max_x = false;
                }
                ux = cx;
                pmax_x = Some(k);
                ip.pins[k].max_x = true;
            }
            if ly > cy {
                if let Some(o) = pmin_y {
                    ip.pins[o].min_y = false;
                }
                ly = cy;
                pmin_y = Some(k);
                ip.pins[k].min_y = true;
            }
            if uy < cy {
                if let Some(o) = pmax_y {
                    ip.pins[o].max_y = false;
                }
                uy = cy;
                pmax_y = Some(k);
                ip.pins[k].max_y = true;
            }
        }
    }
}

/// One axis's B2B system: the triplets, the right-hand side and the start point.
pub struct System {
    pub triplets: Vec<(usize, usize, f32)>,
    pub rhs: Vec<f32>,
    pub x: Vec<f32>,
}

/// `createSparseMatrix`: for every net of 2..maxFanout pins, every pin pair on different
/// instances with a min/max pin among them: weight `scale / (pins − 1) / max(distance,
/// minDiffLength)`; movable pairs couple, a fixed side moves to the right-hand side.
pub fn create_sparse_matrix(c: &Common, ip: &Ip, vars: &IpVars) -> (System, System) {
    let n = ip.place.len();
    let mut ext = vec![usize::MAX; c.insts.len()];
    for (e, &k) in ip.place.iter().enumerate() {
        ext[k] = e;
    }
    let mut sx = System { triplets: Vec::new(), rhs: vec![0.0; n], x: vec![0.0; n] };
    let mut sy = System { triplets: Vec::new(), rhs: vec![0.0; n], x: vec![0.0; n] };
    for (e, &k) in ip.place.iter().enumerate() {
        sx.x[e] = c.insts[k].cx() as f32;
        sy.x[e] = c.insts[k].cy() as f32;
        // A locked instance is its own row, pinned at its centre (its triplets first, in place
        // order, before any net's).
        if c.insts[k].locked {
            sx.triplets.push((e, e, 1.0));
            sy.triplets.push((e, e, 1.0));
            sx.rhs[e] = c.insts[k].cx() as f32;
            sy.rhs[e] = c.insts[k].cy() as f32;
        }
    }
    // A pin's instance, when the placer moves it (`isPlaceInstConnected`).
    let placed = |p: &Pin| p.inst.filter(|&k| !c.insts[k].fixed);
    for net in &ip.net_pins {
        if net.len() <= 1 || net.len() >= vars.max_fanout {
            continue;
        }
        let net_weight = vars.net_weight_scale / (net.len() - 1) as f32;
        for i1 in 1..net.len() {
            let p1 = &ip.pins[net[i1]];
            for &k2 in &net[..i1] {
                let p2 = &ip.pins[k2];
                // The same instance (two ports share "none").
                if p1.inst == p2.inst {
                    continue;
                }
                for axis in 0..2 {
                    let (marked, c1, c2, sys) = if axis == 0 {
                        (p1.min_x || p1.max_x || p2.min_x || p2.max_x, p1.cx, p2.cx, &mut sx)
                    } else {
                        (p1.min_y || p1.max_y || p2.min_y || p2.max_y, p1.cy, p2.cy, &mut sy)
                    };
                    if !marked {
                        continue;
                    }
                    let center = |k: usize| if axis == 0 { c.insts[k].cx() } else { c.insts[k].cy() };
                    let diff = (c1 - c2).abs();
                    let weight = if diff > vars.min_diff_length { net_weight / diff as f32 } else { net_weight / vars.min_diff_length as f32 };
                    match (placed(p1), placed(p2)) {
                        (Some(a), Some(b)) => match (c.insts[a].locked, c.insts[b].locked) {
                            (false, false) => {
                                let (e1, e2) = (ext[a], ext[b]);
                                sys.triplets.push((e1, e1, weight));
                                sys.triplets.push((e2, e2, weight));
                                sys.triplets.push((e1, e2, -weight));
                                sys.triplets.push((e2, e1, -weight));
                                sys.rhs[e1] += -weight * ((c1 - center(a)) - (c2 - center(b))) as f32;
                                sys.rhs[e2] += -weight * ((c2 - center(b)) - (c1 - center(a))) as f32;
                            }
                            // one locked: its pin is a fixed terminal for the other
                            (true, false) => {
                                let e2 = ext[b];
                                sys.triplets.push((e2, e2, weight));
                                sys.rhs[e2] += weight * (c1 - (c2 - center(b))) as f32;
                            }
                            (false, true) => {
                                let e1 = ext[a];
                                sys.triplets.push((e1, e1, weight));
                                sys.rhs[e1] += weight * (c2 - (c1 - center(a))) as f32;
                            }
                            // both locked: nothing
                            (true, true) => {}
                        },
                        (None, Some(b)) => {
                            let e2 = ext[b];
                            sys.triplets.push((e2, e2, weight));
                            sys.rhs[e2] += weight * (c1 - (c2 - center(b))) as f32;
                        }
                        (Some(a), None) => {
                            let e1 = ext[a];
                            sys.triplets.push((e1, e1, weight));
                            sys.rhs[e1] += weight * (c2 - (c1 - center(a))) as f32;
                        }
                        (None, None) => {}
                    }
                }
            }
        }
    }
    (sx, sy)
}

/// `Net::updateBox` + `getHpwl` over every placer net: the database boxes of its terminals'
/// instances, and its ports' pin boxes unless `-skip_io`.
pub fn hpwl(d: &Design, c: &Common, ip: &Ip, v: &Vars) -> i64 {
    let mut total = 0i64;
    for &(net, _) in &c.nets {
        let n = &d.nets[net];
        let (mut lx, mut ly, mut ux, mut uy) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        for t in &n.iterms {
            let b = ip.db_bbox[t.inst];
            lx = lx.min(b.lx);
            ly = ly.min(b.ly);
            ux = ux.max(b.ux);
            uy = uy.max(b.uy);
        }
        if !v.skip_io {
            for p in &n.bterms {
                if let Some(b) = p.bbox {
                    lx = lx.min(b.lx);
                    ly = ly.min(b.ly);
                    ux = ux.max(b.ux);
                    uy = uy.max(b.uy);
                }
            }
        }
        if ux >= lx {
            total += i64::from(ux - lx) + i64::from(uy - ly);
        }
    }
    total
}

/// `updateCoordi`: each movable instance's solution, truncated, clamped to the core (and its
/// region's box), as its centre; the database moved with it (`dbSetCenterLocation`: the placer
/// box's lower-left becomes the database location).
pub fn update_coordi(d: &Design, c: &mut Common, ip: &mut Ip, sx: &[f32], sy: &[f32]) {
    for (e, &k) in ip.place.clone().iter().enumerate() {
        if c.insts[k].locked {
            continue;
        }
        let mut nx = sx[e] as i32;
        let mut ny = sy[e] as i32;
        nx = nx.max(d.core.lx).min(d.core.ux);
        ny = ny.max(d.core.ly).min(d.core.uy);
        let di = c.insts[k].db.expect("a placed instance is a design instance");
        if let Some(r) = d.insts[di].group.and_then(|g| d.groups[g].region) {
            let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
            for b in &d.regions[r].boundaries {
                x0 = x0.min(b.lx);
                y0 = y0.min(b.ly);
                x1 = x1.max(b.ux);
                y1 = y1.max(b.uy);
            }
            nx = nx.max(x0).min(x1);
            ny = ny.max(y0).min(y1);
        }
        set_center_location(c, ip, k, nx, ny);
        let (lx, ly) = (c.insts[k].lx, c.insts[k].ly);
        let b = ip.db_bbox[di];
        ip.db_bbox[di] = Rect { lx, ly, ux: lx + b.dx(), uy: ly + b.dy() };
    }
}

/// `doBicgstabPlace`: GPL-0005, the start points, then the rounds.
pub fn do_bicgstab_place(log: &mut Log, d: &Design, c: &mut Common, v: &Vars, vars: &IpVars) -> Ip {
    log.info("GPL", 5, "---- Execute Conjugate Gradient Initial Placement.");
    let mut ip = pins(d, c, v);
    place_insts_initial_positions(log, d, c, &mut ip, vars);
    for iter in 1..=vars.max_iter {
        update_pin_info(&mut ip);
        let (mut sx, mut sy) = create_sparse_matrix(c, &ip, vars);
        let n = ip.place.len();
        let mx = Csr::from_triplets(n, &sx.triplets);
        let ex = bicgstab::bicgstab(&mx, &sx.rhs, &mut sx.x, vars.max_solver_iter);
        let my = Csr::from_triplets(n, &sy.triplets);
        let ey = bicgstab::bicgstab(&my, &sy.rhs, &mut sy.x, vars.max_solver_iter);
        log.trace(format!("VYGIPS|{iter}|x|{}|{:08x}|{}|{:016x}", ex.iterations, ex.error.to_bits(), if ex.converged { 0 } else { 2 }, fnv(&sx.x)));
        log.trace(format!("VYGIPS|{iter}|y|{}|{:08x}|{}|{:016x}", ey.iterations, ey.error.to_bits(), if ey.converged { 0 } else { 2 }, fnv(&sy.x)));
        if ex.error.is_nan() || ey.error.is_nan() {
            log.warn("GPL", 325, format!("Conjugate gradient initial placement solver failed at iteration {iter}. "));
            break;
        }
        let error_max = ex.error.max(ey.error);
        log.report(format!("[InitialPlace]  Iter: {iter} conjugate gradient residual: {error_max:.8} HPWL: {}", hpwl(d, c, &ip, v)));
        update_coordi(d, c, &mut ip, &sx.x, &sy.x);
        if f64::from(error_max) <= 1e-5 && iter >= 5 {
            break;
        }
    }
    ip
}

/// FNV-1a over the floats' bytes (the instrumented reference's checksum).
pub fn fnv(v: &[f32]) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for f in v {
        for b in f.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(1099511628211);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    // Rule (dbTransform): the eight orientations as odb applies them to a point.
    #[test]
    fn orientations_rotate_and_mirror() {
        assert_eq!(orient_point("R90", 1, 2), (-2, 1));
        assert_eq!(orient_point("MYR90", 1, 2), (-2, -1));
        assert_eq!(orient_point("MXR90", 1, 2), (2, 1));
        assert_eq!(orient_point("MX", 1, 2), (1, -2));
    }
}
