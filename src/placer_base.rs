// SPDX-License-Identifier: Apache-2.0
//! The placer's view of the design: `PlacerBaseCommon` (instances, nets, pins — once) and one
//! `PlacerBase` per region (the top level, then each region's groups).
//!
//! One function per reference function, in call order:
//! - [`placer_base_common`] — `PlacerBaseCommon::init`: die and core, the site, the instances
//!   (DPL's ignore rule; fixed ones snapped outward; padding), the pin-density area adjustment,
//!   the signal and clock nets with their pins;
//! - [`placer_base`] — `PlacerBase::init`: the region box, movable and fixed instances,
//!   [`init_insts_for_unusable_sites`] (dummy instances), then [`print_info`].
//!
//! The design is read into plain records first ([`Design`]), so every rule here is testable
//! without a database.

use crate::log::{Log, Res};

/// A rectangle, `[lx, ux) × [ly, uy)` in DBU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rect {
    pub lx: i32,
    pub ly: i32,
    pub ux: i32,
    pub uy: i32,
}

impl Rect {
    pub fn dx(&self) -> i32 {
        self.ux - self.lx
    }
    pub fn dy(&self) -> i32 {
        self.uy - self.ly
    }
    pub fn area(&self) -> i64 {
        i64::from(self.dx()) * i64::from(self.dy())
    }
    /// `Rect::intersect`: the overlap, or an empty rectangle at the origin when there is none.
    pub fn intersect(&self, o: &Rect) -> Rect {
        let r = Rect { lx: self.lx.max(o.lx), ly: self.ly.max(o.ly), ux: self.ux.min(o.ux), uy: self.uy.min(o.uy) };
        if r.lx > r.ux || r.ly > r.uy {
            Rect::default()
        } else {
            r
        }
    }
    pub fn contains(&self, o: &Rect) -> bool {
        self.lx <= o.lx && self.ly <= o.ly && o.ux <= self.ux && o.uy <= self.uy
    }
}

/// A database instance as the placer reads it.
#[derive(Debug, Clone)]
pub struct DbInst {
    pub name: String,
    pub master: String,
    /// `dbMaster::isCoreAutoPlaceable`.
    pub core_auto_placeable: bool,
    /// `getType().isBlock()`.
    pub master_is_block: bool,
    /// `getBBox()`: its lower-left is `getLocation()`.
    pub bbox: Rect,
    /// `getPlacementStatus().isFixed()` (FIRM, LOCKED, COVER).
    pub fixed: bool,
    /// The group it belongs to (index into [`Design::groups`]).
    pub group: Option<usize>,
    /// `getTransformedHalo()`, when it has a halo.
    pub halo: Option<Rect>,
    /// Its terminals that are not supply (`!getSigType().isSupply()`).
    pub signal_pins: usize,
    /// `getOrient()` (`R0`, `R90`, … `MXR90`) and the master's width and height.
    pub orient: String,
    pub master_w: i32,
    pub master_h: i32,
    /// `isPlaced()` (PLACED, FIRM, LOCKED, COVER).
    pub placed: bool,
}

#[derive(Debug, Clone)]
pub struct Group {
    pub name: String,
    /// `VISUAL_DEBUG` groups are ignored.
    pub visual_debug: bool,
    pub region: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct Region {
    pub boundaries: Vec<Rect>,
    /// Its groups, in `getGroups()` order.
    pub groups: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Blockage {
    pub bbox: Rect,
    /// The instance it belongs to, and whether that instance is fixed.
    pub inst: Option<(String, bool)>,
    pub max_density: f32,
}

/// One block port on a placed net: its name and pin bounding box (`None` when unplaced).
#[derive(Debug, Clone)]
pub struct Port {
    pub name: String,
    pub bbox: Option<Rect>,
}

/// One instance terminal on a net: its instance (index into [`Design::insts`]) and the union of
/// its master terminal's pin shapes, in master coordinates (`None` without shapes).
#[derive(Debug, Clone)]
pub struct Term {
    pub inst: usize,
    pub shape: Option<Rect>,
}

#[derive(Debug, Clone)]
pub struct DbNet {
    pub name: String,
    /// `SIGNAL` or `CLOCK` (the others are skipped).
    pub signal_or_clock: bool,
    /// The net's terminals in `getITerms()` order.
    pub iterms: Vec<Term>,
    pub bterms: Vec<Port>,
}

/// What `PlacerBaseCommon` and `PlacerBase` read.
#[derive(Debug, Clone, Default)]
pub struct Design {
    /// The technology's DBU per micron.
    pub dbu: i32,
    pub die: Rect,
    pub core: Rect,
    /// Each row's box and its site: class is PAD, width, height.
    pub rows: Vec<(Rect, bool, i32, i32)>,
    pub insts: Vec<DbInst>,
    pub nets: Vec<DbNet>,
    pub groups: Vec<Group>,
    /// In `getRegions()` order.
    pub regions: Vec<Region>,
    pub blockages: Vec<Blockage>,
}

/// The options construction reads.
#[derive(Debug, Clone, Copy, Default)]
pub struct Vars {
    pub pad_left: i32,
    pub pad_right: i32,
    pub skip_io: bool,
    pub place_ios: bool,
    pub disable_pin_density_adjust: bool,
}

/// A placer instance (`gpl::Instance`): a design instance (`db`) or a dummy.
#[derive(Debug, Clone, PartialEq)]
pub struct Instance {
    pub db: Option<usize>,
    pub lx: i32,
    pub ly: i32,
    pub ux: i32,
    pub uy: i32,
    pub fixed: bool,
    pub macro_: bool,
}

impl Instance {
    pub fn dx(&self) -> i32 {
        self.ux - self.lx
    }
    pub fn dy(&self) -> i32 {
        self.uy - self.ly
    }
    pub fn cx(&self) -> i32 {
        (self.lx + self.ux) / 2
    }
    pub fn cy(&self) -> i32 {
        (self.ly + self.uy) / 2
    }
    pub fn area(&self) -> i64 {
        i64::from(self.dx()) * i64::from(self.dy())
    }
    pub fn is_place_instance(&self) -> bool {
        self.db.is_some() && !self.fixed
    }

    /// `Instance::extendSizeByScale`: about its centre, each side × scale (truncated); the
    /// area change.
    fn extend_size_by_scale(&mut self, scale: f64) -> i64 {
        let old = self.area();
        if old == 0 {
            return 0;
        }
        let (cx, cy) = (self.cx(), self.cy());
        let new_dx = (f64::from(self.ux - self.lx) * scale) as i32;
        let new_dy = (f64::from(self.uy - self.ly) * scale) as i32;
        self.lx = cx - new_dx / 2;
        self.ux = cx + new_dx / 2;
        self.ly = cy - new_dy / 2;
        self.uy = cy + new_dy / 2;
        self.area() - old
    }

    /// `snapOutward`: down to the site grid at the low side, up at the high side.
    fn snap_outward(&mut self, ox: i32, oy: i32, sx: i32, sy: i32) {
        let down = |v: i32, o: i32, s: i32| ((v - o) / s) * s + o;
        let up = |v: i32, o: i32, s: i32| ((v + s - 1 - o) / s) * s + o;
        self.lx = down(self.lx, ox, sx);
        self.ly = down(self.ly, oy, sy);
        self.ux = up(self.ux, ox, sx);
        self.uy = up(self.uy, oy, sy);
    }
}

/// `PlacerBaseCommon` once built.
#[derive(Debug, Clone)]
pub struct Common {
    pub site_x: i32,
    pub site_y: i32,
    pub insts: Vec<Instance>,
    /// Signal and clock nets: (design net index, pin count).
    pub nets: Vec<(usize, usize)>,
    pub pins: usize,
}

fn dbu_to_microns(d: &Design, v: i32) -> f64 {
    f64::from(v) / f64::from(d.dbu)
}

fn dbu_area_to_microns(d: &Design, a: i64) -> f64 {
    a as f64 / (f64::from(d.dbu) * f64::from(d.dbu))
}

/// `PlacerBaseCommon::init`.
pub fn placer_base_common(log: &mut Log, d: &Design, v: &Vars) -> Res<Common> {
    log.info("GPL", 1, "---- Initialize GPL Main Data Structures");
    log.info("GPL", 2, format!("DBU: {}", d.dbu));
    // The first row whose site is not a PAD site.
    let Some(&(_, _, site_x, site_y)) = d.rows.iter().find(|r| !r.1) else { return log.error("GPL", 305, "Unable to find a site") };
    if !d.die.contains(&d.core) {
        return log.error("GPL", 118, "core area outside of die.");
    }
    log.info("GPL", 3, format!("{:9} ( {:6.3} {:6.3} ) um", "SiteSize:", dbu_to_microns(d, site_x), dbu_to_microns(d, site_y)));
    log.info(
        "GPL",
        4,
        format!(
            "{:9} ( {:6.3} {:6.3} ) ( {:6.3} {:6.3} ) um",
            "CoreBBox:",
            dbu_to_microns(d, d.core.lx),
            dbu_to_microns(d, d.core.ly),
            dbu_to_microns(d, d.core.ux),
            dbu_to_microns(d, d.core.uy)
        ),
    );
    log.info("GPL", 16, format!("Core area:                  {:10.3} um^2", dbu_area_to_microns(d, d.core.area())));
    // Instances: the auto-placeable ones (DPL's rule), in the block's order.
    const ROW_LIMIT: i32 = 6;
    let mut insts = Vec::new();
    for (k, di) in d.insts.iter().enumerate() {
        if !di.core_auto_placeable {
            continue;
        }
        let mut macro_ = di.master_is_block;
        if !macro_ && di.bbox.dy() > ROW_LIMIT * site_y {
            macro_ = true;
            log.warn("GPL", 134, format!("Master {} is not marked as a BLOCK in LEF but is more than {ROW_LIMIT} rows tall.  It will be treated as a macro.", di.master));
        }
        // copyDbLocation: the box, widened by the padding when movable.
        let mut inst = Instance { db: Some(k), lx: di.bbox.lx, ly: di.bbox.ly, ux: di.bbox.lx + di.bbox.dx(), uy: di.bbox.ly + di.bbox.dy(), fixed: di.fixed, macro_ };
        if inst.is_place_instance() {
            inst.lx -= v.pad_left * site_x;
            inst.ux += v.pad_right * site_x;
        }
        if di.bbox.dy() > d.core.dy() {
            return log.error("GPL", 119, format!("instance {} height is larger than core.", di.name));
        }
        if di.bbox.dx() > d.core.dx() {
            return log.error("GPL", 120, format!("instance {} width is larger than core.", di.name));
        }
        // A partly covered site is unusable: fixed instances snap outward to whole sites.
        if inst.fixed {
            inst.snap_outward(d.core.lx, d.core.ly, site_x, site_y);
        }
        insts.push(inst);
    }
    let signal_pins = |i: &Instance| i.db.map_or(0, |k| d.insts[k].signal_pins);
    let (mut total_signal_pins, mut movable_area, mut total_area) = (0i64, 0i64, 0i64);
    for i in insts.iter().filter(|i| i.db.is_some()) {
        total_area += i.area();
        if !i.fixed {
            total_signal_pins += signal_pins(i) as i64;
            movable_area += i.area();
        }
    }
    log.info("GPL", 36, format!("Movable instances area:     {:10.3} um^2", dbu_area_to_microns(d, movable_area)));
    log.info("GPL", 37, format!("Total instances area:       {:10.3} um^2", dbu_area_to_microns(d, total_area)));
    let avg_density = if movable_area > 0 { total_signal_pins as f64 / movable_area as f64 } else { 0.0 };
    if log.debug_on("GPL", "extendPinDensity", 1) {
        return crate::log::refuse("the extendPinDensity debug report is not modelled");
    }
    let mut total_adjustment_area = 0i64;
    if !v.disable_pin_density_adjust {
        for i in insts.iter_mut() {
            if !i.fixed && i.db.is_some() {
                let pins = signal_pins(i);
                if pins > 2 && avg_density > 0.0 {
                    let target_area = pins as f64 / avg_density;
                    // Capped: excess would feed the routability inflation later.
                    let scale = (target_area / i.area() as f64).sqrt().clamp(0.95, 1.2);
                    total_adjustment_area += i.extend_size_by_scale(scale);
                }
            }
        }
    }
    log.info("GPL", 35, format!("Pin density area adjust:    {:10.3} um^2", dbu_area_to_microns(d, total_adjustment_area)));
    // Nets: SIGNAL and CLOCK only; each net's terminals, its ports unless -skip_io.
    let mut nets = Vec::new();
    let mut pins = 0usize;
    for (k, n) in d.nets.iter().enumerate() {
        if !n.signal_or_clock {
            continue;
        }
        let mut count = n.iterms.len();
        if !v.skip_io {
            for p in &n.bterms {
                if p.bbox.is_none() && !v.place_ios {
                    return log.error("GPL", 326, format!("{} toplevel port is not placed.", p.name));
                }
            }
            count += n.bterms.len();
        }
        pins += count;
        nets.push((k, count));
    }
    Ok(Common { site_x, site_y, insts, nets, pins })
}

/// `getMinMaxIdx`: the site indices a span covers, clamped to `[min, max]`.
fn min_max_idx(ll: i32, uu: i32, core_ll: i32, site: i32, min: i32, max: i32) -> (i32, i32) {
    let lower = (ll - core_ll) / site;
    let rem = if uu - core_ll >= site { (uu - core_ll) % site } else { uu - core_ll };
    let upper = if rem == 0 { (uu - core_ll) / site } else { (uu - core_ll) / site + 1 };
    (min.max(lower), max.min(upper))
}

/// One `PlacerBase` once built.
#[derive(Debug, Clone)]
pub struct Base {
    pub group: Option<usize>,
    pub region_bbox: Rect,
    pub region_area: i64,
    /// Indices into [`Common::insts`].
    pub place_insts: Vec<usize>,
    pub fixed_insts: Vec<usize>,
    pub dummies: Vec<Instance>,
    pub place_insts_area: i64,
    pub non_place_insts_area: i64,
    pub std_insts_area: i64,
    pub macro_insts_area: i64,
}

/// `PlacerBase::init(check_density)` for the top level (`group` none) or a group.
pub fn placer_base(log: &mut Log, d: &Design, c: &Common, group: Option<usize>, check_density: bool) -> Res<Base> {
    log.info("GPL", 32, format!("---- Initialize Region: {}", group.map_or("Top-level".to_string(), |g| d.groups[g].name.clone())));
    let core = d.core;
    let region = group.and_then(|g| d.groups[g].region);
    let (region_bbox, region_area) = match region.map(|r| &d.regions[r].boundaries) {
        Some(b) if !b.is_empty() => {
            let mut bb = b[0];
            for r in &b[1..] {
                bb = Rect { lx: bb.lx.min(r.lx), ly: bb.ly.min(r.ly), ux: bb.ux.max(r.ux), uy: bb.uy.max(r.uy) };
            }
            let bb = bb.intersect(&core);
            (bb, bb.area())
        }
        _ => (core, core.area()),
    };
    let mut b = Base { group, region_bbox, region_area, place_insts: vec![], fixed_insts: vec![], dummies: vec![], place_insts_area: 0, non_place_insts_area: 0, std_insts_area: 0, macro_insts_area: 0 };
    let in_this_group = |k: usize| -> bool {
        let g = d.insts[k].group;
        match group {
            None => g.is_none_or(|g| d.groups[g].visual_debug),
            Some(me) => g == Some(me) && !d.groups[me].visual_debug,
        }
    };
    for (idx, inst) in c.insts.iter().enumerate() {
        let Some(k) = inst.db else { continue };
        if inst.fixed {
            // A fixed instance outside the region is none of the placer's business.
            let overlap = if region.is_none_or(|r| d.regions[r].boundaries.is_empty()) {
                let r = Rect { lx: core.lx.max(inst.lx), ly: core.ly.max(inst.ly), ux: core.ux.min(inst.ux), uy: core.uy.min(inst.uy) };
                if r.lx < r.ux && r.ly < r.uy {
                    r.area()
                } else {
                    0
                }
            } else {
                return crate::log::refuse("a fixed instance against a region's boundaries (dbRegion::getOverlapArea) is not modelled");
            };
            if overlap == 0 {
                continue;
            }
            b.fixed_insts.push(idx);
            b.non_place_insts_area += overlap;
        } else {
            if !in_this_group(k) {
                continue;
            }
            b.place_insts.push(idx);
            b.place_insts_area += inst.area();
            if inst.dy() > c.site_y * 6 {
                b.macro_insts_area += inst.area();
            } else {
                b.std_insts_area += inst.area();
            }
        }
    }
    b.dummies = init_insts_for_unusable_sites(log, d, c, &b)?;
    for dm in &b.dummies {
        b.non_place_insts_area += dm.area();
    }
    print_info(log, d, c, &b, check_density)?;
    Ok(b)
}

#[derive(Clone, Copy, PartialEq)]
enum Site {
    Blocked,
    Row,
    FixedInst,
}

/// `initInstsForUnusableSites`: a site grid over the region — off-row, other regions' and
/// blockages' sites blocked, fixed instances' sites taken — then each maximal run of blocked
/// sites, extended down rows while the whole run stays blocked, becomes one dummy instance.
fn init_insts_for_unusable_sites(log: &mut Log, d: &Design, c: &Common, b: &Base) -> Res<Vec<Instance>> {
    let rb = b.region_bbox;
    let (sx, sy) = (c.site_x, c.site_y);
    let count_x = i64::from(rb.dx() / sx);
    let count_y = i64::from(rb.dy() / sy);
    let (cx, cy) = (count_x as i32, count_y as i32);
    let mut grid = vec![Site::Row; (count_x * count_y) as usize];
    let at = |i: i32, j: i32| (i64::from(j) * count_x + i64::from(i)) as usize;
    let mark = |grid: &mut Vec<Site>, r: &Rect, s: Site| {
        let (x0, x1) = min_max_idx(r.lx, r.ux, rb.lx, sx, 0, cx);
        let (y0, y1) = min_max_idx(r.ly, r.uy, rb.ly, sy, 0, cy);
        for i in x0..x1 {
            for j in y0..y1 {
                grid[at(i, j)] = s;
            }
        }
    };
    match b.group {
        None => {
            grid.fill(Site::Blocked);
            for (r, _, _, _) in &d.rows {
                mark(&mut grid, r, Site::Row);
            }
            for g in &d.groups {
                if let Some(r) = g.region {
                    for bd in &d.regions[r].boundaries {
                        mark(&mut grid, bd, Site::Blocked);
                    }
                }
            }
        }
        Some(g) => {
            let bds = &d.regions[d.groups[g].region.expect("a group's PlacerBase has a region")].boundaries;
            if bds.len() > 1 {
                grid.fill(Site::Blocked);
                for bd in bds {
                    mark(&mut grid, bd, Site::Row);
                }
            }
        }
    }
    for bl in &d.blockages {
        if let Some((inst, false)) = &bl.inst {
            return log.error("GPL", 3, format!("Blockages associated with moveable instances  are unsupported and ignored [inst: {inst}]\n"));
        }
        let (x0, x1) = min_max_idx(bl.bbox.lx, bl.bbox.ux, rb.lx, sx, 0, cx);
        let (y0, y1) = min_max_idx(bl.bbox.ly, bl.bbox.uy, rb.ly, sy, 0, cy);
        let filler_density = (100.0 - bl.max_density) / 100.0;
        let (mut cells, mut filled) = (0i32, 0i32);
        for j in y0..y1 {
            for i in x0..x1 {
                if cells == 0 || filled as f32 / cells as f32 <= filler_density {
                    grid[at(i, j)] = Site::Blocked;
                    log.debug("GPL", "dummies", 1, || format!("Blocking site at ({i}, {j}) due to blockage."));
                    filled += 1;
                }
                cells += 1;
            }
        }
    }
    for inst in c.insts.iter().filter(|i| i.fixed) {
        let Some(k) = inst.db else { continue };
        let di = &d.insts[k];
        let in_group = match b.group {
            None => di.group.is_none_or(|g| d.groups[g].visual_debug),
            Some(me) => di.group == Some(me) && !d.groups[me].visual_debug,
        };
        if !in_group {
            continue;
        }
        if inst.macro_ {
            if let Some(h) = di.halo {
                let r = Rect { lx: di.bbox.lx - h.lx, ly: di.bbox.ly - h.ly, ux: di.bbox.ux + h.ux, uy: di.bbox.uy + h.uy };
                let (x0, x1) = min_max_idx(r.lx, r.ux, rb.lx, sx, 0, cx);
                let (y0, y1) = min_max_idx(r.ly, r.uy, rb.ly, sy, 0, cy);
                for i in x0..x1 {
                    for j in y0..y1 {
                        grid[at(i, j)] = Site::Blocked;
                        log.debug("GPL", "dummies", 1, || format!("Blocking site at ({i}, {j}) due to fixed macro {} halo.", di.name));
                    }
                }
            }
        }
        let (x0, x1) = min_max_idx(inst.lx, inst.ux, rb.lx, sx, 0, cx);
        let (y0, y1) = min_max_idx(inst.ly, inst.uy, rb.ly, sy, 0, cy);
        for i in x0..x1 {
            for j in y0..y1 {
                grid[at(i, j)] = Site::FixedInst;
                log.debug("GPL", "dummies", 1, || format!("Blocking site at ({i}, {j}) due to fixed instance {}.", di.name));
            }
        }
    }
    let mut dummies = Vec::new();
    for j in 0..cy {
        let mut i = 0;
        while i < cx {
            if grid[at(i, j)] == Site::Blocked {
                let start = i;
                while i < cx && grid[at(i, j)] == Site::Blocked {
                    i += 1;
                }
                let end_x = i;
                let mut end_y = j + 1;
                while end_y < cy && (start..end_x).all(|x| grid[at(x, end_y)] == Site::Blocked) {
                    end_y += 1;
                }
                for y in j..end_y {
                    for x in start..end_x {
                        grid[at(x, y)] = Site::FixedInst;
                    }
                }
                dummies.push(Instance { db: None, lx: rb.lx + sx * start, ly: rb.ly + sy * j, ux: rb.lx + sx * end_x, uy: rb.ly + sy * end_y, fixed: true, macro_: false });
            }
            // The reference's `for` steps past the cell that ended the run.
            i += 1;
        }
    }
    Ok(dummies)
}

/// `PlacerBase::printInfo`.
fn print_info(log: &mut Log, d: &Design, c: &Common, b: &Base, check_density: bool) -> Res<()> {
    log.info("GPL", 6, format!("Number of instances:        {:10}", b.place_insts.len() + b.fixed_insts.len() + b.dummies.len()));
    log.info("GPL", 7, format!("Movable instances:          {:10}", b.place_insts.len()));
    log.info("GPL", 8, format!("Fixed instances:            {:10}", b.fixed_insts.len()));
    log.info("GPL", 9, format!("Dummy instances:            {:10}", b.dummies.len()));
    log.info("GPL", 10, format!("Number of nets:             {:10}", c.nets.len()));
    log.info("GPL", 11, format!("Number of pins:             {:10}", c.pins));
    let m = |v: i32| dbu_to_microns(d, v);
    log.info("GPL", 12, format!("{:15} ( {:6.3} {:6.3} ) ( {:6.3} {:6.3} ) um", "Die BBox:", m(d.die.lx), m(d.die.ly), m(d.die.ux), m(d.die.uy)));
    let r = b.region_bbox;
    log.info("GPL", 13, format!("{:15} ( {:6.3} {:6.3} ) ( {:6.3} {:6.3} ) um", "Placement BBox:", m(r.lx), m(r.ly), m(r.ux), m(r.uy)));
    // A float quotient over an int64 difference, then × 100 in float.
    let util = b.place_insts_area as f32 / (b.region_area - b.non_place_insts_area) as f32 * 100.0;
    log.info("GPL", 14, format!("Region name: {}.", b.group.map_or("top-level".to_string(), |g| d.groups[g].name.clone())));
    log.info("GPL", 15, format!("Region area:                {:10.3} um^2", dbu_area_to_microns(d, b.region_area)));
    log.info("GPL", 17, format!("Fixed instances area:       {:10.3} um^2", dbu_area_to_microns(d, b.non_place_insts_area)));
    log.info("GPL", 18, format!("Movable instances area:     {:10.3} um^2", dbu_area_to_microns(d, b.place_insts_area)));
    log.info("GPL", 19, format!("{:27} {:10.3} %", "Utilization:", util));
    log.info("GPL", 20, format!("Standard cells area:        {:10.3} um^2", dbu_area_to_microns(d, b.std_insts_area)));
    log.info("GPL", 21, format!("Large instances area:       {:10.3} um^2", dbu_area_to_microns(d, b.macro_insts_area)));
    if check_density && f64::from(util) >= 100.1 {
        return log.error("GPL", 301, format!("Utilization {util:.3} % exceeds 100%."));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Rule (getMinMaxIdx): the lower index truncates, the upper rounds up unless the span ends on
    // a site boundary; both clamped.
    #[test]
    fn site_spans_round_outward() {
        assert_eq!(min_max_idx(0, 380, 0, 190, 0, 10), (0, 2));
        assert_eq!(min_max_idx(10, 381, 0, 190, 0, 10), (0, 3));
        assert_eq!(min_max_idx(-500, 5000, 0, 190, 0, 10), (0, 10));
    }

    // Rule (snapOutward): fixed instances snap down at the low side and up at the high side.
    #[test]
    fn fixed_instances_snap_outward() {
        let mut i = Instance { db: Some(0), lx: 195, ly: 10, ux: 385, uy: 1410, fixed: true, macro_: false };
        i.snap_outward(0, 0, 190, 1400);
        assert_eq!((i.lx, i.ly, i.ux, i.uy), (190, 0, 570, 2800));
    }

    // Rule (extendSizeByScale): about the integer centre, each side × scale truncated, half each
    // way with integer division.
    #[test]
    fn extension_is_about_the_integer_centre() {
        let mut i = Instance { db: Some(0), lx: 0, ly: 0, ux: 381, uy: 1400, fixed: false, macro_: false };
        let delta = i.extend_size_by_scale(1.1);
        assert_eq!((i.lx, i.ux, i.ly, i.uy), (190 - 209, 190 + 209, 700 - 770, 700 + 770));
        assert_eq!(delta, 418 * 1540 - 381 * 1400);
    }
}
