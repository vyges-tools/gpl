// SPDX-License-Identifier: Apache-2.0
//! `vyges-gpl <job.json>`: replays a job's steps — `read_db` (the design as the reference held it
//! when `global_placement` ran), `set_debug_level`, then `global_placement` — and prints the run's
//! output lines as JSON: `{"lines": [...], "stop": null | "error GPL-0130" | "refused: …",
//! "outcome": "initially_placed"}`, with `trace`: the raw-value probe lines.

use serde_json::{json, Value};
use std::collections::HashMap;
use vyges_gpl::log::{refuse, Log, Res, Stop};
use vyges_gpl::placer_base::{Blockage, DbInst, DbNet, Design, Group, IoPort, Port, Rect, Region, Term};
use vyges_gpl::run::Outcome;
use vyges_opendb::Db;

fn some(s: String) -> Option<String> {
    (!s.is_empty()).then_some(s)
}

fn e(err: vyges_opendb::Error) -> Stop {
    Stop::Refused(format!("database: {err}"))
}

/// `updateDbGCells` then `reportHpwlMetric`: each instance written where the placer has it, then
/// GPL-1018.
fn update_db_and_report_hpwl(db: &mut Db, log: &mut Log, locations: &[(usize, i32, i32)], io: &[vyges_gpl::run::IoWrite]) -> Result<(), String> {
    use vyges_gpl::route_base::Router;
    let stop = |s: Stop| match s {
        Stop::Refused(m) => m,
        Stop::Error { tool, id } => format!("{tool}-{id:04}"),
    };
    vyges_gpl::db_router::DbRouter::new(db).write_locations(log, locations).map_err(stop)?;
    // `updateDbIoPins`: every pin of the port destroyed, one PLACED pin at the solved centre
    for (port, layer, rect) in io {
        db.bterm_clear_unfixed_bpins(port).map_err(|e| format!("{port}: {e}"))?;
        if db.num_bterm_get_b_pins(port) != 0 {
            return Err(format!("{port}: a FIXED pin on a port concurrent IO placement moves is not modelled"));
        }
        let k = db.create_bterm_pin(port, layer, *rect).map_err(|e| format!("{port}: {e}"))?;
        db.bpin_set_placement_status(port, k, "PLACED").map_err(|e| format!("{port}: {e}"))?;
    }
    vyges_gpl::db_router::DbRouter::new(db).report_hpwl(log).map_err(stop)
}

/// The design as `PlacerBaseCommon` / `PlacerBase` read it, in the block's own orders.
fn design(db: &Db) -> Res<Design> {
    let mut d = Design { dbu: db.tech_get_db_units_per_micron(), ..Design::default() };
    d.die = Rect { lx: db.block_get_die_area_x_min(), ly: db.block_get_die_area_y_min(), ux: db.block_get_die_area_x_max(), uy: db.block_get_die_area_y_max() };
    d.core = Rect { lx: db.block_get_core_area_x_min(), ly: db.block_get_core_area_y_min(), ux: db.block_get_core_area_x_max(), uy: db.block_get_core_area_y_max() };
    let rows = db.block_get_rows();
    let mut seen = std::collections::HashSet::new();
    for r in &rows {
        if !seen.insert(r) {
            return refuse(format!("row name {r} is not unique: rows are read by name"));
        }
        let site = db.row_get_site(r);
        let bbox = Rect { lx: db.row_get_b_box_x_min(r), ly: db.row_get_b_box_y_min(r), ux: db.row_get_b_box_x_max(r), uy: db.row_get_b_box_y_max(r) };
        let pad = db.site_get_class(&site).map_err(e)? == "PAD";
        d.rows.push((bbox, pad, db.site_get_width(&site), db.site_get_height(&site)));
    }
    let group_names = db.block_get_groups();
    let group_ix: HashMap<String, usize> = group_names.iter().enumerate().map(|(i, g)| (g.clone(), i)).collect();
    let region_names = db.block_get_regions();
    let region_ix: HashMap<String, usize> = region_names.iter().enumerate().map(|(i, r)| (r.clone(), i)).collect();
    for g in &group_names {
        let region = some(db.group_get_region(g)).map(|r| region_ix.get(&r).copied().ok_or_else(|| Stop::Refused(format!("group {g}: region {r} not in the block"))));
        d.groups.push(Group { name: db.group_get_name(g), visual_debug: db.group_get_type(g).map_err(e)? == "VISUAL_DEBUG", region: region.transpose()? });
    }
    for r in &region_names {
        let boundaries = db.region_boundaries(r).map_err(e)?.into_iter().map(|(lx, ly, ux, uy)| Rect { lx, ly, ux, uy }).collect();
        let groups = db.region_get_groups(r).iter().map(|g| group_ix.get(g).copied().ok_or_else(|| Stop::Refused(format!("region {r}: group {g} not in the block")))).collect::<Res<Vec<_>>>()?;
        d.regions.push(Region { boundaries, groups });
    }
    for name in db.block_get_insts() {
        let master = db.inst_get_master(&name);
        let b = db.inst_bbox(&name).map_err(e)?;
        let [lx, ly, ux, uy] = b[..] else { return refuse(format!("{name}: no bounding box")) };
        let halo = match db.inst_halo(&name).map_err(e)? {
            None => None,
            Some((l, b, r, t, _)) => {
                if db.inst_get_orient(&name) != "R0" {
                    return refuse(format!("{name}: a halo on an oriented instance (getTransformedHalo) is not modelled"));
                }
                Some(Rect { lx: l as i32, ly: b as i32, ux: r as i32, uy: t as i32 })
            }
        };
        let signal_pins = db.inst_get_i_terms(&name).iter().filter(|t| {
            let (i, p) = t.rsplit_once('/').unwrap_or(("", t));
            !matches!(db.iterm_get_sig_type(i, p).as_str(), "POWER" | "GROUND")
        }).count();
        let group = some(db.inst_get_group(&name)).map(|g| group_ix.get(&g).copied().ok_or_else(|| Stop::Refused(format!("{name}: group {g} not in the block")))).transpose()?;
        d.insts.push(DbInst {
            orient: db.inst_get_orient(&name),
            master_w: db.master_get_width(&master) as i32,
            master_h: db.master_get_height(&master) as i32,
            placed: db.inst_is_placed(&name),
            core_auto_placeable: db.master_is_core_auto_placeable(&master),
            master_is_block: db.master_is_block(&master),
            bbox: Rect { lx, ly, ux, uy },
            fixed: db.inst_is_fixed(&name),
            group,
            halo,
            signal_pins,
            master,
            name,
        });
    }
    let inst_ix: HashMap<String, usize> = d.insts.iter().enumerate().map(|(k, i)| (i.name.clone(), k)).collect();
    // Each master terminal's pin box: the union of its pins' boxes.
    let mut shapes: HashMap<(String, String), Option<Rect>> = HashMap::new();
    let mut mterm_order: HashMap<(String, String), usize> = HashMap::new();
    for name in db.block_get_nets() {
        let sig = db.net_get_sig_type(&name);
        let mut iterms = Vec::new();
        for t in db.net_get_i_terms(&name) {
            let (i, p) = t.rsplit_once('/').unwrap_or(("", &t));
            let k = *inst_ix.get(i).ok_or_else(|| Stop::Refused(format!("{t}: instance not in the block")))?;
            let master = d.insts[k].master.clone();
            let shape = *shapes.entry((master.clone(), p.to_string())).or_insert_with(|| {
                let mut u: Option<Rect> = None;
                for m in 0..db.num_mterm_get_m_pins(&master, p) {
                    let r = Rect { lx: db.mpin_get_b_box_x_min(&master, p, m), ly: db.mpin_get_b_box_y_min(&master, p, m), ux: db.mpin_get_b_box_x_max(&master, p, m), uy: db.mpin_get_b_box_y_max(&master, p, m) };
                    if r.lx > r.ux || r.ly > r.uy {
                        continue;
                    }
                    u = Some(match u {
                        None => r,
                        Some(a) => Rect { lx: a.lx.min(r.lx), ly: a.ly.min(r.ly), ux: a.ux.max(r.ux), uy: a.uy.max(r.uy) },
                    });
                }
                u
            });
            if !mterm_order.contains_key(&(master.clone(), p.to_string())) {
                for (n, (m, _)) in db.master_mterms(&master).map_err(e)?.into_iter().enumerate() {
                    mterm_order.insert((master.clone(), m), n);
                }
            }
            let order = mterm_order.get(&(master.clone(), p.to_string())).copied().unwrap_or(usize::MAX);
            iterms.push(Term { inst: k, shape, order });
        }
        let bterms = db
            .net_get_b_terms(&name)
            .into_iter()
            .map(|b| {
                let r = Rect { lx: db.bterm_get_b_box_x_min(&b), ly: db.bterm_get_b_box_y_min(&b), ux: db.bterm_get_b_box_x_max(&b), uy: db.bterm_get_b_box_y_max(&b) };
                let fixed = matches!(db.bterm_get_first_pin_placement_status(&b).as_str(), "LOCKED" | "FIRM" | "COVER");
                Port { bbox: (r.lx <= r.ux && r.ly <= r.uy).then_some(r), name: b, fixed }
            })
            .collect();
        d.nets.push(DbNet { signal_or_clock: sig == "SIGNAL" || sig == "CLOCK", iterms, bterms, name });
    }
    let boxes = db.blockage_boxes().map_err(e)?;
    for (k, (lx, ly, ux, uy)) in boxes.into_iter().enumerate() {
        let inst = some(db.blockage_get_instance(k)).map(|i| {
            let fixed = db.inst_is_fixed(&i);
            (i, fixed)
        });
        d.blockages.push(Blockage { bbox: Rect { lx, ly, ux, uy }, inst, max_density: db.blockage_get_max_density(k) });
    }
    read_io(db, &mut d)?;
    Ok(d)
}

/// What `-place_ios` reads: every block terminal in block order, the pin blockages, the
/// top-layer pin grid and the first horizontal / vertical routing layers.
fn read_io(db: &Db, d: &mut Design) -> Res<()> {
    let net_ix: HashMap<String, usize> = d.nets.iter().enumerate().map(|(i, n)| (n.name.clone(), i)).collect();
    for b in db.bterm_names() {
        let r = Rect { lx: db.bterm_get_b_box_x_min(&b), ly: db.bterm_get_b_box_y_min(&b), ux: db.bterm_get_b_box_x_max(&b), uy: db.bterm_get_b_box_y_max(&b) };
        let status = db.bterm_get_first_pin_placement_status(&b);
        let fixed = matches!(status.as_str(), "LOCKED" | "FIRM" | "COVER");
        let placed = fixed || status == "PLACED";
        let net = net_ix.get(&db.bterm_net(&b)).copied();
        let mirror = db.bterm_has_mirrored_b_term(&b).then(|| db.bterm_get_mirrored_b_term(&b));
        let constraint = db.bterm_constraint_region(&b).map_err(e)?;
        d.ports.push(IoPort { name: b, net, bbox: (r.lx <= r.ux && r.ly <= r.uy).then_some(r), fixed, placed, constraint, mirror });
    }
    d.io.blocked = db.blocked_regions_for_pins().map_err(e)?;
    d.io.top_grid = db.bterm_top_layer_grid().map_err(e)?.map(|g| {
        let w = db.layer_get_width(&g.layer) as i32;
        (g.layer, g.pin_width, g.pin_height, w)
    });
    for (layer, dir) in db.layers_with_direction().map_err(e)? {
        if db.layer_get_routing_level(&layer) == 0 {
            continue;
        }
        let w = db.layer_get_width(&layer) as i32;
        if dir == "HORIZONTAL" && d.io.hor.is_none() {
            d.io.hor = Some((layer, w));
        } else if dir == "VERTICAL" && d.io.ver.is_none() {
            d.io.ver = Some((layer, w));
        }
    }
    d.io.die_points = db.die_area_polygon().map_err(e)?.len();
    Ok(())
}

/// `utl::physicalCoreCount`: distinct (package, core) pairs over the CPUs this process may run on
/// (`Cpus_allowed_list`), from sysfs; without topology, the CPU count.
fn physical_core_count() -> usize {
    let allowed = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find_map(|l| l.strip_prefix("Cpus_allowed_list:").map(|v| v.trim().to_string())))
        .unwrap_or_default();
    let mut cpus = Vec::new();
    for part in allowed.split(',').filter(|p| !p.is_empty()) {
        let (a, b) = part.split_once('-').unwrap_or((part, part));
        if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
            cpus.extend(a..=b);
        }
    }
    let read = |cpu: usize, f: &str| std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/topology/{f}")).ok().and_then(|s| s.trim().parse::<i64>().ok());
    let cores: std::collections::BTreeSet<(i64, i64)> = cpus.iter().filter_map(|&c| Some((read(c, "physical_package_id")?, read(c, "core_id")?))).collect();
    if !cores.is_empty() {
        cores.len()
    } else if !cpus.is_empty() {
        cpus.len()
    } else {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    }
}

fn run(job: &Value) -> Result<Value, String> {
    let mut db: Option<Db> = None;
    let mut threads = vyges_gpl::run::Threads { asked: 1, cores: physical_core_count() };
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut log = Log::default();
    let finish = |log: Log, stop: Option<String>, extra: Value| {
        let (status, reason) = match stop {
            None => ("placed".to_string(), None),
            Some(s) => match s.split_once(": ") {
                Some((st, r)) => (st.to_string(), Some(r.to_string())),
                None => (s, None),
            },
        };
        let mut v = json!({"tool": "vyges-gpl", "status": status, "reason": reason, "lines": log.lines, "trace": log.trace});
        if let (Some(o), Value::Object(x)) = (v.as_object_mut(), extra) {
            o.extend(x);
        }
        v
    };
    for step in job["steps"].as_array().ok_or("steps")? {
        let cmd = step["cmd"].as_str().ok_or("cmd")?;
        let args: Vec<String> = step["args"].as_array().map(|a| a.iter().filter_map(Value::as_str).map(String::from).collect()).unwrap_or_default();
        let r: Res<Option<Outcome>> = (|| {
            match cmd {
                "read_db" => {
                    let path = args.last().ok_or_else(|| Stop::Refused("read_db path".into()))?;
                    db = Some(Db::open(path).map_err(|e| Stop::Refused(format!("{path}: {e}")))?);
                }
                "set_debug_level" => {
                    let [tool, group, level] = args.as_slice() else { return refuse("set_debug_level: needs a tool, a group and a level") };
                    let level: i32 = level.parse().map_err(|_| Stop::Refused(format!("set_debug_level {level}: not an integer")))?;
                    log.debug.insert((tool.clone(), group.clone()), level);
                }
                // `placement_cluster`: its names (one list, or several), each found as `get_cells`
                // finds an exact name (GPL-0095 when none is), GPL-0096; kept for the placer.
                "placement_cluster" => {
                    let db = db.as_ref().ok_or_else(|| Stop::Refused("placement_cluster before read_db".into()))?;
                    let names: Vec<String> = args.iter().flat_map(|a| a.split_whitespace()).map(|n| n.trim_matches(|c| c == '{' || c == '}').to_string()).filter(|n| !n.is_empty()).collect();
                    if names.is_empty() {
                        return log.error("GPL", 94, "placement_cluster requires a list of instances.");
                    }
                    if names.iter().any(|n| n.contains(['*', '?'])) {
                        return refuse("placement_cluster with a wildcard pattern is not modelled");
                    }
                    let all = db.block_get_insts();
                    let ix: std::collections::HashMap<&str, usize> = all.iter().enumerate().map(|(i, n)| (n.as_str(), i)).collect();
                    let mut members = Vec::new();
                    for n in &names {
                        match ix.get(n.as_str()) {
                            Some(&k) => members.push(k),
                            None => return log.error("GPL", 95, format!("Instances {{{n}}} for placement_cluster command were not found.")),
                        }
                    }
                    log.info("GPL", 96, format!("Created placement cluster of {} instances.", members.len()));
                    clusters.push(members);
                }
                "set_thread_count" => {
                    let [n] = args.as_slice() else { return refuse("set_thread_count: needs one count") };
                    threads.asked = n.parse().map_err(|_| Stop::Refused(format!("set_thread_count {n}: not a count")))?;
                }
                "global_placement" => {
                    log.lines.clear();
                    let dbr = db.as_ref().ok_or_else(|| Stop::Refused("global_placement before read_db".into()))?;
                    let mut d = design(dbr)?;
                    d.clusters = clusters.clone();
                    // The database the run writes mid-command: incremental placement's first pass
                    // (and its HPWL), routability's congestion passes.
                    let mut router = vyges_gpl::db_router::DbRouter::new(db.as_mut().expect("read before global_placement"));
                    return vyges_gpl::run::global_placement(&mut log, &d, &args, threads, Some(&mut router)).map(Some);
                }
                other => return refuse(format!("{other}: not a modelled step")),
            }
            Ok(None)
        })();
        match r {
            Ok(None) => {}
            Ok(Some(Outcome::InitiallyPlaced(c, bases, ip, skip_nesterov))) => {
                let extra = json!({"insts": c.insts.len(), "nets": c.nets.len(), "pins": c.pins,
                    "bases": bases.iter().map(|b| json!({"group": b.group, "place": b.place_insts.len(), "fixed": b.fixed_insts.len(), "dummies": b.dummies.len()})).collect::<Vec<_>>()});
                if !skip_nesterov {
                    // What is modelled ends here; Nesterov placement is not, so the command is
                    // refused rather than reported as done.
                    let mut v = finish(log, Some("refused: Nesterov placement is not modelled yet".into()), extra);
                    v["modelled_through"] = json!("initial_place");
                    return Ok(v);
                }
                // -skip_nesterov_place: initial placement is the whole command. Each instance it
                // moved goes to its database box's lower-left, PLACED (dbSetCenterLocation,
                // dbSetPlaced).
                let db = db.as_mut().expect("read before global_placement");
                let names = db.block_get_insts();
                for &k in &ip.place {
                    let di = c.insts[k].db.expect("a placed instance is a design instance");
                    let b = ip.db_bbox[di];
                    let e = |e: vyges_opendb::Error| format!("{}: {e}", names[di]);
                    db.set_inst_location(&names[di], b.lx, b.ly).map_err(e)?;
                    db.inst_set_placement_status(&names[di], "PLACED").map_err(e)?;
                }
                if let Some(path) = job["write_def"].as_str() {
                    db.write_def(path).map_err(|e| format!("{path}: {e}"))?;
                }
                return Ok(finish(log, None, extra));
            }
            Ok(Some(Outcome::Placed(c, bases, locations, iter, io))) => {
                let extra = json!({"insts": c.insts.len(), "nets": c.nets.len(), "pins": c.pins, "nesterov_iterations": iter,
                    "bases": bases.iter().map(|b| json!({"group": b.group, "place": b.place_insts.len(), "fixed": b.fixed_insts.len(), "dummies": b.dummies.len()})).collect::<Vec<_>>()});
                // `updateDbGCells`: each placed instance at its location, PLACED.
                let db = db.as_mut().expect("read before global_placement");
                update_db_and_report_hpwl(db, &mut log, &locations, &io)?;
                if let Some(path) = job["write_def"].as_str() {
                    db.write_def(path).map_err(|e| format!("{path}: {e}"))?;
                }
                return Ok(finish(log, None, extra));
            }
            Ok(Some(Outcome::Unchanged(c, bases))) => {
                // Incremental placement with every instance placed: nothing moved; the design as read.
                let extra = json!({"insts": c.insts.len(), "nets": c.nets.len(), "pins": c.pins, "unchanged": true,
                    "bases": bases.iter().map(|b| json!({"group": b.group, "place": b.place_insts.len(), "fixed": b.fixed_insts.len(), "dummies": b.dummies.len()})).collect::<Vec<_>>()});
                if let (Some(path), Some(db)) = (job["write_def"].as_str(), db.as_mut()) {
                    db.write_def(path).map_err(|e| format!("{path}: {e}"))?;
                }
                return Ok(finish(log, None, extra));
            }
            Err(Stop::Error { tool, id }) => {
                let mut v = finish(log, Some(format!("error: {tool}-{id:04}")), json!({}));
                v["code"] = json!(format!("{tool}-{id:04}"));
                return Ok(v);
            }
            Err(Stop::Refused(why)) => return Ok(finish(log, Some(format!("refused: {why}")), json!({}))),
        }
    }
    // No global_placement step: nothing was placed or checked.
    Ok(finish(log, Some("vacuous: the job has no global_placement step".into()), json!({})))
}

const USAGE: &str = "\
vyges physical gpl — global placement: the placer's view of the design (instances, nets, pins,
regions), the initial placement, then the Nesterov electrostatic placement

USAGE:
  vyges physical gpl global_placement <job.json> [-o FILE]
  vyges physical gpl --describe
  vyges physical gpl --help
  vyges physical gpl --version

JOB FIELDS:
  steps        required — the commands in order, each {\"cmd\": ..., \"args\": [...]}:
               read_db, set_debug_level, set_thread_count, then global_placement [options]
  write_def    write the placed design, as DEF, to this path

MODELLED:
  the placer's construction (GPL-0001..0021, 0035..0037), the initial placement (every
  bound-to-bound round and its [InitialPlace] line), and the Nesterov placement: the fillers,
  the pre-placement perturbation, the bin grid (GPL-0023..0030), the DCT density field, the
  weighted-average wirelength gradient, the accelerated descent with back-tracking, every
  progress row, convergence (GPL-1001..1009) and the final area and HPWL (GPL-1014, 1018) —
  all in the reference's own operation order; region groups each their own Nesterov region
  (GPL-1016). Refused: -timing_driven, -routability_driven, -virtual_cts,
  -enable_routing_congestion, -incremental, -place_ios, placement clusters.

OPTIONS:
  -o FILE               write the JSON report to FILE instead of stdout
  --json                accepted; the report is JSON either way
  --describe            print a machine-readable JSON description of the command
  --bug-report          file a bug (central: vyges/community)
  --feature-request     request a feature (central)
  --sponsor             sponsor Vyges (github.com/sponsors/vyges-ip)
  --star                star this tool on GitHub

THREADS:
  Runs on one thread; there is no thread option. A job's set_thread_count is read as the
  reference reads it, but only decides the thread count the run reports.

EXIT STATUS:
  0  placed       -skip_nesterov_place: the design initially placed
  2  vacuous      the job has no global_placement step. NOT a pass.
  2  error        usage, unreadable input, or an error the command raises (its line is reported)
  3  refused      an input this engine does not model — or a Nesterov placement to run
";

const PIN_TOKEN: &str = "@OPENROAD_PIN@";

fn describe() -> String {
    DESCRIBE.replace(PIN_TOKEN, vyges_opendb::OPENROAD_PIN)
}

const DESCRIBE: &str = r#"{
  "schema": "vyges-tool-descriptor/1.1",
  "openroad_pin": "@OPENROAD_PIN@",
  "name": "gpl",
  "summary": "global placement: the placer's construction, the bound-to-bound initial placement and the Nesterov (electrostatic density) placement",
  "maturity": "structured",
  "provenance_limitations": [
    "input_hash covers the argument vector, not the content of the job file or of the design files it names.",
    "status is one of placed, vacuous, refused or error. placed means the command ran to the end: the initial placement (and, unless -skip_nesterov_place, the Nesterov placement to the target overflow) written to the database. The declared assertion passes on placed only. Exit status is 0 for placed, 2 for vacuous and for error, 3 for refused.",
    "Modelled: the placer's construction, the initial placement and the Nesterov placement (fillers, the pre-placement perturbation, the bin grid, the DCT density field, the weighted-average wirelength gradient, the accelerated descent with back-tracking, the divergence revert) over the top level and every region group, incremental placement, placement clusters routability-driven placement (RUDY or the global router: cell inflation, filler removal, net reweighting, the revert to the best pass) and concurrent IO placement (-place_ios), each bit-exact against the reference: every progress row and every iteration's state. Refused rather than guessed: -timing_driven, -virtual_cts, -enable_routing_congestion."
  ],
  "invocation": {
    "args_template": ["global_placement", "{job}"],
    "optional": [ { "arg": "out", "flag": "-o" } ],
    "emits_json": true
  },
  "inputs": {
    "type": "object",
    "required": ["job"],
    "properties": {
      "job": { "type": "string", "description": "path to a JSON job: {steps: [{cmd, args}], write_def}" },
      "out": { "type": "string", "description": "write the JSON report to FILE instead of stdout" }
    }
  },
  "consumes": ["job"],
  "threads": {
    "parallel": false,
    "guidance": "Runs on one thread; there is no thread option. A job's set_thread_count step is accepted as the reference accepts it, but only decides the thread count the run reports, not the placement. Several runs can go side by side, one core each."
  },
  "artifacts": [],
  "assertion": {
    "id": "design-placed",
    "field": "status",
    "pass_when": { "in": ["placed"] }
  }
}"#;

fn link(flag: &str) -> Option<(&'static str, &'static str)> {
    Some(match flag {
        "--bug-report" => ("Report a bug", "https://github.com/vyges/community/issues/new?template=bug_report_template.yaml"),
        "--feature-request" => ("Request a feature", "https://github.com/vyges/community/issues/new?labels=enhancement"),
        "--sponsor" => ("Sponsor Vyges", "https://github.com/sponsors/vyges-ip"),
        "--star" => ("Star this tool", "https://github.com/vyges-tools/gpl"),
        _ => return None,
    })
}

fn exit_for(status: &str) -> u8 {
    match status {
        "placed" => 0,
        "refused" => 3,
        _ => 2,
    }
}

/// The run's lines as structured events (`vyges-events/1.0`): each `[INFO|WARNING|ERROR TOOL-nnnn]
/// text` line the reference's logger would print under its own code and severity, any other line
/// (`report` text) as information; a refusal under `GPL-REFUSED`. The JSON report is unchanged.
fn emit_events(report: &serde_json::Value) {
    use vyges_events::{emit, Event, Severity};
    for l in report["lines"].as_array().into_iter().flatten().filter_map(serde_json::Value::as_str) {
        let parsed = l.strip_prefix('[').and_then(|r| r.split_once("] ")).and_then(|(head, text)| head.split_once(' ').map(|(sev, code)| (sev, code, text)));
        let ev = match parsed {
            Some((sev, code, text)) => {
                let severity = match sev {
                    "ERROR" | "CRITICAL" => Severity::Error,
                    "WARNING" => Severity::Warn,
                    _ => Severity::Info,
                };
                Event::new("vyges-gpl", severity, text.to_string()).with_code(code)
            }
            None => Event::new("vyges-gpl", Severity::Info, l.to_string()),
        };
        emit(&ev);
    }
    if report["status"] == "refused" {
        if let Some(why) = report["reason"].as_str() {
            emit(&Event::new("vyges-gpl", Severity::Warn, why.to_string()).with_code("GPL-REFUSED"));
        }
    }
}

fn main() -> std::process::ExitCode {
    use std::process::ExitCode;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut out: Option<String> = None;
    let mut positional: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => match args.get(i + 1) {
                Some(v) => {
                    out = Some(v.clone());
                    i += 1;
                }
                None => {
                    eprintln!("vyges-gpl: -o needs a FILE");
                    return ExitCode::from(2);
                }
            },
            "--json" => {}
            a if a.starts_with('-') && !positional.is_empty() => {
                eprintln!("vyges-gpl: unknown option {a}\n\n{USAGE}");
                return ExitCode::from(2);
            }
            a => positional.push(a),
        }
        i += 1;
    }
    match positional.first().copied() {
        None | Some("-h") | Some("--help") => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Some("-V") | Some("--version") => {
            println!("vyges-gpl {} ({})\nCopyright (c) Vyges. Apache-2.0.", env!("CARGO_PKG_VERSION"), env!("VYGES_GIT_SHA"));
            return ExitCode::SUCCESS;
        }
        Some("--describe") => {
            println!("{}", describe());
            return ExitCode::SUCCESS;
        }
        Some(f) if link(f).is_some() => {
            let (label, url) = link(f).expect("a link flag");
            println!("{label}:\n  {url}");
            return ExitCode::SUCCESS;
        }
        Some("global_placement") if positional.len() == 2 => {}
        _ => {
            eprint!("{USAGE}");
            return ExitCode::from(2);
        }
    }
    let path = positional[1];
    let report = match std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}")).and_then(|t| serde_json::from_str::<Value>(&t).map_err(|e| format!("{path}: {e}"))) {
        Err(e) => json!({ "tool": "vyges-gpl", "status": "error", "reason": e }),
        Ok(job) => match run(&job) {
            Ok(r) => r,
            Err(e) => json!({ "tool": "vyges-gpl", "status": "error", "reason": e }),
        },
    };
    let status = report["status"].as_str().unwrap_or("error").to_string();
    emit_events(&report);
    let text = format!("{report}\n");
    match &out {
        Some(f) => {
            if let Err(e) = std::fs::write(f, &text) {
                eprintln!("vyges-gpl: {f}: {e}");
                return ExitCode::from(2);
            }
        }
        None => print!("{text}"),
    }
    ExitCode::from(exit_for(&status))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_descriptor_is_valid_json_and_names_the_engine() {
        let d: Value = serde_json::from_str(&describe()).expect("--describe must be valid JSON");
        assert_eq!(d["name"], "gpl");
        assert_eq!(d["openroad_pin"].as_str().map(str::len), Some(40));
    }

    // ⛔ `maturity` has exactly THREE legal values — discovered, structured, workflow-validated —
    // and the parser maps anything else to `discovered`, where the verdict is suppressed to
    // `unknown`. This engine said `experimental`, which threw every verdict away. `structured`:
    // a versioned operation, a normalized result and a declared assertion; the correlation
    // harness lives outside the repository, so not `workflow-validated`.
    #[test]
    fn the_maturity_is_a_legal_rung() {
        let d: Value = serde_json::from_str(&describe()).expect("--describe must be valid JSON");
        let m = d["maturity"].as_str().expect("maturity is a string");
        assert!(["discovered", "structured", "workflow-validated"].contains(&m), "{m}");
        assert_eq!(m, "structured");
    }

    // Rule (vacuous convention): only a placed design exits 0; a refusal never does.
    #[test]
    fn only_placed_exits_zero() {
        assert_eq!(exit_for("placed"), 0);
        assert_eq!(exit_for("refused"), 3);
        assert_eq!(exit_for("vacuous"), 2);
        assert_eq!(exit_for("error"), 2);
    }
}
