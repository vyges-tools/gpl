// SPDX-License-Identifier: Apache-2.0
//! `vyges-gpl <job.json>`: replays a job's steps — `read_db` (the design as the reference held it
//! when `global_placement` ran), `set_debug_level`, then `global_placement` — and prints the run's
//! output lines as JSON: `{"lines": [...], "stop": null | "error GPL-0130" | "refused: …",
//! "outcome": "constructed"}`.

use serde_json::{json, Value};
use std::collections::HashMap;
use vyges_gpl::log::{refuse, Log, Res, Stop};
use vyges_gpl::placer_base::{Blockage, DbInst, DbNet, Design, Group, Port, Rect, Region};
use vyges_gpl::run::Outcome;
use vyges_opendb::Db;

fn some(s: String) -> Option<String> {
    (!s.is_empty()).then_some(s)
}

fn e(err: vyges_opendb::Error) -> Stop {
    Stop::Refused(format!("database: {err}"))
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
    for name in db.block_get_nets() {
        let sig = db.net_get_sig_type(&name);
        let bterms = db
            .net_get_b_terms(&name)
            .into_iter()
            .map(|b| {
                let r = Rect { lx: db.bterm_get_b_box_x_min(&b), ly: db.bterm_get_b_box_y_min(&b), ux: db.bterm_get_b_box_x_max(&b), uy: db.bterm_get_b_box_y_max(&b) };
                Port { bbox: (r.lx <= r.ux && r.ly <= r.uy).then_some(r), name: b }
            })
            .collect();
        d.nets.push(DbNet { signal_or_clock: sig == "SIGNAL" || sig == "CLOCK", iterms: db.net_get_i_term_count(&name) as usize, bterms, name });
    }
    let boxes = db.blockage_boxes().map_err(e)?;
    for (k, (lx, ly, ux, uy)) in boxes.into_iter().enumerate() {
        let inst = some(db.blockage_get_instance(k)).map(|i| {
            let fixed = db.inst_is_fixed(&i);
            (i, fixed)
        });
        d.blockages.push(Blockage { bbox: Rect { lx, ly, ux, uy }, inst, max_density: db.blockage_get_max_density(k) });
    }
    Ok(d)
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
    let mut log = Log::default();
    let finish = |log: Log, stop: Option<String>, extra: Value| {
        let mut v = json!({"lines": log.lines, "stop": stop});
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
                "set_thread_count" => {
                    let [n] = args.as_slice() else { return refuse("set_thread_count: needs one count") };
                    threads.asked = n.parse().map_err(|_| Stop::Refused(format!("set_thread_count {n}: not a count")))?;
                }
                "global_placement" => {
                    log.lines.clear();
                    let db = db.as_ref().ok_or_else(|| Stop::Refused("global_placement before read_db".into()))?;
                    let d = design(db)?;
                    return vyges_gpl::run::global_placement(&mut log, &d, &args, threads).map(Some);
                }
                other => return refuse(format!("{other}: not a modelled step")),
            }
            Ok(None)
        })();
        match r {
            Ok(None) => {}
            Ok(Some(Outcome::Constructed(c, bases))) => {
                let extra = json!({"outcome": "constructed", "insts": c.insts.len(), "nets": c.nets.len(), "pins": c.pins,
                    "bases": bases.iter().map(|b| json!({"group": b.group, "place": b.place_insts.len(), "fixed": b.fixed_insts.len(), "dummies": b.dummies.len()})).collect::<Vec<_>>()});
                return Ok(finish(log, None, extra));
            }
            Err(Stop::Error { tool, id }) => return Ok(finish(log, Some(format!("error {tool}-{id:04}")), json!({}))),
            Err(Stop::Refused(why)) => return Ok(finish(log, Some(format!("refused: {why}")), json!({}))),
        }
    }
    Ok(finish(log, Some("refused: no global_placement step".into()), json!({})))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: vyges-gpl <job.json>");
        std::process::exit(2);
    };
    let job: Value = match std::fs::read_to_string(path).map_err(|e| e.to_string()).and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string())) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("{path}: {e}");
            std::process::exit(2);
        }
    };
    match run(&job) {
        Ok(v) => println!("{v}"),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
