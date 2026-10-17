// SPDX-License-Identifier: Apache-2.0
//! The command and its sequence: `global_placement` (the Tcl proc: its checks, in its order),
//! then `Replace::doInitialPlace` — each stage its own function, in the reference's order.
//!
//! Modelled so far: the placer's construction (slice 1) — `PlacerBaseCommon`, then one
//! `PlacerBase` for the top level and one per region group — and initial placement (slice 2). A
//! run stops there ([`Outcome::InitiallyPlaced`]): Nesterov placement is slice 3.

use crate::log::{refuse, Log, Res};
use crate::initial_place::{self, Ip, IpVars};
use crate::nesterov;
use crate::placer_base::{self, Base, Common, Design, Vars};
use std::collections::BTreeMap;

const KEYS: &[&str] = &[
    "-bin_grid_count", "-density", "-init_density_penalty", "-init_wirelength_coef", "-min_phi_coef", "-max_phi_coef", "-overflow",
    "-reference_hpwl", "-initial_place_max_iter", "-initial_place_max_fanout", "-routability_check_overflow",
    "-routability_snapshot_overflow", "-routability_max_density", "-routability_target_rc_metric", "-routability_inflation_ratio_coef",
    "-routability_max_inflation_ratio", "-routability_min_congestion_for_inflation", "-routability_max_inflation_total",
    "-routability_net_weight_max", "-routability_congested_nets_percentage", "-routability_rc_coefficients",
    "-timing_driven_net_reweight_overflow", "-timing_driven_net_weight_max", "-timing_driven_nets_percentage",
    "-timing_driven_repair_tns_end_percent", "-keep_resize_below_overflow", "-virtual_cts_max_skew_fraction", "-random_seed",
    "-perturb_dist", "-pad_left", "-pad_right",
];
const FLAGS: &[&str] = &[
    "-skip_initial_place", "-force_center_initial_place", "-skip_nesterov_place", "-timing_driven", "-timing_driven_repair_timing",
    "-routability_driven", "-virtual_cts", "-routability_use_grt", "-skip_io", "-place_ios", "-incremental", "-disable_revert_if_diverge",
    "-disable_pin_density_adjust", "-enable_routing_congestion",
];

/// How a run ended: the placer built (its common data and each region's base, the top level
/// first unless it had no movable instance) and initially placed; Nesterov placement is next.
#[derive(Debug)]
pub enum Outcome {
    /// The placer, its bases, the initial placement's state (each database box as it left it), and
    /// whether the command stops there (`-skip_nesterov_place`).
    InitiallyPlaced(Common, Vec<Base>, Ip, bool),
    /// Nesterov placement ran: the placer, its bases, each placed instance's database location
    /// (design instance, lower-left) and the final iteration.
    Placed(Common, Vec<Base>, Vec<(usize, i32, i32)>, i32),
    /// Incremental placement found every instance placed (GPL-0156): the command ends with the
    /// placer built and nothing moved.
    Unchanged(Common, Vec<Base>),
}

/// `sta::parse_key_args` then `check_argc_eq0`: exact keys and flags, else the first key, then
/// the first flag, that the argument is a prefix of (`-bin` is `-bin_grid_count`); STA-0562 for
/// none; STA-0564 for a positional argument.
fn parse_key_args(log: &mut Log, cmd: &str, args: &[String], keys: &[&str], flags: &[&str]) -> Res<(BTreeMap<String, String>, Vec<String>)> {
    let (mut k, mut f) = (BTreeMap::new(), Vec::new());
    let mut positional = false;
    // `is_keyword_arg`: a dash, then a letter.
    let keyword = |a: &str| a.len() >= 2 && a.starts_with('-') && a.as_bytes()[1].is_ascii_alphabetic();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        if !keyword(a) {
            positional = true;
            i += 1;
            continue;
        }
        // An exact key, an exact flag, then the FIRST key, else the first flag, it begins.
        let key = keys.iter().find(|x| **x == a).or_else(|| if flags.contains(&a) { None } else { keys.iter().find(|x| x.starts_with(a)) });
        let flag = if key.is_some() { None } else { flags.iter().find(|x| **x == a).or_else(|| flags.iter().find(|x| x.starts_with(a))) };
        if let Some(key) = key {
            let Some(v) = args.get(i + 1) else {
                let id = if keys.contains(&a) { 560 } else { 561 };
                return log.error("STA", id, format!("{cmd} {key} missing value."));
            };
            k.insert(key.to_string(), v.clone());
            i += 2;
        } else if let Some(flag) = flag {
            f.push(flag.to_string());
            i += 1;
        } else {
            return log.error("STA", 562, format!("{cmd} {a} is not a known keyword or flag."));
        }
    }
    // `check_argc_eq0`.
    if positional {
        return log.error("STA", 564, format!("{cmd} positional arguments not supported."));
    }
    Ok((k, f))
}

/// The thread count asked for (`set_thread_count`, 1 by default) and the machine's physical cores.
#[derive(Debug, Clone, Copy)]
pub struct Threads {
    pub asked: usize,
    pub cores: usize,
}

/// `placementThreads()`: the asked count, capped at the physical cores (GPL-0190 once, when capped).
fn placement_threads(log: &mut Log, t: Threads) -> usize {
    if t.asked <= t.cores {
        return t.asked;
    }
    log.info("GPL", 190, format!("Using {} of {} threads for placement: one per core, SMT siblings idle.", t.cores, t.asked));
    t.cores
}

/// Writes a Nesterov run's placement into the database and reports its HPWL (GPL-1018) — for a
/// run that is not the command's last.
pub type DbHook<'a> = &'a mut dyn FnMut(&mut Log, &[(usize, i32, i32)]) -> Res<()>;

/// The `global_placement` proc.
pub fn global_placement(log: &mut Log, d: &Design, args: &[String], threads: Threads, hook: DbHook<'_>) -> Res<Outcome> {
    let (keys, flags) = parse_key_args(log, "global_placement", args, KEYS, FLAGS)?;
    let flag = |f: &str| flags.iter().any(|x| x == f);
    if flag("-place_ios") {
        for (other, id, what) in [("-skip_io", 169, "-skip_io"), ("-incremental", 170, "-incremental"), ("-skip_nesterov_place", 182, "-skip_nesterov_place"), ("-timing_driven", 179, "-timing_driven"), ("-routability_driven", 181, "-routability_driven")] {
            if flag(other) {
                return log.error("GPL", id, format!("-place_ios cannot be used with {what} placement."));
            }
        }
    }
    // replace_initial_place_cmd: the options, then doInitialPlace.
    let int = |k: &str| -> Res<i32> { keys.get(k).map_or(Ok(0), |v| v.trim().parse::<i32>().or_else(|_| refuse(format!("{k} {v}: not an integer")))) };
    let vars = Vars { pad_left: int("-pad_left")?, pad_right: int("-pad_right")?, skip_io: flag("-skip_io"), place_ios: flag("-place_ios"), disable_pin_density_adjust: flag("-disable_pin_density_adjust") };
    let mut ip = IpVars { force_center: flag("-force_center_initial_place"), ..IpVars::default() };
    if flag("-skip_initial_place") {
        ip.max_iter = 0;
    }
    if keys.contains_key("-initial_place_max_iter") {
        ip.max_iter = int("-initial_place_max_iter")?.max(0) as usize;
    }
    if keys.contains_key("-initial_place_max_fanout") {
        ip.max_fanout = int("-initial_place_max_fanout")?.max(0) as usize;
    }
    // `PlaceOptions::skipIo()`, applied after the keys: no initial placement either.
    if flag("-skip_io") {
        ip.max_iter = 0;
    }
    // The thread count does not change construction; the message is the command's first line.
    let _threads = placement_threads(log, threads);
    if flag("-incremental") {
        // replace_incremental_place_cmd: the options, then doIncrementalPlace.
        let o = nesterov_opts(&keys, &flags, &vars)?;
        return do_incremental_place(log, d, &vars, &ip, o, hook);
    }
    let skip_nesterov = flag("-skip_nesterov_place");
    // replace_nesterov_place_cmd's options, checked before any work so a refusal costs nothing.
    let np = if skip_nesterov { None } else { Some(nesterov_opts(&keys, &flags, &vars)?) };
    let outcome = do_initial_place(log, d, &vars, &ip, skip_nesterov)?;
    match (outcome, np) {
        (Outcome::InitiallyPlaced(mut c, bases, mut ip, false), Some(o)) => {
            let mut nb = nesterov::init_nesterov_place(log, d, &mut c, &mut ip, &bases, &o)?;
            let iter = nesterov::do_nesterov_place(log, d, &mut nb)?;
            let locations = nesterov::placed_locations(&c, &nb);
            Ok(Outcome::Placed(c, bases, locations, iter))
        }
        (other, _) => Ok(other),
    }
}

/// `getOptions` for the plain Nesterov mode; the modes it does not model are refused.
fn nesterov_opts(keys: &BTreeMap<String, String>, flags: &[String], v: &Vars) -> Res<nesterov::Opts> {
    for f in ["-timing_driven", "-timing_driven_repair_timing", "-routability_driven", "-virtual_cts", "-enable_routing_congestion"] {
        if flags.iter().any(|x| x == f) {
            return refuse(format!("global_placement {f} is not modelled"));
        }
    }
    let float = |k: &str, d: f32| -> Res<f32> { keys.get(k).map_or(Ok(d), |s| s.trim().parse::<f32>().or_else(|_| refuse(format!("{k} {s}: not a number")))) };
    let int = |k: &str, d: i32| -> Res<i32> { keys.get(k).map_or(Ok(d), |s| s.trim().parse::<i32>().or_else(|_| refuse(format!("{k} {s}: not an integer")))) };
    let mut o = nesterov::Opts::default();
    match keys.get("-density").map(String::as_str) {
        Some("uniform") => o.uniform_density = true,
        Some(_) => o.density = float("-density", 0.7)?,
        None => {}
    }
    if keys.contains_key("-bin_grid_count") {
        o.bin_cnt = Some(int("-bin_grid_count", 0)?);
    }
    o.init_density_penalty = float("-init_density_penalty", o.init_density_penalty)?;
    o.init_wirelength_coef = float("-init_wirelength_coef", o.init_wirelength_coef)?;
    o.min_phi_coef = float("-min_phi_coef", o.min_phi_coef)?;
    o.max_phi_coef = float("-max_phi_coef", o.max_phi_coef)?;
    o.overflow = float("-overflow", o.overflow)?;
    o.reference_hpwl = float("-reference_hpwl", o.reference_hpwl)?;
    o.seed = int("-random_seed", o.seed)?;
    o.perturb_dist = float("-perturb_dist", o.perturb_dist)?;
    o.disable_revert_if_diverge = flags.iter().any(|x| x == "-disable_revert_if_diverge");
    o.pad_left = v.pad_left;
    Ok(o)
}

/// `Replace::doInitialPlace`, to the placer's construction.
fn do_initial_place(log: &mut Log, d: &Design, v: &Vars, ipv: &IpVars, skip_nesterov: bool) -> Res<Outcome> {
    check_has_core_rows(log, d)?;
    if v.place_ios {
        return refuse("-place_ios (checkPlaceIosSupported: the die polygon) is not modelled");
    }
    let mut common = placer_base::placer_base_common(log, d, v)?;
    let mut bases = vec![placer_base::placer_base(log, d, &common, None, true)?];
    for region in &d.regions {
        for &g in &region.groups {
            bases.push(placer_base::placer_base(log, d, &common, Some(g), true)?);
        }
    }
    if bases[0].place_insts.is_empty() {
        log.warn("GPL", 123, "No placeable instances in the top-level region. Removing it.");
        bases.remove(0);
    }
    let ip = initial_place::do_bicgstab_place(log, d, &mut common, v, ipv);
    Ok(Outcome::InitiallyPlaced(common, bases, ip, skip_nesterov))
}

/// `Replace::doIncrementalPlace`: the placer, with every instance already PLACED locked; the
/// placed and unplaced counts (GPL-0154/0155) — nothing unplaced ends the command (GPL-0156); then
/// a rough pass with the locked ones held (overflow at least 0.2, at most 300 iterations, uniform
/// target density), every instance unlocked (GPL-0133), and — when the asked overflow is lower —
/// the Nesterov loop continued from the next iteration to it, on the same placer.
fn do_incremental_place(log: &mut Log, d: &Design, v: &Vars, ipv: &IpVars, o: nesterov::Opts, hook: DbHook<'_>) -> Res<Outcome> {
    check_has_core_rows(log, d)?;
    log.info("GPL", 83, "Execute incremental mode global placement.");
    let mut common = placer_base::placer_base_common(log, d, v)?;
    let (mut placed, mut unplaced) = (0usize, 0usize);
    let mut pb_of = vec![None; d.insts.len()];
    for (i, inst) in common.insts.iter().enumerate() {
        if let Some(k) = inst.db {
            pb_of[k] = Some(i);
        }
    }
    for (k, di) in d.insts.iter().enumerate() {
        if di.placed && !di.fixed {
            // `dbToPb(inst)->lock()`
            if let Some(i) = pb_of[k] {
                common.insts[i].locked = true;
            }
            placed += 1;
        } else if !di.placed {
            unplaced += 1;
        }
    }
    let mut bases = vec![placer_base::placer_base(log, d, &common, None, true)?];
    for region in &d.regions {
        for &g in &region.groups {
            bases.push(placer_base::placer_base(log, d, &common, Some(g), true)?);
        }
    }
    log.info("GPL", 154, format!("Identified {placed} placed instances"));
    log.info("GPL", 155, format!("Identified {unplaced} not placed instances"));
    if unplaced == 0 {
        log.info("GPL", 156, "Identified all instances as placed. Skipping incremental mode");
        return Ok(Outcome::Unchanged(common, bases));
    }
    let mut locked = o.clone();
    locked.overflow = o.overflow.max(0.2);
    locked.max_iter = 300;
    locked.uniform_density = true;
    // doInitialPlace on the placer that exists: no construction, no GPL-0123.
    if v.place_ios {
        return refuse("-place_ios (checkPlaceIosSupported: the die polygon) is not modelled");
    }
    let mut ip = initial_place::do_bicgstab_place(log, d, &mut common, v, ipv);
    let mut np = nesterov::init_nesterov_place(log, d, &mut common, &mut ip, &bases, &locked)?;
    let mut iter = nesterov::do_nesterov_place(log, d, &mut np)?;
    // `updateDb` then `reportHpwlMetric` at the end of this pass
    hook(log, &nesterov::placed_locations(&common, &np))?;
    log.info("GPL", 133, "Unlocking all instances");
    nesterov::unlock_all(&mut np, &mut common);
    if o.overflow < locked.overflow {
        iter = nesterov::continue_nesterov_place(log, d, &mut np, o.overflow, o.max_iter, iter + 1)?;
    }
    let locations = nesterov::placed_locations(&common, &np);
    Ok(Outcome::Placed(common, bases, locations, iter))
}

/// `checkHasCoreRows` (`dbHasCoreRows`: a row whose site is not a PAD site).
fn check_has_core_rows(log: &mut Log, d: &Design) -> Res<()> {
    if !d.rows.iter().any(|r| !r.1) {
        return log.error("GPL", 130, "No rows defined in design. Use initialize_floorplan to add rows.");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    // Rule (sta::parse_key_args): exact matches first; else the FIRST key, then flag, the argument
    // begins (`-bin` is `-bin_grid_count`); none is STA-0562; a positional argument STA-0564.
    #[test]
    fn keys_match_by_prefix() {
        let mut log = Log::default();
        let (k, f) = parse_key_args(&mut log, "global_placement", &s(&["-bin", "64", "-skip_io", "-disable_pin"]), KEYS, FLAGS).unwrap();
        assert_eq!(k.get("-bin_grid_count").map(String::as_str), Some("64"));
        assert_eq!(f, ["-skip_io", "-disable_pin_density_adjust"]);
        assert!(parse_key_args(&mut log, "global_placement", &s(&["-verbose"]), KEYS, FLAGS).is_err());
        assert_eq!(log.lines.last().unwrap(), "[ERROR STA-0562] global_placement -verbose is not a known keyword or flag.");
        assert!(parse_key_args(&mut log, "global_placement", &s(&["x"]), KEYS, FLAGS).is_err());
        assert_eq!(log.lines.last().unwrap(), "[ERROR STA-0564] global_placement positional arguments not supported.");
    }
}
