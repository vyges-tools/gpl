// SPDX-License-Identifier: Apache-2.0
//! The command and its sequence: `global_placement` (the Tcl proc: its checks, in its order),
//! then `Replace::doInitialPlace` — each stage its own function, in the reference's order.
//!
//! Modelled so far: the placer's construction (slice 1) — `PlacerBaseCommon`, then one
//! `PlacerBase` for the top level and one per region group — and initial placement (slice 2). A
//! run stops there ([`Outcome::InitiallyPlaced`]): Nesterov placement is slice 3.

use crate::log::{refuse, Log, Res};
use crate::initial_place::{self, Ip, IpVars};
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

/// The `global_placement` proc.
pub fn global_placement(log: &mut Log, d: &Design, args: &[String], threads: Threads) -> Res<Outcome> {
    let (keys, flags) = parse_key_args(log, "global_placement", args, KEYS, FLAGS)?;
    let flag = |f: &str| flags.iter().any(|x| x == f);
    if flag("-place_ios") {
        for (other, id, what) in [("-skip_io", 169, "-skip_io"), ("-incremental", 170, "-incremental"), ("-skip_nesterov_place", 182, "-skip_nesterov_place"), ("-timing_driven", 179, "-timing_driven"), ("-routability_driven", 181, "-routability_driven")] {
            if flag(other) {
                return log.error("GPL", id, format!("-place_ios cannot be used with {what} placement."));
            }
        }
    }
    if flag("-incremental") {
        return refuse("global_placement -incremental (doIncrementalPlace) is not modelled");
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
    do_initial_place(log, d, &vars, &ip, flag("-skip_nesterov_place"))
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
    if d.regions.iter().any(|r| !r.groups.is_empty()) {
        return refuse("initial placement with region groups is not modelled");
    }
    let ip = initial_place::do_bicgstab_place(log, d, &mut common, v, ipv);
    Ok(Outcome::InitiallyPlaced(common, bases, ip, skip_nesterov))
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
