# vyges-gpl

Global placement. The engine builds the placer's view of the design — instances, nets, pins,
regions, the sites a placement may not use — runs the initial placement (rounds of a
bound-to-bound net model solved by BiCGSTAB, in exactly the arithmetic order of the reference
solver) and then the Nesterov electrostatic placement, so every coordinate matches bit for bit.

```sh
vyges install physical
vyges physical gpl global_placement job.json
vyges physical gpl --describe
```

Documentation: [`docs/src/gpl.md`](docs/src/gpl.md) (an mdBook; `mdbook build docs`), with the
[CLI reference](docs/src/reference/vyges-gpl.md).

## The job

```json
{
  "steps": [
    { "cmd": "read_db",          "args": ["floorplan.odb"] },
    { "cmd": "global_placement", "args": ["-skip_nesterov_place"] }
  ],
  "write_def": "placed.def"
}
```

The report carries the command's own lines in order — the placer's header (core, site, instance
and pin counts, areas, utilisation) and each `[InitialPlace]` round with its residual and HPWL —
and a `status`:

```json
{ "tool": "vyges-gpl", "status": "placed", "insts": 294, "nets": 364, "pins": 1122,
  "lines": [ "[INFO GPL-0001] ---- Initialize GPL Main Data Structures", … ] }
```

## What is modelled

- the placer's construction: instances (fixed ones snapped outward to whole sites, padding),
  the pin-density area adjustment, signal and clock nets with their pins, regions, dummy
  instances over unusable sites;
- the initial placement: start points, the bound-to-bound matrix for x and y, the solver with
  its restart, the clamp to the core and region, the database written as the reference writes it;
- the Nesterov (electrostatic) placement: fillers, the pre-placement perturbation, the bin grid,
  the DCT density field, the weighted-average wirelength gradient, the accelerated descent with
  back-tracking and the divergence revert — over the top level and every region group, each
  region its own density field;
- incremental placement: the instances already placed held where they are while the new ones
  settle, then everything released for a second pass;
- placement clusters: a group of instances moved as one cell, sized to their total area.

Refused rather than guessed: `-timing_driven`, `-routability_driven`, `-virtual_cts`,
`-place_ios`.

## Build

Check out `vyges-tools/opendb`, `opendb-lib`, `loom` and `layout` beside this repository (see
`.github/workflows/ci.yml`), then:

```sh
cargo build --release --features cli
cargo test  --release --features cli
```

Without `--features cli` the library builds with no C++ dependency.

Licensed under Apache-2.0.
