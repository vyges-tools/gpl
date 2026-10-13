# vyges-gpl

Global placement. Today the engine builds the placer's view of the design — instances, nets, pins,
regions, the sites a placement may not use — and runs the initial placement: rounds of a
bound-to-bound net model solved by BiCGSTAB, in exactly the arithmetic order of the reference
solver, so every coordinate matches bit for bit. The Nesterov placement comes next; until then a
full `global_placement` is refused after the initial placement, and `-skip_nesterov_place`
places the design.

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
  its restart, the clamp to the core and region, the database written as the reference writes it.

Refused rather than guessed: `-incremental`, `-place_ios`, region groups, placement clusters.

## Build

Check out `vyges-tools/opendb`, `opendb-lib`, `loom` and `layout` beside this repository (see
`.github/workflows/ci.yml`), then:

```sh
cargo build --release --features cli
cargo test  --release --features cli
```

Without `--features cli` the library builds with no C++ dependency.

Licensed under Apache-2.0.
