# vyges-gpl — global placement

> **Physical construction — opt-in, and not part of `vyges install loom`.** Install the group with
> `vyges install physical` (or just this engine with `vyges install gpl`), then run
> `vyges physical gpl`. It is also a standalone `vyges-gpl` binary on your PATH, which is what the
> dispatch calls and what a flow author targets directly.

`vyges-gpl` answers: **where does each movable instance start?** It builds the placer's view of
the design and runs the initial placement — and every coordinate it produces matches the
reference placer's bit for bit, because the solver's arithmetic is carried out in the same order.

| stage | decides |
| --- | --- |
| **construction** | the core, the site, the instances (fixed ones snapped outward to whole sites, padding on the movable ones), each instance grown or shrunk toward the design's average pin density (between 0.95× and 1.2× per side), the signal and clock nets and their pins, regions, dummy instances over every site no row covers |
| **start points** | a grouped instance at its region's centre, a placed one at its own centre, the rest at the core's centre |
| **each round** | each net's extreme pins; the bound-to-bound weights (a net's weight over its pin count, over the distance, never under 1500 DBU); the x and y systems solved by BiCGSTAB; the instances moved, clamped to the core and region |
| **stop** | after 20 rounds, or once the residual is at most 1e-5 after round 5 |

**Not yet:** the Nesterov placement. A full `global_placement` is refused after the initial
placement (`modelled_through: initial_place`); with `-skip_nesterov_place` the initial placement
is the whole command and the design is written (`write_def`).

## Status and exit codes

| status | exit | meaning |
| --- | --- | --- |
| `placed` | 0 | `-skip_nesterov_place`: the design initially placed |
| `refused` | 3 | an input this engine does not model, or a Nesterov placement to run (`reason`) |
| `vacuous` | 2 | the job has no `global_placement` step — NOT a pass |
| `error` | 2 | an error the command raises; its line is the last in `lines` |
