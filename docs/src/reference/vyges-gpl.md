# vyges physical gpl — CLI reference

_Generated from `vyges physical gpl --help` — this page is the tool's own output, verbatim._

```text
vyges physical gpl — global placement: the placer's view of the design (instances, nets, pins,
regions), the initial placement, then the Nesterov electrostatic placement

USAGE:
  vyges physical gpl global_placement <job.json> [-o FILE]
  vyges physical gpl --describe
  vyges physical gpl --help
  vyges physical gpl --version

JOB FIELDS:
  steps        required — the commands in order, each {"cmd": ..., "args": [...]}:
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
```
