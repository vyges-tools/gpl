# vyges physical gpl — CLI reference

_Generated from `vyges physical gpl --help` — this page is the tool's own output, verbatim._

```text
vyges physical gpl — global placement: the placer's view of the design (instances, nets, pins,
regions) and the initial placement (Nesterov placement is not modelled yet)

USAGE:
  vyges physical gpl global_placement <job.json> [-o FILE]
  vyges physical gpl --describe
  vyges physical gpl --help
  vyges physical gpl --version

JOB FIELDS:
  steps        required — the commands in order, each {"cmd": ..., "args": [...]}:
               read_db, set_debug_level, set_thread_count, then global_placement [options]
  write_def    with -skip_nesterov_place: write the placed design, as DEF, to this path

MODELLED:
  the placer's construction (GPL-0001..0021, 0035..0037: instances, fixed ones snapped to
  sites, padding, the pin-density area adjustment, nets and pins, unusable-site dummies), and the
  initial placement — every bound-to-bound round and its [InitialPlace] line, the solver's
  arithmetic in the reference's own operation order. With -skip_nesterov_place that is the
  whole command (status placed); otherwise the command is refused after it
  (modelled_through: initial_place). Refused: -incremental, -place_ios, region groups,
  placement clusters.

OPTIONS:
  -o FILE               write the JSON report to FILE instead of stdout
  --json                accepted; the report is JSON either way
  --describe            print a machine-readable JSON description of the command
  --bug-report          file a bug (central: vyges/community)
  --feature-request     request a feature (central)
  --sponsor             sponsor Vyges (github.com/sponsors/vyges-ip)
  --star                star this tool on GitHub

EXIT STATUS:
  0  placed       -skip_nesterov_place: the design initially placed
  2  vacuous      the job has no global_placement step. NOT a pass.
  2  error        usage, unreadable input, or an error the command raises (its line is reported)
  3  refused      an input this engine does not model — or a Nesterov placement to run
```
