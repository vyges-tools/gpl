// SPDX-License-Identifier: Apache-2.0
//! Global placement of a design: the placer's view of it (instances, nets, pins, regions), then
//! (later slices) initial placement and the Nesterov loop.
//!
//! The modules follow the command's call sequence:
//! - [`run`]: the command and `doInitialPlace`, a thin sequencer of the stages in their order.
//! - [`placer_base`]: the placer's construction — `PlacerBaseCommon` and one `PlacerBase` per
//!   region.
//! - [`initial_place`]: the bound-to-bound initial placement, solved by [`bicgstab`].
//! - [`log`]: the output lines and how a run stops.
//! - [`ooura`]: the 2-D cosine/sine transforms of the density solver.

// The crate's own name inside it: the command-line entry ([`cli_main`]) is built into the
// library too, and its `vyges_gpl::…` paths resolve here.
extern crate self as vyges_gpl;

pub mod bicgstab;
#[cfg(feature = "cli")]
pub mod db_router;
pub mod initial_place;
pub mod io_pins;
pub mod ld;
pub mod log;
pub mod nesterov;
pub mod normal;
pub mod ooura;
pub mod order;
pub mod placer_base;
pub mod route_base;
pub mod run;
pub mod timing_driven;

/// The `vyges-gpl` command-line entry (the binary's `main`), built into the library too so a
/// suite binary can link every engine and run the one it is invoked as.
#[cfg(feature = "suite")]
#[doc(hidden)]
#[path = "main.rs"]
pub mod cli_main;
