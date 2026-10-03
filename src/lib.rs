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

pub mod bicgstab;
pub mod initial_place;
pub mod log;
pub mod placer_base;
pub mod run;
