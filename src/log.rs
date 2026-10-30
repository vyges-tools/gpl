// SPDX-License-Identifier: Apache-2.0
//! The run's output, line by line as the reference's logger writes it, and how a run stops.

use vyges_events::logger::Logger;

/// The run's message lines, written by the suite's logger (`vyges_events::logger::Logger`): the
/// reference's line format, its per-code print limit and its debug levels.
#[derive(Debug, Default)]
pub struct Log {
    logger: Logger,
    /// Raw-value probe lines (`VYG…`), compared with an instrumented reference's; never output.
    pub trace: Vec<String>,
}

/// Why a run ended early: the reference's own error (its line is in the log), or a behaviour this
/// engine does not model (no line; the run is refused, never guessed).
#[derive(Debug, Clone, PartialEq)]
pub enum Stop {
    Error { tool: &'static str, id: u32 },
    Refused(String),
}

pub type Res<T> = Result<T, Stop>;

/// A refusal.
pub fn refuse<T>(why: impl Into<String>) -> Res<T> {
    Err(Stop::Refused(why.into()))
}

impl Log {
    /// A log whose lines are also printed to stderr as they are written (the run's report keeps them
    /// too) and emitted as events under `tool`.
    pub fn with_events(tool: &str) -> Self {
        Log { logger: Logger::new().events(tool).stderr(true), trace: Vec::new() }
    }

    /// The lines printed so far.
    pub fn lines(&self) -> &[String] {
        self.logger.lines()
    }

    /// The lines printed so far; the log keeps its counts and debug levels.
    pub fn take_lines(&mut self) -> Vec<String> {
        self.logger.take_lines()
    }

    pub fn info(&mut self, tool: &str, id: u32, msg: impl AsRef<str>) {
        self.logger.info(tool, id16(id), msg);
    }

    pub fn warn(&mut self, tool: &str, id: u32, msg: impl AsRef<str>) {
        self.logger.warn(tool, id16(id), msg);
    }

    /// `logger_->error`: the line, then the run stops.
    pub fn error<T>(&mut self, tool: &'static str, id: u32, msg: impl AsRef<str>) -> Res<T> {
        let Err(_) = self.logger.error::<()>(tool, id16(id), msg) else { unreachable!("an error stops") };
        Err(Stop::Error { tool, id })
    }

    /// `logger_->report`: the text alone.
    pub fn report(&mut self, msg: impl AsRef<str>) {
        self.logger.report(msg);
    }

    /// A line another engine's logger printed (the resizer's, at a reweight), written as that
    /// logger wrote it: a coded line keeps its level and code, anything else is report text.
    pub fn relay(&mut self, line: &str) {
        let coded = line.strip_prefix('[').and_then(|r| r.split_once("] ")).and_then(|(head, text)| {
            let (level, code) = head.split_once(' ')?;
            let (tool, id) = vyges_events::logger::parse_code(code)?;
            Some((level, tool, id, text))
        });
        match coded {
            Some(("INFO", tool, id, text)) => self.logger.info(tool, id, text),
            Some(("WARNING", tool, id, text)) => self.logger.warn(tool, id, text),
            Some(("ERROR", tool, id, text)) => {
                let _ = self.logger.error::<()>(tool, id, text);
            }
            _ => self.logger.report(line),
        }
    }

    /// A raw-value probe line.
    pub fn trace(&mut self, line: String) {
        self.trace.push(line);
    }

    /// `set_debug_level <tool> <group> <level>`.
    pub fn set_debug_level(&mut self, tool: &str, group: &str, level: i32) {
        self.logger.set_debug_level(tool, group, level);
    }

    /// `debugPrint`'s check: the group's level is at least `level`.
    pub fn debug_on(&self, tool: &str, group: &str, level: i32) -> bool {
        self.logger.debug_on(tool, group, level)
    }

    pub fn debug(&mut self, tool: &str, group: &str, level: i32, msg: impl FnOnce() -> String) {
        self.logger.debug(tool, group, level, msg);
    }
}

fn id16(id: u32) -> u16 {
    u16::try_from(id).expect("a message id fits four digits")
}

/// `fmt`'s `{}` for a float, from the value's own shortest digits (`fixed`: `{}`, `sci`: `{:e}`):
/// fixed for decimal exponents in [-4, 16), a whole number without a fraction; else `d.ddde±XX`.
fn fmt_g(finite_nonzero: bool, fixed: String, sci: String) -> String {
    if !finite_nonzero {
        return match fixed.as_str() {
            "NaN" => "nan".into(),
            "inf" | "-inf" => fixed,
            _ => fixed.trim_end_matches(".0").to_string(),
        };
    }
    let (mantissa, exp) = sci.split_once('e').expect("exponent form");
    let exp: i32 = exp.parse().expect("an exponent");
    if (-4..16).contains(&exp) {
        fixed.strip_suffix(".0").map(String::from).unwrap_or(fixed)
    } else {
        format!("{mantissa}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

/// `fmt`'s `{}` for a `float`.
pub fn fmt_f32(v: f32) -> String {
    fmt_g(v.is_finite() && v != 0.0, format!("{v}"), format!("{v:e}"))
}

/// `fmt`'s `{}` for a `double`.
pub fn fmt_f64(v: f64) -> String {
    fmt_g(v.is_finite() && v != 0.0, format!("{v}"), format!("{v:e}"))
}

/// `fmt`'s `{:0.Ne}`: N fraction digits, a signed exponent of at least two digits.
pub fn fmt_e(v: f64, digits: usize) -> String {
    let s = format!("{v:.digits$e}");
    let (mantissa, exp) = s.split_once('e').expect("exponent form");
    let exp: i32 = exp.parse().expect("an exponent");
    format!("{mantissa}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Rule (fmt `{}`): shortest round-trip digits; fixed for exponents in [-4, 16), whole numbers
    // without a fraction; else scientific with a two-digit signed exponent.
    #[test]
    fn floats_print_as_fmt_does() {
        assert_eq!(fmt_f64(0.01), "0.01");
        assert_eq!(fmt_f64(1.0), "1");
        assert_eq!(fmt_f32(1.5e-14), "1.5e-14");
        assert_eq!(fmt_f32(0.1), "0.1");
        assert_eq!(fmt_f64(2.5e-5), "2.5e-05");
        assert_eq!(fmt_e(1.234e-14, 2), "1.23e-14");
        assert_eq!(fmt_e(12345.0, 2), "1.23e+04");
    }

    // Rule (logger): a message with newlines is several output lines; an error stops the run.
    #[test]
    fn lines_split_and_errors_stop() {
        let mut l = Log::default();
        l.warn("CTS", 104, "a\nb");
        assert_eq!(l.lines(), ["[WARNING CTS-0104] a", "b"]);
        let r: Res<()> = l.error("CTS", 73, "x");
        assert_eq!(r, Err(Stop::Error { tool: "CTS", id: 73 }));
    }

    // Rule (logger): a relayed line prints as it was written, coded or not.
    #[test]
    fn relayed_lines_print_unchanged() {
        let mut l = Log::default();
        for line in ["[INFO RSZ-0058] Using max wire length 693um.", "[WARNING RSZ-0062] x", "Iteration | Area", "[DEBUG RSZ-repair_net] y"] {
            l.relay(line);
        }
        assert_eq!(l.lines(), ["[INFO RSZ-0058] Using max wire length 693um.", "[WARNING RSZ-0062] x", "Iteration | Area", "[DEBUG RSZ-repair_net] y"]);
    }

    // Rule (logger): the run has one logger, so a relayed code counts toward its print limit.
    #[test]
    fn relayed_lines_count_toward_the_print_limit() {
        let mut l = Log::default();
        for _ in 0..1002 {
            l.relay("[WARNING RSZ-0062] x");
        }
        assert_eq!(l.lines().len(), 1001);
        assert_eq!(l.lines()[1000], "[WARNING RSZ-0062] message limit (1000) reached. This message will no longer print.");
    }
}
