//! Transcript emitter.
//!
//! Three line kinds, on purpose. `CHECK` lines carry only observable
//! results (counts, digests, byte lengths, error text) and must be
//! byte-identical between a native run and a wasm run of the same
//! phase; that comparison is the point of the harness. `STAGE` lines
//! carry memory, which is a different quantity on each platform and is
//! labelled with what produced it. `NOTE` lines are prose and are not
//! compared.

/// How many individual failures are printed before the emitter stops
/// listing them. A systemic failure would otherwise bury the
/// transcript; the count in `SUMMARY` stays exact either way.
const FAILURE_PRINT_LIMIT: u32 = 20;

/// Accumulates counts and prints the transcript as it goes.
pub struct Report {
    stage: u32,
    checks: u32,
    failures: u32,
    printed_failures: u32,
    high_water: Option<(u64, &'static str)>,
}

impl Report {
    /// A fresh report with no stages and no failures.
    pub fn new() -> Self {
        Self {
            stage: 0,
            checks: 0,
            failures: 0,
            printed_failures: 0,
            high_water: None,
        }
    }

    /// Print a memory reading labelled with `label`.
    pub fn stage(&mut self, label: &str) {
        self.stage += 1;
        match crate::mem::sample() {
            Some((kib, source)) => {
                let worse = self.high_water.is_none_or(|(prev, _)| kib > prev);
                if worse {
                    self.high_water = Some((kib, source));
                }
                println!("STAGE {:02} {:<34} mem={} KiB", self.stage, label, kib);
            }
            None => println!("STAGE {:02} {:<34} mem=not measured", self.stage, label),
        }
    }

    /// Record an observable result. These lines are the ones compared
    /// across platforms.
    pub fn check(&mut self, name: &str, value: &str) {
        self.checks += 1;
        println!("CHECK {name} = {value}");
    }

    /// Record an observable numeric result.
    pub fn check_u64(&mut self, name: &str, value: u64) {
        self.check(name, &value.to_string());
    }

    /// Record a digest, rendered fixed-width so a diff is unambiguous.
    pub fn check_digest(&mut self, name: &str, value: u64) {
        self.check(name, &format!("{value:016x}"));
    }

    /// Assert `actual == expected`, recording both and counting a
    /// failure when they differ.
    pub fn expect_u64(&mut self, name: &str, actual: u64, expected: u64) {
        self.check_u64(name, actual);
        if actual != expected {
            self.fail(name, &format!("expected {expected}, got {actual}"));
        }
    }

    /// Assert two digests match.
    pub fn expect_digest(&mut self, name: &str, actual: u64, expected: u64) {
        self.check_digest(name, actual);
        if actual != expected {
            self.fail(
                name,
                &format!("expected {expected:016x}, got {actual:016x}"),
            );
        }
    }

    /// Record a failure. Counted exactly, printed up to a cap.
    pub fn fail(&mut self, name: &str, detail: &str) {
        self.failures += 1;
        if self.printed_failures < FAILURE_PRINT_LIMIT {
            self.printed_failures += 1;
            println!("FAIL  {name}: {detail}");
        } else if self.printed_failures == FAILURE_PRINT_LIMIT {
            self.printed_failures += 1;
            println!("FAIL  further failures suppressed; SUMMARY carries the exact count");
        }
    }

    /// Print a multi-line database property, one transcript line per
    /// source line, so it stays diffable.
    pub fn property(&mut self, name: &str, text: &str) {
        for line in text.lines() {
            println!("PROP  {name} | {line}");
        }
    }

    /// Print a line of prose that is not part of the comparison.
    pub fn note(&mut self, text: &str) {
        println!("NOTE  {text}");
    }

    /// Print the summary and report whether the phase passed.
    pub fn finish(&mut self, phase: &str) -> bool {
        match self.high_water {
            Some((kib, source)) => {
                println!("NOTE  memory source: {source}");
                println!("HIGHWATER {kib} KiB");
            }
            None => println!("HIGHWATER not measured"),
        }
        println!(
            "SUMMARY phase={} checks={} failures={}",
            phase, self.checks, self.failures
        );
        self.failures == 0
    }
}
