//! Phase reporting and the memory table.
//!
//! Output is line-oriented and greppable so a CI log is readable
//! without a parser: one `PASS <phase>` line per completed phase, an
//! optional memory column, and a summary table at the end.

use crate::mem;

/// Accumulates phase results and one memory reading per phase.
pub struct Reporter {
    report_memory: bool,
    samples: Vec<(String, Option<u64>)>,
    source: Option<&'static str>,
}

impl Reporter {
    /// Create a reporter. `report_memory` adds the per-phase memory
    /// column and the summary table.
    pub fn new(report_memory: bool) -> Self {
        Self {
            report_memory,
            samples: Vec::new(),
            source: None,
        }
    }

    /// Record and print a completed phase.
    pub fn pass(&mut self, phase: &str) {
        let reading = mem::sample();
        if let Some((_, source)) = reading {
            self.source = Some(source);
        }
        let kib = reading.map(|(kib, _)| kib);
        match (self.report_memory, kib) {
            (true, Some(kib)) => println!("PASS  {phase:<32} mem={kib} KiB"),
            (true, None) => println!("PASS  {phase:<32} mem=not measured on this target"),
            (false, _) => println!("PASS  {phase}"),
        }
        self.samples.push((phase.to_string(), kib));
    }

    /// Print an informational line that is not a phase result.
    pub fn note(&self, text: &str) {
        println!("      {text}");
    }

    /// Print the per-phase memory table and the high-water mark.
    ///
    /// Prints "not measured" rather than a zero when the target has no
    /// reading, so an unmeasurable platform cannot be mistaken for a
    /// free one.
    pub fn summary(&self) {
        if !self.report_memory {
            return;
        }
        println!();
        match self.source {
            Some(source) => {
                println!("memory by phase ({source}):");
                for (phase, kib) in &self.samples {
                    match kib {
                        Some(kib) => println!("  {phase:<32} {kib:>8} KiB"),
                        None => println!("  {phase:<32} not measured"),
                    }
                }
                match self.peak() {
                    Some(peak) => println!("HIGH-WATER {peak} KiB ({source})"),
                    None => println!("HIGH-WATER not measured"),
                }
            }
            None => println!("memory: not measured on this target"),
        }
    }

    /// Largest reading seen, or `None` when nothing was measurable.
    pub fn peak(&self) -> Option<u64> {
        self.samples.iter().filter_map(|(_, kib)| *kib).max()
    }
}
