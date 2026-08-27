//! Generator for Jepsen-format histories of concurrent lark transactions.
//!
//! See README.md for how to check a generated history with elle-cli.

mod cli;
mod faults;
mod history;
mod model;
mod runner;
mod verify;

use cli::{Config, Isolation};

fn main() {
    let cfg = match cli::parse(std::env::args().skip(1)) {
        Ok(cfg) => cfg,
        Err(message) => {
            eprintln!("{}", message);
            std::process::exit(2);
        }
    };

    if let Err(message) = dispatch(&cfg) {
        eprintln!("elle-gen: {}", message);
        std::process::exit(1);
    }
}

fn dispatch(cfg: &Config) -> Result<(), String> {
    if let Some(path) = &cfg.verify_only {
        return report_verification(&verify::verify(path, cfg.model)?);
    }

    warn_unreachable_isolation(cfg.isolation);

    if cfg.worker.is_some() {
        return runner::run_worker(cfg);
    }

    let count = runner::run(cfg)?;
    eprintln!(
        "wrote {} operations to {} (model {}, isolation {})",
        count,
        cfg.out.display(),
        cfg.model.as_str(),
        cfg.isolation.as_str()
    );
    report_verification(&verify::verify(&cfg.out, cfg.model)?)
}

/// lark provides snapshot isolation only. Say so before generating a
/// history at a level the engine cannot reach, so nobody reads the
/// checker verdict as a lark bug when it is legal snapshot-isolated
/// behavior.
fn warn_unreachable_isolation(isolation: Isolation) {
    let gap = match isolation {
        Isolation::ReadCommitted => return,
        Isolation::RepeatableRead => {
            "snapshot isolation is incomparable with repeatable-read: it permits write \
             skew (G2-item), which repeatable-read forbids"
        }
        Isolation::Serializable => {
            "snapshot isolation is strictly weaker than serializable: it permits write \
             skew (G2-item), which serializable forbids"
        }
    };
    eprintln!(
        "warning: {} isolation is NOT reachable on this tree. lark exposes snapshot \
         isolation only, through TransactionDb and OptimisticTransactionDb, and {}. \
         The generated history is a snapshot-isolated history; check it with \
         `--consistency-models snapshot-isolation` for a verdict that means something \
         about lark. A failure at {} may be legal behavior, not a bug.",
        isolation.as_str(),
        gap,
        isolation.as_str()
    );
}

fn report_verification(report: &verify::Report) -> Result<(), String> {
    eprintln!(
        "built-in check: {} operations, {} committed, {} anomaly witnesses",
        report.ops,
        report.committed,
        report.witnesses.len()
    );
    for witness in report.witnesses.iter().take(5) {
        eprintln!("  {}", witness);
    }
    if report.anomalous() {
        return Err("history contains consistency anomalies".to_string());
    }
    Ok(())
}
