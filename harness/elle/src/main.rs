//! Generator for Jepsen-format histories of concurrent regolith transactions.
//!
//! See README.md for how to check a generated history with elle-cli.

mod cli;
mod faults;
mod history;
mod model;
mod runner;
mod verify;

use cli::Config;

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
