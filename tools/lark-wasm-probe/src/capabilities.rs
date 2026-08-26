//! Cross-check between what the host does and what lark says it does.
//!
//! [`crate::host`] measures each primitive by calling it. `Env`
//! declares the same set through [`lark_kv::Capabilities`]. This phase
//! puts the two side by side and fails when they disagree, which is
//! the only way to catch an environment that inherited a flag from a
//! `cfg` instead of from the platform - the exact mistake that made
//! `sync_dir` claim a durability WASI does not provide.
//!
//! Only the flags a host probe can settle are compared. Nothing here
//! infers a capability from a `cfg`, because that would be checking
//! the claim against itself.

use lark_kv::{Capabilities, Db};

use crate::host::Finding;

/// Compare `db`'s declared capabilities against the measured
/// `findings`, naming every disagreement.
pub fn check(db: &Db, findings: &[Finding]) -> Result<(), String> {
    let declared = db.capabilities();
    let mut wrong = Vec::new();

    for (probe, claimed, flag) in [
        (
            "directory fsync (sync_dir)",
            declared.sync_dir,
            "Capabilities::sync_dir",
        ),
        ("hard_link", declared.hard_link, "Capabilities::hard_link"),
        (
            "rename over existing (atomic_rename)",
            declared.atomic_rename,
            "Capabilities::atomic_rename",
        ),
        (
            "file sync_all (durable_sync)",
            declared.durable_sync,
            "Capabilities::durable_sync",
        ),
        (
            "thread::spawn (threads)",
            declared.threads,
            "Capabilities::threads",
        ),
    ] {
        let measured = match findings.iter().find(|f| f.name == probe) {
            Some(finding) => finding.outcome.works(),
            None => return Err(format!("no host probe named {probe}")),
        };
        if measured != claimed {
            wrong.push(format!(
                "{flag} says {claimed} but the host probe measured {measured}"
            ));
        }
    }

    if wrong.is_empty() {
        Ok(())
    } else {
        Err(wrong.join("; "))
    }
}

/// Render the declared capabilities for the report.
pub fn describe(declared: Capabilities) -> String {
    format!(
        "hard_link={} sync_dir={} atomic_rename={} file_lock={} threads={} durable_sync={}",
        declared.hard_link,
        declared.sync_dir,
        declared.atomic_rename,
        declared.file_lock,
        declared.threads,
        declared.durable_sync,
    )
}
