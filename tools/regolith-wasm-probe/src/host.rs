//! Host-primitive probes.
//!
//! Every capability regolith's `Env` reports for a target is measured here
//! rather than assumed from a `cfg`. `cfg!(unix)` is false on
//! `wasm32-wasip1` even though almost all of POSIX works there, and a
//! directory fsync that returns `EBADF` is exactly the case where a
//! `cfg`-derived answer would claim a durability the host does not
//! provide.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Duration;

/// Outcome of one host probe.
pub enum Outcome {
    /// The primitive worked.
    Works,
    /// The primitive failed, with the host's own message.
    Fails(String),
    /// The primitive was not exercised, because calling it on this
    /// target aborts the module rather than returning an error.
    #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
    Skipped(&'static str),
}

impl Outcome {
    fn from(result: std::io::Result<()>) -> Self {
        match result {
            Ok(()) => Outcome::Works,
            Err(e) => Outcome::Fails(format!("{e}")),
        }
    }

    /// Whether the primitive is usable on this host.
    pub fn works(&self) -> bool {
        matches!(self, Outcome::Works)
    }

    /// Render as a fixed-width verdict for the report table.
    pub fn verdict(&self) -> String {
        match self {
            Outcome::Works => "works".to_string(),
            Outcome::Fails(msg) => format!("FAILS: {msg}"),
            #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
            Outcome::Skipped(why) => format!("skipped: {why}"),
        }
    }
}

/// One probed primitive and what the host did with it.
pub struct Finding {
    /// Primitive name, matching the `Capabilities` field it feeds.
    pub name: &'static str,
    /// What happened.
    pub outcome: Outcome,
}

/// Probe every host primitive regolith's `Env` depends on, under `root`.
///
/// `root` must already exist and be writable. Nothing here is fatal:
/// the caller decides which failures matter.
pub fn probe(root: &Path) -> std::io::Result<Vec<Finding>> {
    fs::create_dir_all(root)?;
    let mut out = Vec::new();

    out.push(Finding {
        name: "create_dir_all",
        outcome: Outcome::from(fs::create_dir_all(root.join("nested/deep"))),
    });

    let data = root.join("probe.bin");
    out.push(Finding {
        name: "write + read",
        outcome: Outcome::from(write_then_read(&data)),
    });
    out.push(Finding {
        name: "file sync_all (durable_sync)",
        outcome: Outcome::from(file_sync(&data)),
    });
    out.push(Finding {
        name: "directory fsync (sync_dir)",
        outcome: Outcome::from(dir_sync(root)),
    });
    out.push(Finding {
        name: "hard_link",
        outcome: Outcome::from(hard_link(root, &data)),
    });
    out.push(Finding {
        name: "rename over existing (atomic_rename)",
        outcome: Outcome::from(rename_over(root)),
    });
    out.push(Finding {
        name: "read_dir",
        outcome: Outcome::from(read_dir(root)),
    });
    out.push(Finding {
        name: "set_len (truncate)",
        outcome: Outcome::from(truncate(&data)),
    });
    out.push(Finding {
        name: "append after reopen",
        outcome: Outcome::from(append_after_reopen(root)),
    });
    out.push(Finding {
        name: "seek from end",
        outcome: Outcome::from(seek_from_end(&data)),
    });
    out.push(Finding {
        name: "create_new exclusivity (file_lock proxy)",
        outcome: Outcome::from(create_new_twice(root)),
    });
    out.push(Finding {
        name: "thread::spawn (threads)",
        outcome: Outcome::from(spawn()),
    });
    out.push(Finding {
        name: "thread::sleep",
        outcome: Outcome::from(sleep()),
    });
    out.push(Finding {
        name: "Instant::now (monotonic clock)",
        outcome: monotonic_clock(),
    });
    out.push(Finding {
        name: "SystemTime::now (wall clock)",
        outcome: wall_clock(),
    });

    let _ = fs::remove_dir_all(root.join("nested"));
    Ok(out)
}

fn write_then_read(path: &Path) -> std::io::Result<()> {
    fs::write(path, b"regolith-probe-payload")?;
    let back = fs::read(path)?;
    if back != b"regolith-probe-payload" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "read back different bytes than were written",
        ));
    }
    Ok(())
}

fn file_sync(path: &Path) -> std::io::Result<()> {
    let mut f = OpenOptions::new().write(true).open(path)?;
    f.write_all(b"!")?;
    f.sync_all()
}

fn dir_sync(dir: &Path) -> std::io::Result<()> {
    File::open(dir)?.sync_all()
}

fn hard_link(root: &Path, src: &Path) -> std::io::Result<()> {
    let dst = root.join("probe.link");
    let _ = fs::remove_file(&dst);
    fs::hard_link(src, &dst)?;
    let linked = fs::read(&dst)?;
    let original = fs::read(src)?;
    fs::remove_file(&dst)?;
    if linked != original {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "hard link does not observe the source contents",
        ));
    }
    Ok(())
}

fn rename_over(root: &Path) -> std::io::Result<()> {
    let from = root.join("rename.src");
    let to = root.join("rename.dst");
    fs::write(&from, b"new")?;
    fs::write(&to, b"old")?;
    fs::rename(&from, &to)?;
    let back = fs::read(&to)?;
    fs::remove_file(&to)?;
    if back != b"new" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "rename did not replace the destination contents",
        ));
    }
    Ok(())
}

fn read_dir(root: &Path) -> std::io::Result<()> {
    let mut seen = 0usize;
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let _ = entry.file_type()?;
        seen += 1;
    }
    if seen == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "read_dir returned no entries for a directory known to hold files",
        ));
    }
    Ok(())
}

fn truncate(path: &Path) -> std::io::Result<()> {
    let f = OpenOptions::new().write(true).open(path)?;
    f.set_len(4)?;
    let len = f.metadata()?.len();
    if len != 4 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("set_len(4) left a file of {len} bytes"),
        ));
    }
    Ok(())
}

fn append_after_reopen(root: &Path) -> std::io::Result<()> {
    let path = root.join("append.bin");
    let _ = fs::remove_file(&path);
    fs::write(&path, b"aaa")?;
    let mut f = OpenOptions::new().append(true).open(&path)?;
    f.write_all(b"bbb")?;
    f.sync_all()?;
    drop(f);
    let back = fs::read(&path)?;
    fs::remove_file(&path)?;
    if back != b"aaabbb" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "append after reopen did not land at the end of the file",
        ));
    }
    Ok(())
}

fn seek_from_end(path: &Path) -> std::io::Result<()> {
    let mut f = File::open(path)?;
    f.seek(SeekFrom::End(-2))?;
    let mut buf = [0u8; 2];
    f.read_exact(&mut buf)
}

fn create_new_twice(root: &Path) -> std::io::Result<()> {
    let path = root.join("exclusive.lock");
    let _ = fs::remove_file(&path);
    let _first = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    let second = OpenOptions::new().write(true).create_new(true).open(&path);
    let _ = fs::remove_file(&path);
    match second {
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Ok(_) => Err(std::io::Error::other(
            "create_new succeeded twice on the same path",
        )),
        Err(e) => Err(e),
    }
}

fn spawn() -> std::io::Result<()> {
    let handle = std::thread::Builder::new()
        .name("regolith-probe".to_string())
        .spawn(|| {})?;
    handle
        .join()
        .map_err(|_| std::io::Error::other("spawned thread panicked"))
}

fn sleep() -> std::io::Result<()> {
    std::thread::sleep(Duration::from_millis(1));
    Ok(())
}

#[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
fn monotonic_clock() -> Outcome {
    let start = std::time::Instant::now();
    let _ = start.elapsed();
    Outcome::Works
}

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
fn monotonic_clock() -> Outcome {
    Outcome::Skipped("Instant::now aborts the module on this target")
}

#[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
fn wall_clock() -> Outcome {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) if d.as_secs() > 0 => Outcome::Works,
        Ok(_) => Outcome::Fails("wall clock reported the Unix epoch".to_string()),
        Err(e) => Outcome::Fails(format!("{e}")),
    }
}

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
fn wall_clock() -> Outcome {
    Outcome::Skipped("SystemTime::now aborts the module on this target")
}
