#![cfg(all(target_arch = "wasm32", target_os = "unknown"))]

//! The main-thread half of the OPFS contract.
//!
//! Browsers refuse `createSyncAccessHandle` outside a Worker. lark must
//! say so plainly at mount rather than hanging, blocking, or opening a
//! database whose writes cannot land, so those are the assertions here.
//!
//! Run against a real browser (`geckodriver` or `chromedriver` on PATH,
//! headless unless `NO_HEADLESS=1`):
//!
//! ```text
//! CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
//!   cargo test --target wasm32-unknown-unknown --test wasm_opfs_main
//! ```
//!
//! `wasm-pack test` cannot be used here: it appends `--tests` to the
//! cargo invocation, which builds every test target in the package, and
//! the rest of `tests/` is native-only.

use lark_kv::env::Env;
use lark_kv::env::opfs::{OpfsEnv, OpfsError, OpfsMode, OpfsOptions};
use lark_kv::{Db, Options};
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn demanding_sync_access_handles_off_a_worker_fails_at_mount() {
    let options = OpfsOptions {
        force_mode: Some(OpfsMode::Sah),
        ..OpfsOptions::default()
    };
    let error = OpfsEnv::mount("lark-test-main-forced", options)
        .await
        .expect_err("sync access handles must not be available on the main thread");

    assert!(
        matches!(error, OpfsError::SyncHandlesUnavailable(_)),
        "expected SyncHandlesUnavailable, got {error:?}"
    );
    assert!(
        error.to_string().contains("Web Worker"),
        "the error must name the requirement, got: {error}"
    );
}

#[wasm_bindgen_test]
async fn the_probe_falls_back_to_mirror_on_the_main_thread() {
    let env = OpfsEnv::mount("lark-test-main-probe", OpfsOptions::default())
        .await
        .expect("mount");
    assert_eq!(env.mode(), OpfsMode::Mirror);

    let caps = env.capabilities();
    assert!(
        !caps.durable_sync,
        "mirror mode is durable only across persist and must say so"
    );
    assert!(!caps.sync_dir);
    assert!(!caps.file_lock);
    assert!(!caps.threads);
}

#[wasm_bindgen_test]
async fn a_database_opened_on_the_main_thread_round_trips_through_persist() {
    let name = "lark-test-main-lifecycle";
    {
        let env = OpfsEnv::mount(name, OpfsOptions::default())
            .await
            .expect("mount");
        let mut options = Options::embedded();
        options.env = env.as_env();

        let db = Db::open(env.db_path(), options).expect("open");
        db.put(b"main", b"thread").expect("put");
        db.flush().expect("flush");
        db.close().expect("close");

        env.persist().await.expect("persist");
        assert_eq!(env.pending_bytes(), 0);
    }

    let env = OpfsEnv::mount(name, OpfsOptions::default())
        .await
        .expect("remount");
    let mut options = Options::embedded();
    options.env = env.as_env();
    let db = Db::open(env.db_path(), options).expect("reopen");
    assert_eq!(
        db.get(b"main").expect("get").as_deref(),
        Some(&b"thread"[..])
    );
    db.close().expect("close");
}

#[wasm_bindgen_test]
async fn a_database_larger_than_the_mirror_bound_is_refused_at_mount() {
    let name = "lark-test-main-residency";
    {
        let env = OpfsEnv::mount(name, OpfsOptions::default())
            .await
            .expect("mount");
        let mut options = Options::embedded();
        options.env = env.as_env();
        let db = Db::open(env.db_path(), options).expect("open");
        for i in 0..64u32 {
            db.put(format!("k{i:04}").as_bytes(), &[b'v'; 512])
                .expect("put");
        }
        db.flush().expect("flush");
        db.close().expect("close");
        env.persist().await.expect("persist");
    }

    let tiny = OpfsOptions {
        max_resident_bytes: 1024,
        ..OpfsOptions::default()
    };
    let error = OpfsEnv::mount(name, tiny)
        .await
        .expect_err("a database over the bound must be refused, not silently loaded");
    assert!(
        matches!(error, OpfsError::ResidencyExceeded { .. }),
        "expected ResidencyExceeded, got {error:?}"
    );
    assert!(
        error.to_string().contains("worker"),
        "the error must name the way out, got: {error}"
    );
}

#[wasm_bindgen_test]
async fn mirror_mode_refuses_immediate_durability() {
    // Mirror mode batches everything until `persist`, so its
    // `sync_all` returns `Ok(())` without making anything durable.
    // Accepting `DurabilityMode::Immediate` there would report a
    // guarantee the backend cannot keep, so `Options::validate`
    // refuses it at open rather than silently downgrading it.
    let env = OpfsEnv::mount("lark-test-main-durability", OpfsOptions::default())
        .await
        .expect("mount falls back to mirror mode on the main thread");
    assert_eq!(env.mode(), OpfsMode::Mirror);
    assert!(
        !env.capabilities().durable_sync,
        "precondition: mirror mode does not provide durable sync"
    );

    let options = Options {
        env: std::sync::Arc::new(env.clone()),
        durability: lark_kv::DurabilityMode::Immediate,
        max_background_compactions: 0,
        ..Options::default()
    };
    let error = Db::open("/lark-test-main-durability", options)
        .err()
        .expect("Immediate durability must be refused on a non-durable env");
    let text = error.to_string();
    assert!(
        text.contains("durable_sync"),
        "the refusal must name the capability, got: {text}"
    );
}
