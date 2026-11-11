// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// ARQVIO Tauri desktop shell -- engine lifecycle (ARQVIO_V4.md section 12.2).
//
// The bundled engine runtime (scripts/build_runtime.py: CPython shipped as engine-runtime/arqvio-engine.exe, the engine
// source in engine-runtime/app) runs as a DETACHED process, so a campaign can keep running after the window closes:
//   * start: an engine already serving 127.0.0.1:8741 that answers /healthz as this app and this version is
//     reattached (the campaign it runs carries on); anything else on the port is an orphan -- cleared, then a fresh
//     engine is started;
//   * close: the window does not close on its own; the UI asks (Stop it now / Keep running in background / Cancel)
//     and calls app_exit. A second close click within 5 s while the UI has not answered exits and leaves the engine
//     running -- closing never hangs on a frozen page;
//   * Ctrl+Shift held while ARQVIO starts opens the interface chooser (boot_info.ctrl_shift).

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Command as StdCommand, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager, WindowEvent};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

const ENGINE_PORT: u16 = 8741;
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Default, serde::Serialize)]
pub struct BootInfo {
    /// "attached" (a running engine was reattached), "started", "external", "failed", or "" while it is decided
    pub engine: String,
    pub detail: String,
    pub ctrl_shift: bool,
}

#[derive(Clone, Default)]
pub struct EngineState {
    pub boot: Arc<Mutex<BootInfo>>,
    pub last_close_request: Arc<Mutex<Option<Instant>>>,
}

/// External-engine mode: with ARQVIO_EXTERNAL_ENGINE=1 the shell starts no engine and kills no process; the UI talks
/// to the engine already serving 127.0.0.1:8741 (a source checkout's engine, for development).
pub fn external_engine() -> bool {
    std::env::var("ARQVIO_EXTERNAL_ENGINE").map(|v| v == "1").unwrap_or(false)
}

#[cfg(windows)]
#[link(name = "user32")]
extern "system" {
    fn GetAsyncKeyState(vkey: i32) -> i16;
}

/// Ctrl and Shift both held right now (read once, at startup).
fn ctrl_shift_held() -> bool {
    #[cfg(windows)]
    unsafe {
        let down = |k: i32| (GetAsyncKeyState(k) as u16 & 0x8000) != 0;
        return down(0x11) && down(0x10); // VK_CONTROL, VK_SHIFT
    }
    #[allow(unreachable_code)]
    false
}

fn arqvio_home() -> PathBuf {
    if let Ok(h) = std::env::var("ARQVIO_HOME") {
        return PathBuf::from(h);
    }
    let base = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).unwrap_or_else(|_| String::from("."));
    PathBuf::from(base).join("ARQVIO")
}

fn pid_file_path() -> PathBuf {
    arqvio_home().join("engine.pid")
}

/// The bundled runtime: (interpreter, runtime root, engine source root), or why it is not there.
fn runtime_paths(app: &AppHandle) -> Result<(PathBuf, PathBuf, PathBuf), String> {
    let resources = app.path().resource_dir().map_err(|e| format!("no resource directory: {}", e))?;
    let root = resources.join("engine-runtime");
    let exe = root.join("arqvio-engine.exe");
    let src = root.join("app");
    if !exe.is_file() || !src.join("engine").join("__main__.py").is_file() {
        return Err(format!("the engine runtime is missing from this install ({})", root.display()));
    }
    Ok((exe, root, src))
}

/// GET /healthz on the loopback port: Some(body) when something answered HTTP 200.
fn healthz() -> Option<serde_json::Value> {
    let addr = SocketAddr::from(([127, 0, 0, 1], ENGINE_PORT));
    let mut s = TcpStream::connect_timeout(&addr, Duration::from_millis(800)).ok()?;
    let _ = s.set_read_timeout(Some(Duration::from_secs(10)));
    s.write_all(b"GET /healthz HTTP/1.0\r\nHost: 127.0.0.1\r\nAccept: application/json\r\n\r\n").ok()?;
    let mut raw = String::new();
    s.read_to_string(&mut raw).ok()?;
    let (head, body) = raw.split_once("\r\n\r\n")?;
    if !head.starts_with("HTTP/1.1 200") && !head.starts_with("HTTP/1.0 200") {
        return None;
    }
    serde_json::from_str(body).ok()
}

fn port_open() -> bool {
    TcpStream::connect_timeout(&SocketAddr::from(([127, 0, 0, 1], ENGINE_PORT)), Duration::from_millis(800)).is_ok()
}

/// This app's engine, this version: safe to reattach to.
fn ours(h: &serde_json::Value) -> bool {
    h.get("app").and_then(|v| v.as_str()) == Some("arqvio")
        && h.get("version").and_then(|v| v.as_str()) == Some(APP_VERSION)
}

fn kill_orphans() {
    #[cfg(windows)]
    {
        // An engine left by an older install (or a crash) holds port 8741; its campaign and worker processes run the
        // same interpreter (arqvio-engine.exe, arqvio-enginew.exe). Clear them by image name, or the new engine dies
        // with Errno 10048 (address in use).
        let _ = StdCommand::new("taskkill").args(["/F", "/T", "/IM", "arqvio-engine*"]).creation_flags(0x0800_0000).output();
        std::thread::sleep(Duration::from_millis(1000));
    }
    let _ = std::fs::remove_file(pid_file_path());
}

/// Start the engine detached from this process: it outlives the window when the owner keeps a campaign running.
fn spawn_detached(app: &AppHandle) -> Result<u32, String> {
    let (exe, root, src) = runtime_paths(app)?;
    let logs = arqvio_home().join("logs");
    let _ = std::fs::create_dir_all(&logs);
    let out = std::fs::File::create(logs.join("engine.stdout.log")).map_err(|e| e.to_string())?;
    let err = std::fs::File::create(logs.join("engine.stderr.log")).map_err(|e| e.to_string())?;
    let mut cmd = StdCommand::new(&exe);
    cmd.args(["-m", "engine"])
        .current_dir(&src)
        .env("PYTHONHOME", &root)
        .env("PYTHONPATH", &src)
        .env("PYTHONNOUSERSITE", "1")
        .env("ARQVIO_APP", "1") // the app starts empty: only the owner's own markets, never the repo registry
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err);
    #[cfg(windows)]
    cmd.creation_flags(0x0000_0008 | 0x0000_0200); // DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    let child = cmd.spawn().map_err(|e| format!("the engine runtime did not start: {}", e))?;
    let pid = child.id();
    let path = pid_file_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, pid.to_string());
    Ok(pid)
}

/// Stop the engine and everything it started (campaign process, power-mode workers).
fn stop_engine() {
    if external_engine() {
        return;
    }
    #[cfg(windows)]
    {
        if let Ok(pid) = std::fs::read_to_string(pid_file_path()) {
            let pid = pid.trim().to_string();
            if !pid.is_empty() {
                let _ = StdCommand::new("taskkill").args(["/F", "/T", "/PID", &pid]).creation_flags(0x0800_0000).output();
            }
        }
    }
    let _ = std::fs::remove_file(pid_file_path());
}

fn set_boot(state: &EngineState, engine: &str, detail: String) {
    if let Ok(mut b) = state.boot.lock() {
        b.engine = engine.to_string();
        b.detail = detail;
    }
}

/// Reattach or start, off the UI thread; the boot screen polls /api/v2/app/boot meanwhile.
fn bring_up_engine(app: AppHandle, state: EngineState) {
    std::thread::spawn(move || {
        if external_engine() {
            set_boot(&state, "external", "ARQVIO_EXTERNAL_ENGINE=1: using the engine on 127.0.0.1:8741".into());
            let _ = app.emit("engine-state", "external");
            return;
        }
        // An engine busy with its first MetaTrader 5 probe can take seconds to answer: ask again before deciding,
        // and never clear a port whose program has not said what it is (it may be our engine mid-campaign).
        let mut answer = healthz();
        for _ in 0..2 {
            if answer.is_some() || !port_open() {
                break;
            }
            answer = healthz();
        }
        if answer.is_none() && port_open() {
            let why = format!("port {} is in use by a program that does not answer as ARQVIO; close it, then reopen ARQVIO", ENGINE_PORT);
            set_boot(&state, "failed", why.clone());
            let _ = app.emit("engine-failed", why);
            return;
        }
        match answer {
            Some(h) if ours(&h) => {
                log::info!("reattached to the running engine");
                set_boot(&state, "attached", format!("reattached to the running ARQVIO engine {}", APP_VERSION));
                let _ = app.emit("engine-state", "attached");
                return;
            }
            Some(h) => log::warn!("port {} answers as {:?}: an orphan, cleared", ENGINE_PORT, h.get("version")),
            None => {}
        }
        kill_orphans();
        match spawn_detached(&app) {
            Ok(pid) => {
                set_boot(&state, "started", format!("engine started (pid {})", pid));
                let _ = app.emit("engine-state", "started");
            }
            Err(e) => {
                log::warn!("{}", e);
                set_boot(&state, "failed", e.clone());
                let _ = app.emit("engine-failed", e);
            }
        }
    });
}

#[tauri::command]
fn get_engine_status() -> bool {
    healthz().map(|h| ours(&h)).unwrap_or(false)
}

#[tauri::command]
fn boot_info(state: tauri::State<'_, EngineState>) -> BootInfo {
    state.boot.lock().map(|b| b.clone()).unwrap_or_default()
}

/// The close dialog's answer: stop the engine (and its campaign) or leave it running, then quit.
#[tauri::command]
fn app_exit(app: AppHandle, stop: bool) {
    if stop {
        stop_engine();
    }
    app.exit(0);
}

fn main() {
    let engine_state = EngineState::default();
    if let Ok(mut b) = engine_state.boot.lock() {
        b.ctrl_shift = ctrl_shift_held();
    }
    let close_state = engine_state.clone();

    // The engine is detached on purpose: exiting the app never stops it implicitly (app_exit(stop) does).
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(engine_state.clone())
        .invoke_handler(tauri::generate_handler![get_engine_status, boot_info, app_exit])
        .setup(move |app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(tauri_plugin_log::Builder::default().level(log::LevelFilter::Info).build())?;
            }
            bring_up_engine(app.handle().clone(), engine_state.clone());
            if let Some(main_window) = app.get_webview_window("main") {
                let w = main_window.clone();
                let handle = app.handle().clone();
                main_window.on_window_event(move |event| {
                    if let WindowEvent::CloseRequested { api, .. } = event {
                        let mut last = close_state.last_close_request.lock().unwrap();
                        if last.map(|t| t.elapsed() < Duration::from_secs(5)).unwrap_or(false) {
                            handle.exit(0); // the page did not answer the first click: quit, the engine keeps running
                            return;
                        }
                        *last = Some(Instant::now());
                        api.prevent_close();
                        let _ = w.emit("close-requested", ());
                    }
                });
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
