/// Top-level application context. Lightweight — no HTTP clients, no auth.
/// Wired together in main.rs and passed into the Repl.
pub struct AppContext {
    pub run_history_dir: std::path::PathBuf,
    pub db_dir: std::path::PathBuf,
}

impl AppContext {
    pub fn new() -> Self {
        let base = dirs::home_dir().unwrap_or_default().join(".mlc");
        let run_history_dir = base.join("runs");
        let db_dir = base.join("db");
        std::fs::create_dir_all(&run_history_dir).ok();
        std::fs::create_dir_all(&db_dir).ok();
        Self { run_history_dir, db_dir }
    }
}
