mod app;
mod repl;
mod runner;
mod analysis;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

use app::AppContext;
use repl::Repl;

#[derive(Parser, Debug)]
#[command(
    name = "mlc",
    about = "mlc — ML training monitor",
    version,
    disable_help_subcommand = true,
)]
struct Cli {
    /// Command to run and monitor (e.g. `mlc python train.py`).
    /// Omit to open the interactive REPL.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    cmd: Vec<String>,

    /// Working directory for the command (defaults to current directory).
    #[arg(long)]
    cwd: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Logging to ~/.mlc/logs/cli.log
    let log_dir = dirs::home_dir().unwrap_or_default().join(".mlc").join("logs");
    std::fs::create_dir_all(&log_dir).ok();
    let file_appender = tracing_appender::rolling::never(&log_dir, "cli.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .with_writer(non_blocking)
        .with_ansi(false)
        .init();

    // SIGTERM → graceful shutdown
    let _ = ctrlc::set_handler(|| {
        let _ = crossterm::terminal::disable_raw_mode();
        std::process::exit(0);
    });

    let initial_run = if cli.cmd.is_empty() {
        None
    } else {
        Some((cli.cmd.join(" "), cli.cwd))
    };

    let app = AppContext::new();
    let repl = Repl::new(app, initial_run);
    repl.run().await?;

    Ok(())
}
