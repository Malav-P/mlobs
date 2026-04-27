mod app;
mod repl;
mod runner;
mod analysis;

use anyhow::Result;
use clap::{Parser, Subcommand};
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
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run and monitor a training command.
    Run {
        /// Command and arguments to execute (e.g. `mlc run python train.py --lr 0.01`).
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, required = true)]
        cmd: Vec<String>,

        /// Working directory for the command (defaults to current directory).
        #[arg(long)]
        cwd: Option<String>,
    },
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

    let initial_run = match cli.command {
        Some(Commands::Run { cmd, cwd }) => Some((cmd.join(" "), cwd)),
        None => None,
    };

    let app = AppContext::new();
    let repl = Repl::new(app, initial_run);
    repl.run().await?;

    Ok(())
}
