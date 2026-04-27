pub mod commands;

use std::collections::VecDeque;
use std::io::stdout;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use crossterm::{
    event::{Event, EventStream, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures_util::StreamExt;
use indexmap::IndexMap;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph},
    Frame, Terminal,
};
use tokio::sync::mpsc;

use crate::analysis::{AlertLevel, AnalysisClient, InsightReport};
use crate::app::AppContext;
use crate::runner::{
    classify, history::{list_runs, new_run_id, save_run, RunRecord}, LineClass, MetricEvent,
    RunnerEvent, SubprocessRunner,
};
use crate::runner::extractor::RateGate;
use commands::build_registry;

const HISTORY_FILE: &str = ".mlc/history";
const MAX_HISTORY: usize = 500;
const MAX_STDOUT_LINES: usize = 5000;
const MAX_ALERT_FEED: usize = 200;

// ---------------------------------------------------------------------------
// Alert types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Alert {
    pub level: AlertLevel,
    pub signal: String,
    pub metric: String,
    pub description: String,
    pub confidence: f64,
    pub suggestions: Vec<String>,
    pub received_at: Instant,
}

impl Alert {
    fn prefix(&self) -> &'static str {
        match self.level {
            AlertLevel::Critical => "✖ CRITICAL",
            AlertLevel::Warning  => "⚠ WARNING",
            AlertLevel::Info     => "  INFO",
        }
    }

    fn color(&self) -> Color {
        match self.level {
            AlertLevel::Critical => Color::Red,
            AlertLevel::Warning  => Color::Yellow,
            AlertLevel::Info     => Color::DarkGray,
        }
    }
}

#[derive(Clone)]
struct CriticalState {
    alert: Alert,
    cursor: usize,       // 0=stop, 1=continue, 2=details
    detail_open: bool,
    raw_report: String,
}

// ---------------------------------------------------------------------------
// Run view state — everything the TUI needs during an active run
// ---------------------------------------------------------------------------

#[derive(Default)]
struct RunViewState {
    stdout_lines: VecDeque<String>,
    stdout_style: VecDeque<StdoutLineStyle>,
    stdout_scroll: usize,
    metrics: IndexMap<String, Vec<(f64, f64)>>,
    selected_metric: Option<String>,
    metric_index: usize,
    alert_feed: VecDeque<Alert>,
    alert_scroll: usize,
    active_critical: Option<CriticalState>,
    status: RunStatus,
    new_points_since_inspect: u32,
    // Which pane has keyboard focus: true = stdout, false = alert feed
    focus_stdout: bool,
    // True once the subprocess exits — user stays on run view until Esc
    finished: bool,
}

#[derive(Clone, Copy, Debug, Default)]
enum StdoutLineStyle {
    #[default]
    Plain,
    Error,
    Metric,
}

#[derive(Default)]
struct RunStatus {
    command: String,
    run_id: String,
    start_time: Option<Instant>,
    epoch: Option<u64>,
    health: HealthSignal,
    exit_code: Option<i32>,
}

#[derive(Clone, Default, Debug)]
enum HealthSignal {
    #[default]
    Unknown,
    Healthy,
    Warning(String),
    Critical(String),
}

impl RunViewState {
    fn push_stdout(&mut self, line: String, style: StdoutLineStyle) {
        if self.stdout_lines.len() >= MAX_STDOUT_LINES {
            self.stdout_lines.pop_front();
            self.stdout_style.pop_front();
        }
        self.stdout_lines.push_back(line);
        self.stdout_style.push_back(style);
    }

    fn push_alert(&mut self, alert: Alert) {
        if self.alert_feed.len() >= MAX_ALERT_FEED {
            self.alert_feed.pop_front();
        }
        self.alert_feed.push_back(alert);
    }
}

// ---------------------------------------------------------------------------
// App mode
// ---------------------------------------------------------------------------

/// Whether the user is composing a /run command from the home screen.
#[derive(Default, PartialEq)]
enum HomeInputMode {
    #[default]
    None,
    Composing,
}

#[derive(Default, PartialEq)]
enum AppMode {
    #[default]
    Idle,
    Running,
}

// ---------------------------------------------------------------------------
// Simple line editor (preserved from original)
// ---------------------------------------------------------------------------

struct LineEditor {
    buf: Vec<char>,
    cursor: usize,
    history: Vec<String>,
    history_pos: Option<usize>,
    saved_buf: String,
}

impl LineEditor {
    fn new(history: Vec<String>) -> Self {
        Self { buf: Vec::new(), cursor: 0, history, history_pos: None, saved_buf: String::new() }
    }

    fn text(&self) -> String { self.buf.iter().collect() }

    fn insert(&mut self, c: char) {
        self.buf.insert(self.cursor, c);
        self.cursor += 1;
        self.history_pos = None;
    }

    fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.buf.remove(self.cursor);
            self.history_pos = None;
        }
    }

    fn delete(&mut self) {
        if self.cursor < self.buf.len() {
            self.buf.remove(self.cursor);
        }
    }

    fn left(&mut self)  { if self.cursor > 0 { self.cursor -= 1; } }
    fn right(&mut self) { if self.cursor < self.buf.len() { self.cursor += 1; } }
    fn home(&mut self)  { self.cursor = 0; }
    fn end(&mut self)   { self.cursor = self.buf.len(); }

    fn history_prev(&mut self) {
        if self.history.is_empty() { return; }
        let new_pos = match self.history_pos {
            None => { self.saved_buf = self.text(); self.history.len() - 1 }
            Some(0) => return,
            Some(p) => p - 1,
        };
        self.history_pos = Some(new_pos);
        let entry: Vec<char> = self.history[new_pos].chars().collect();
        self.cursor = entry.len();
        self.buf = entry;
    }

    fn history_next(&mut self) {
        match self.history_pos {
            None => {}
            Some(p) if p + 1 >= self.history.len() => {
                self.history_pos = None;
                let saved: Vec<char> = self.saved_buf.chars().collect();
                self.cursor = saved.len();
                self.buf = saved;
            }
            Some(p) => {
                self.history_pos = Some(p + 1);
                let entry: Vec<char> = self.history[p + 1].chars().collect();
                self.cursor = entry.len();
                self.buf = entry;
            }
        }
    }

    fn commit(&mut self) -> String {
        let text = self.text();
        if !text.trim().is_empty() {
            if self.history.last().map(|s| s.as_str()) != Some(&text) {
                self.history.push(text.clone());
                if self.history.len() > MAX_HISTORY {
                    self.history.remove(0);
                }
            }
        }
        self.buf.clear();
        self.cursor = 0;
        self.history_pos = None;
        self.saved_buf.clear();
        text
    }
}

// ---------------------------------------------------------------------------
// REPL
// ---------------------------------------------------------------------------

pub struct Repl {
    app: AppContext,
    initial_run: Option<(String, Option<String>)>,
}

impl Repl {
    pub fn new(app: AppContext, initial_run: Option<(String, Option<String>)>) -> Self {
        Self { app, initial_run }
    }

    pub async fn run(self) -> Result<()> {
        let cmd_history = load_history();
        let mut editor = LineEditor::new(cmd_history);
        let _registry = build_registry();

        // Terminal setup
        let mut out = stdout();
        execute!(out, EnterAlternateScreen)?;
        terminal::enable_raw_mode()?;
        let backend = CrosstermBackend::new(out);
        let mut terminal = Terminal::new(backend)?;
        terminal.hide_cursor()?;

        // Tickers
        let mut draw_tick = tokio::time::interval(Duration::from_millis(16));
        draw_tick.tick().await;
        let mut insights_tick = tokio::time::interval(Duration::from_secs(5));
        insights_tick.tick().await;

        let mut event_stream = EventStream::new();

        // Run state
        let mut app_mode = AppMode::Idle;
        let mut run_view = RunViewState::default();
        run_view.focus_stdout = true;

        // Runner channel (None when idle)
        let mut runner_rx: Option<mpsc::Receiver<RunnerEvent>> = None;
        let mut runner: Option<SubprocessRunner> = None;
        let mut analysis_client: Option<AnalysisClient> = None;
        let mut analysis_proc: Option<tokio::process::Child> = None;
        let mut rate_gate = RateGate::new();

        // Home screen state
        let mut home_input = HomeInputMode::None;
        let mut recent_runs: Vec<RunRecord> =
            list_runs(&self.app.run_history_dir, 8).unwrap_or_default();

        // If launched with `mlc run <cmd>`, auto-start
        if let Some((cmd, cwd)) = self.initial_run {
            let (client, proc, rx, sub) = start_run(
                &cmd,
                cwd.as_deref(),
                &self.app,
                &mut run_view,
            ).await;
            analysis_client = Some(client);
            analysis_proc = Some(proc);
            runner_rx = Some(rx);
            runner = Some(sub);
            app_mode = AppMode::Running;
        }

        'main: loop {
            tokio::select! {
                biased;

                // ---- Draw tick ----
                _ = draw_tick.tick() => {
                    terminal.draw(|f| {
                        render(f, &app_mode, &run_view, &editor, &recent_runs, &home_input);
                    })?;
                }

                // ---- Insights poll (only during run) ----
                _ = insights_tick.tick(), if app_mode == AppMode::Running => {
                    if let Some(ref client) = analysis_client {
                        if let Ok(report) = client.get_insights().await {
                            apply_insight_report(&mut run_view, report);
                        }
                    }
                }

                // ---- Runner events (only during run) ----
                Some(ev) = async {
                    match &mut runner_rx {
                        Some(rx) => rx.recv().await,
                        None => std::future::pending().await,
                    }
                } => {
                    match ev {
                        RunnerEvent::StdoutLine(line) => {
                            let class = classify(&line, false);
                            match &class {
                                LineClass::Metric(metrics) => {
                                    run_view.push_stdout(line.clone(), StdoutLineStyle::Metric);
                                    if let Some(ref mut client) = analysis_client {
                                        for m in metrics {
                                            if rate_gate.should_pass(&m.key) {
                                                let step = m.step.or_else(|| {
                                                    run_view.metrics.get(&m.key)
                                                        .map(|v| v.len() as u64)
                                                });
                                                let ts = SystemTime::now()
                                                    .duration_since(UNIX_EPOCH)
                                                    .unwrap_or_default()
                                                    .as_secs_f64();
                                                client.send_event(MetricEvent {
                                                    run_id: run_view.status.run_id.clone(),
                                                    key: m.key.clone(),
                                                    value: m.value,
                                                    step,
                                                    ts,
                                                }).await;
                                                // Also update local chart data
                                                let s = step.unwrap_or_else(|| {
                                                    run_view.metrics.get(&m.key)
                                                        .map(|v| v.len() as u64)
                                                        .unwrap_or(0)
                                                });
                                                run_view.metrics
                                                    .entry(m.key.clone())
                                                    .or_default()
                                                    .push((s as f64, m.value));
                                                // Update epoch from step if it looks like one
                                                if m.key.contains("epoch") {
                                                    run_view.status.epoch = Some(m.value as u64);
                                                }
                                                run_view.new_points_since_inspect += 1;
                                            }
                                        }
                                        // Flush if interval elapsed
                                        client.flush().await;
                                        // Auto-select first metric for chart
                                        if run_view.selected_metric.is_none() {
                                            run_view.selected_metric = run_view.metrics.keys().next().cloned();
                                        }
                                    }
                                }
                                LineClass::Error => {
                                    run_view.push_stdout(line, StdoutLineStyle::Error);
                                }
                                LineClass::Progress => {
                                    // Replace last line (tqdm overwrite)
                                    if !run_view.stdout_lines.is_empty() {
                                        if let Some(last) = run_view.stdout_lines.back_mut() {
                                            *last = line;
                                        }
                                    } else {
                                        run_view.push_stdout(line, StdoutLineStyle::Plain);
                                    }
                                }
                                LineClass::Plain => {
                                    run_view.push_stdout(line, StdoutLineStyle::Plain);
                                }
                                LineClass::Blank => {}
                            }
                        }
                        RunnerEvent::StderrLine(line) => {
                            let class = classify(&line, true);
                            match class {
                                LineClass::Blank => {}
                                _ => run_view.push_stdout(line, StdoutLineStyle::Error),
                            }
                        }
                        RunnerEvent::Exited { code } => {
                            run_view.status.exit_code = Some(code.unwrap_or(-1));
                            let exit_msg = match code {
                                Some(0) => "Process exited successfully.".to_string(),
                                Some(c) => format!("Process exited with code {}.", c),
                                None    => "Process terminated.".to_string(),
                            };
                            run_view.push_stdout(
                                format!("[mlc] {}", exit_msg),
                                StdoutLineStyle::Plain,
                            );
                            // Flush remaining events and fetch final report
                            if let Some(ref mut client) = analysis_client {
                                client.flush().await;
                                let final_report = client.get_insights().await.ok()
                                    .and_then(|r| serde_json::to_value(r).ok());
                                let cwd = std::env::current_dir()
                                    .map(|p| p.display().to_string())
                                    .unwrap_or_default();
                                let db_path = self.app.db_dir
                                    .join(format!("{}.sqlite", run_view.status.run_id))
                                    .display().to_string();
                                let mut record = RunRecord::new(
                                    &run_view.status.run_id,
                                    &run_view.status.command,
                                    &cwd,
                                    &db_path,
                                );
                                record.ended_at = Some(chrono::Utc::now().to_rfc3339());
                                record.exit_code = code;
                                record.final_report = final_report;
                                save_run(&self.app.run_history_dir, &record).ok();
                            }
                            // Kill analysis process
                            if let Some(ref mut proc) = analysis_proc {
                                proc.kill().await.ok();
                            }
                            runner = None;
                            runner_rx = None;
                            analysis_client = None;
                            analysis_proc = None;
                            // Stay on run view — user presses Esc to return home
                            run_view.finished = true;
                            let _ = exit_msg; // already pushed above
                        }
                    }
                }

                // ---- Crossterm events ----
                maybe_ev = event_stream.next() => {
                    let Some(Ok(ev)) = maybe_ev else { break 'main; };

                    match ev {
                        Event::Resize(_, _) => {}

                        Event::Mouse(_) => {}

                        Event::Key(key) => {
                            if key.kind != KeyEventKind::Press { continue; }

                            // ---- Run view keys (active or finished) ----
                            if app_mode == AppMode::Running {
                                // Intervention panel eats keys first
                                if let Some(ref mut crit) = run_view.active_critical {
                                    match (key.code, key.modifiers) {
                                        (KeyCode::Right, _) | (KeyCode::Tab, _) => {
                                            crit.cursor = (crit.cursor + 1) % 3;
                                            continue;
                                        }
                                        (KeyCode::Left, _) => {
                                            crit.cursor = (crit.cursor + 2) % 3;
                                            continue;
                                        }
                                        (KeyCode::Enter, _) => {
                                            let cursor = crit.cursor;
                                            match cursor {
                                                0 => {
                                                    // Stop run
                                                    if let Some(r) = runner.take() { r.kill(); }
                                                }
                                                1 => {
                                                    // Continue — dismiss panel
                                                    run_view.active_critical = None;
                                                }
                                                2 => {
                                                    // Toggle details
                                                    if let Some(ref mut c) = run_view.active_critical {
                                                        c.detail_open = !c.detail_open;
                                                    }
                                                }
                                                _ => {}
                                            }
                                            continue;
                                        }
                                        (KeyCode::Esc, _) => {
                                            run_view.active_critical = None;
                                            continue;
                                        }
                                        _ => {}
                                    }
                                }

                                // Esc when finished and no panel → go home
                                if run_view.finished && key.code == KeyCode::Esc {
                                    recent_runs = list_runs(&self.app.run_history_dir, 8)
                                        .unwrap_or_default();
                                    run_view = RunViewState::default();
                                    run_view.focus_stdout = true;
                                    app_mode = AppMode::Idle;
                                    continue;
                                }

                                // Kill keybind — only while process is still running
                                if !run_view.finished
                                    && key.code == KeyCode::Char('k')
                                    && key.modifiers == KeyModifiers::NONE
                                {
                                    if let Some(r) = runner.take() { r.kill(); }
                                    continue;
                                }

                                // Tab switches focus between stdout and alert feed
                                if key.code == KeyCode::Tab && key.modifiers == KeyModifiers::NONE {
                                    run_view.focus_stdout = !run_view.focus_stdout;
                                    continue;
                                }

                                // Up/Down/PgUp/PgDn scroll the focused pane
                                match key.code {
                                    KeyCode::Up | KeyCode::PageUp => {
                                        let delta = if key.code == KeyCode::PageUp { 5 } else { 1 };
                                        if run_view.focus_stdout {
                                            run_view.stdout_scroll += delta;
                                        } else {
                                            run_view.alert_scroll += delta;
                                        }
                                        continue;
                                    }
                                    KeyCode::Down | KeyCode::PageDown => {
                                        let delta = if key.code == KeyCode::PageDown { 5 } else { 1 };
                                        if run_view.focus_stdout {
                                            run_view.stdout_scroll = run_view.stdout_scroll.saturating_sub(delta);
                                        } else {
                                            run_view.alert_scroll = run_view.alert_scroll.saturating_sub(delta);
                                        }
                                        continue;
                                    }
                                    _ => {}
                                }

                                // Left/Right cycle through detected metrics in the chart
                                let metric_count = run_view.metrics.len();
                                if metric_count > 1 {
                                    match key.code {
                                        KeyCode::Right if key.modifiers == KeyModifiers::NONE => {
                                            run_view.metric_index = (run_view.metric_index + 1) % metric_count;
                                            run_view.selected_metric = run_view.metrics
                                                .get_index(run_view.metric_index)
                                                .map(|(k, _)| k.clone());
                                            continue;
                                        }
                                        KeyCode::Left if key.modifiers == KeyModifiers::NONE => {
                                            run_view.metric_index = (run_view.metric_index + metric_count - 1) % metric_count;
                                            run_view.selected_metric = run_view.metrics
                                                .get_index(run_view.metric_index)
                                                .map(|(k, _)| k.clone());
                                            continue;
                                        }
                                        _ => {}
                                    }
                                }

                                // Consume remaining keys during run (don't fall through to home input)
                                continue;
                            }

                            // ---- Home screen keys (Idle) ----
                            match (key.code, key.modifiers) {
                                (KeyCode::Char('c'), KeyModifiers::CONTROL)
                                | (KeyCode::Char('d'), KeyModifiers::CONTROL) => {
                                    break 'main;
                                }
                                (KeyCode::Char('q'), KeyModifiers::NONE)
                                    if home_input == HomeInputMode::None =>
                                {
                                    break 'main;
                                }
                                (KeyCode::Char('r'), KeyModifiers::NONE)
                                    if home_input == HomeInputMode::None =>
                                {
                                    home_input = HomeInputMode::Composing;
                                }
                                (KeyCode::Esc, _) if home_input == HomeInputMode::Composing => {
                                    editor.buf.clear();
                                    editor.cursor = 0;
                                    home_input = HomeInputMode::None;
                                }
                                (KeyCode::Enter, _) if home_input == HomeInputMode::Composing => {
                                    let text = editor.commit();
                                    home_input = HomeInputMode::None;
                                    let args = text.trim();
                                    if !args.is_empty() {
                                        let cmd = if args.starts_with("/run ") {
                                            args.trim_start_matches("/run").trim().to_string()
                                        } else {
                                            args.to_string()
                                        };
                                        let (client, proc, rx, sub) = start_run(
                                            &cmd,
                                            None,
                                            &self.app,
                                            &mut run_view,
                                        ).await;
                                        analysis_client = Some(client);
                                        analysis_proc = Some(proc);
                                        runner_rx = Some(rx);
                                        runner = Some(sub);
                                        app_mode = AppMode::Running;
                                    }
                                }
                                (KeyCode::Backspace, _)
                                | (KeyCode::Char('h'), KeyModifiers::CONTROL)
                                    if home_input == HomeInputMode::Composing =>
                                { editor.backspace(); }
                                (KeyCode::Delete, _)
                                    if home_input == HomeInputMode::Composing =>
                                { editor.delete(); }
                                (KeyCode::Left, _)
                                    if home_input == HomeInputMode::Composing =>
                                { editor.left(); }
                                (KeyCode::Right, _)
                                    if home_input == HomeInputMode::Composing =>
                                { editor.right(); }
                                (KeyCode::Home, _) | (KeyCode::Char('a'), KeyModifiers::CONTROL)
                                    if home_input == HomeInputMode::Composing =>
                                { editor.home(); }
                                (KeyCode::End, _) | (KeyCode::Char('e'), KeyModifiers::CONTROL)
                                    if home_input == HomeInputMode::Composing =>
                                { editor.end(); }
                                (KeyCode::Up, _)
                                    if home_input == HomeInputMode::Composing =>
                                { editor.history_prev(); }
                                (KeyCode::Down, _)
                                    if home_input == HomeInputMode::Composing =>
                                { editor.history_next(); }
                                (KeyCode::Char(c), KeyModifiers::NONE)
                                | (KeyCode::Char(c), KeyModifiers::SHIFT)
                                    if home_input == HomeInputMode::Composing =>
                                { editor.insert(c); }
                                _ => {}
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Cleanup
        if let Some(r) = runner.take() { r.kill(); }
        if let Some(ref mut proc) = analysis_proc { proc.kill().await.ok(); }
        terminal.show_cursor()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal::disable_raw_mode()?;
        save_history(&editor.history);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Start a run — spawns analysis service + subprocess
// ---------------------------------------------------------------------------

async fn start_run(
    command: &str,
    cwd: Option<&str>,
    app: &AppContext,
    run_view: &mut RunViewState,
) -> (AnalysisClient, tokio::process::Child, mpsc::Receiver<RunnerEvent>, SubprocessRunner) {
    // Reset run view
    *run_view = RunViewState::default();
    run_view.focus_stdout = true;

    let run_id = new_run_id(command);
    run_view.status.command = command.to_string();
    run_view.status.run_id = run_id.clone();
    run_view.status.start_time = Some(Instant::now());

    // Pick a random port for the analysis service
    let port = pick_port();
    let db_path = app.db_dir.join(format!("{}.sqlite", run_id));

    // Start the analysis service subprocess
    let analysis_proc = spawn_analysis_service(port, &run_id, &db_path).await;

    run_view.push_stdout(
        format!("[mlc] starting: {}", command),
        StdoutLineStyle::Plain,
    );
    run_view.push_stdout(
        format!("[mlc] analysis service → http://127.0.0.1:{}/insights", port),
        StdoutLineStyle::Plain,
    );

    // Give the analysis service a moment to boot
    tokio::time::sleep(Duration::from_millis(300)).await;

    let analysis_client = AnalysisClient::new(port);

    // Spawn the training subprocess
    let (runner_tx, runner_rx) = mpsc::channel::<RunnerEvent>(256);
    let sub = SubprocessRunner::spawn(command, cwd, runner_tx).await
        .unwrap_or_else(|e| {
            tracing::error!("Failed to spawn process: {}", e);
            panic!("Failed to spawn: {}", e);
        });

    (analysis_client, analysis_proc, runner_rx, sub)
}

fn pick_port() -> u16 {
    use std::net::TcpListener;
    // Try to bind to port 0 and let the OS pick an ephemeral port
    TcpListener::bind("127.0.0.1:0")
        .ok()
        .and_then(|l| l.local_addr().ok())
        .map(|a| a.port())
        .unwrap_or(52341)
}

async fn spawn_analysis_service(
    port: u16,
    run_id: &str,
    db_path: &std::path::Path,
) -> tokio::process::Child {
    // Resolve the mlc-analysis package directory.
    // MLC_ANALYSIS_DIR overrides everything (useful for installed distributions).
    // In dev: CARGO_MANIFEST_DIR points to packages/mlc-cli; go up two levels.
    let analysis_dir = std::env::var("MLC_ANALYSIS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent().unwrap()  // packages/
                .parent().unwrap()  // repo root
                .join("packages/mlc-analysis")
        });

    // Use the venv python if present, otherwise fall back to system python3.
    let venv_python = analysis_dir.join(".venv/bin/python3");
    let python = if venv_python.exists() {
        venv_python.display().to_string()
    } else {
        "python3".to_string()
    };

    let main_py = analysis_dir.join("main.py");

    tokio::process::Command::new(&python)
        .args([
            main_py.to_str().unwrap(),
            "--port", &port.to_string(),
            "--run-id", run_id,
            "--db", &db_path.display().to_string(),
        ])
        .current_dir(&analysis_dir)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .unwrap_or_else(|e| panic!(
            "failed to spawn analysis service (python={}): {} — run `uv sync` in packages/mlc-analysis",
            python, e
        ))
}

// ---------------------------------------------------------------------------
// Apply InsightReport to run view state
// ---------------------------------------------------------------------------

fn apply_insight_report(run_view: &mut RunViewState, report: InsightReport) {
    run_view.status.health = if report.healthy {
        HealthSignal::Healthy
    } else {
        let top = report.alerts.iter()
            .find(|a| a.level == AlertLevel::Critical)
            .or_else(|| report.alerts.first());
        match top {
            Some(a) if a.level == AlertLevel::Critical => HealthSignal::Critical(a.signal.clone()),
            Some(a) => HealthSignal::Warning(a.signal.clone()),
            None => HealthSignal::Unknown,
        }
    };

    for ia in &report.alerts {
        let alert = Alert {
            level: ia.level.clone(),
            signal: ia.signal.clone(),
            metric: ia.metric.clone(),
            description: ia.description.clone(),
            confidence: ia.confidence,
            suggestions: ia.suggestions.clone(),
            received_at: Instant::now(),
        };
        run_view.push_alert(alert.clone());

        // Escalate to intervention panel on critical
        if ia.level == AlertLevel::Critical && run_view.active_critical.is_none() {
            run_view.active_critical = Some(CriticalState {
                alert,
                cursor: 1, // default to "continue"
                detail_open: false,
                raw_report: serde_json::to_string(&report).unwrap_or_default(),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

fn render(
    f: &mut Frame,
    app_mode: &AppMode,
    run_view: &RunViewState,
    editor: &LineEditor,
    recent_runs: &[RunRecord],
    home_input: &HomeInputMode,
) {
    match app_mode {
        AppMode::Idle => render_home(f, editor, recent_runs, home_input),
        AppMode::Running => render_running(f, run_view),
    }
}

fn render_home(
    f: &mut Frame,
    editor: &LineEditor,
    recent_runs: &[RunRecord],
    home_input: &HomeInputMode,
) {
    let area = f.area();

    let composing = *home_input == HomeInputMode::Composing;
    let v_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(if composing { 3 } else { 1 }),
        ])
        .split(area);

    // ---- Main panel ----
    let mut lines: Vec<Line> = Vec::new();

    // Header
    lines.push(Line::styled(
        format!(" mlc v{} — ML Training Monitor", env!("CARGO_PKG_VERSION")),
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
    ));
    lines.push(Line::raw(""));

    // Recent runs
    if recent_runs.is_empty() {
        lines.push(Line::styled(
            " No previous runs.",
            Style::default().fg(Color::DarkGray),
        ));
    } else {
        lines.push(Line::styled(
            " Recent Runs",
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ));
        lines.push(Line::styled(
            " ─────────────────────────────────────────────────",
            Style::default().fg(Color::DarkGray),
        ));
        for record in recent_runs {
            let status_icon = match record.exit_code {
                Some(0) => Span::styled("✓", Style::default().fg(Color::Green)),
                Some(_) => Span::styled("✖", Style::default().fg(Color::Red)),
                None    => Span::styled("?", Style::default().fg(Color::DarkGray)),
            };
            let exit_str = match record.exit_code {
                Some(c) => format!("exit {}", c),
                None    => "killed".to_string(),
            };
            // Exit code is authoritative; report health only applies on clean exit
            let report_healthy = record.final_report.as_ref()
                .and_then(|r| r.get("healthy"))
                .and_then(|v| v.as_bool());
            let (health_str, health_color) = match (record.exit_code, report_healthy) {
                (None,    _)           => ("● killed",  Color::Yellow),
                (Some(0), Some(false)) => ("● issue",   Color::Yellow),
                (Some(0), _)           => ("● healthy", Color::Green),
                (Some(_), _)           => ("● failed",  Color::Red),
            };

            let cmd_display = if record.command.len() > 36 {
                format!("{}…", &record.command[..35])
            } else {
                record.command.clone()
            };

            lines.push(Line::from(vec![
                Span::raw("  "),
                status_icon,
                Span::raw("  "),
                Span::styled(format!("{:<38}", cmd_display), Style::default().fg(Color::White)),
                Span::styled(format!("{:<10}", exit_str), Style::default().fg(Color::DarkGray)),
                Span::styled(health_str.to_string(), Style::default().fg(health_color)),
            ]));
        }
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Line::styled(" mlc ", Style::default().fg(Color::Cyan)));
    f.render_widget(Paragraph::new(lines).block(block), v_chunks[0]);

    // ---- Bottom: input bar or keybind hint ----
    if composing {
        render_input_bar(f, editor, v_chunks[1]);
    } else {
        let hint = Line::from(vec![
            Span::raw("  "),
            Span::styled("[R]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(" new run   ", Style::default().fg(Color::DarkGray)),
            Span::styled("[Q]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(" quit   ", Style::default().fg(Color::DarkGray)),
            Span::styled("[Ctrl+C]", Style::default().fg(Color::DarkGray)),
            Span::styled(" force quit", Style::default().fg(Color::DarkGray)),
        ]);
        f.render_widget(Paragraph::new(hint), v_chunks[1]);
    }
}

fn render_running(f: &mut Frame, run_view: &RunViewState) {
    let area = f.area();

    let has_critical = run_view.active_critical.is_some();
    let detail_open = run_view.active_critical.as_ref().map_or(false, |c| c.detail_open);
    let intervention_height = if has_critical { if detail_open { 8 } else { 5 } } else { 0 };

    let v_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),                          // top bar
            Constraint::Min(1),                             // main content
            Constraint::Length(6),                          // alert feed
            Constraint::Length(intervention_height),        // intervention panel
            Constraint::Length(1),                          // keybind hint
        ])
        .split(area);

    // Top bar
    render_top_bar(f, run_view, v_chunks[0]);

    // Main content: stdout (left 60%) | chart (right 40%)
    let h_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(v_chunks[1]);

    render_stdout(f, run_view, h_chunks[0]);
    render_chart(f, run_view, h_chunks[1]);

    // Alert feed
    render_alert_feed(f, run_view, v_chunks[2]);

    // Intervention panel (if critical)
    if has_critical {
        render_intervention_panel(f, run_view, v_chunks[3]);
    }

    // Keybind hint bar (replaces input bar — no typing during a run)
    let hint = if run_view.finished {
        Line::from(vec![
            Span::styled("  [ESC]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(" home   ", Style::default().fg(Color::DarkGray)),
            Span::styled("[Tab]", Style::default().fg(Color::DarkGray)),
            Span::styled(" switch pane   ", Style::default().fg(Color::DarkGray)),
            Span::styled("[↑/↓]", Style::default().fg(Color::DarkGray)),
            Span::styled(" scroll   ", Style::default().fg(Color::DarkGray)),
            Span::styled("[←/→]", Style::default().fg(Color::DarkGray)),
            Span::styled(" cycle metrics", Style::default().fg(Color::DarkGray)),
        ])
    } else {
        Line::from(vec![
            Span::styled("  [K]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            Span::styled(" kill   ", Style::default().fg(Color::DarkGray)),
            Span::styled("[Tab]", Style::default().fg(Color::DarkGray)),
            Span::styled(" switch pane   ", Style::default().fg(Color::DarkGray)),
            Span::styled("[↑/↓]", Style::default().fg(Color::DarkGray)),
            Span::styled(" scroll   ", Style::default().fg(Color::DarkGray)),
            Span::styled("[←/→]", Style::default().fg(Color::DarkGray)),
            Span::styled(" cycle metrics", Style::default().fg(Color::DarkGray)),
        ])
    };
    f.render_widget(Paragraph::new(hint), v_chunks[4]);
}

fn render_top_bar(f: &mut Frame, run_view: &RunViewState, area: Rect) {
    let elapsed = run_view.status.start_time
        .map(|t| {
            let s = t.elapsed().as_secs();
            format!("{:02}:{:02}:{:02}", s / 3600, (s % 3600) / 60, s % 60)
        })
        .unwrap_or_else(|| "--:--:--".to_string());

    let epoch_str = run_view.status.epoch
        .map(|e| format!("epoch {}", e))
        .unwrap_or_default();

    let (health_str, health_color) = match &run_view.status.health {
        HealthSignal::Unknown  => ("●", Color::DarkGray),
        HealthSignal::Healthy  => ("● healthy", Color::Green),
        HealthSignal::Warning(s) => (s.as_str(), Color::Yellow),
        HealthSignal::Critical(s) => (s.as_str(), Color::Red),
    };

    let cmd_short = if run_view.status.command.len() > 30 {
        format!("{}…", &run_view.status.command[..29])
    } else {
        run_view.status.command.clone()
    };

    let right_span = if run_view.finished {
        let exit_label = match run_view.status.exit_code {
            Some(0) => Span::styled("  ✓ done", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Some(c) => Span::styled(format!("  ✖ exit {}", c), Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            None    => Span::styled("  ● stopped", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        };
        exit_label
    } else {
        Span::styled("  [K] kill", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))
    };

    let line = Line::from(vec![
        Span::styled(format!(" {} ", cmd_short), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(format!("│ {} │ {} │ ", elapsed, epoch_str), Style::default().fg(Color::DarkGray)),
        Span::styled(health_str.to_string(), Style::default().fg(health_color)),
        right_span,
    ]);
    f.render_widget(Paragraph::new(line), area);
}

fn render_stdout(f: &mut Frame, run_view: &RunViewState, area: Rect) {
    let h = area.height.saturating_sub(2) as usize; // subtract border
    let total = run_view.stdout_lines.len();
    let max_scroll = total.saturating_sub(h);
    let scroll = max_scroll.saturating_sub(run_view.stdout_scroll);

    let lines: Vec<Line> = run_view.stdout_lines.iter()
        .zip(run_view.stdout_style.iter())
        .skip(scroll)
        .take(h)
        .map(|(text, style)| {
            let color = match style {
                StdoutLineStyle::Error  => Color::Red,
                StdoutLineStyle::Metric => Color::Cyan,
                StdoutLineStyle::Plain  => Color::Reset,
            };
            Line::styled(text.clone(), Style::default().fg(color))
        })
        .collect();

    let border_style = if run_view.focus_stdout {
        Style::default().fg(Color::White)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(" stdout ");
    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn fmt_val(v: f64) -> String {
    let abs = v.abs();
    if abs == 0.0 {
        "0".to_string()
    } else if abs >= 1e4 || (abs < 1e-3 && abs > 0.0) {
        format!("{:.2e}", v)
    } else {
        format!("{:.3}", v)
    }
}

fn render_chart(f: &mut Frame, run_view: &RunViewState, area: Rect) {
    let metric_count = run_view.metrics.len();

    if run_view.metrics.is_empty() {
        let block = Block::default().borders(Borders::ALL).title(" metrics ");
        f.render_widget(
            Paragraph::new("No metrics yet").block(block).style(Style::default().fg(Color::DarkGray)),
            area,
        );
        return;
    }

    let key = match &run_view.selected_metric {
        Some(k) => k.clone(),
        None => {
            let block = Block::default().borders(Borders::ALL).title(" metrics ");
            f.render_widget(Paragraph::new("").block(block), area);
            return;
        }
    };

    let title = Line::from(vec![
        Span::raw(" "),
        Span::styled(key.clone(), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(
            format!(" ({}/{}) ", run_view.metric_index + 1, metric_count),
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    let mut block = Block::default().borders(Borders::ALL).title(title);
    if metric_count > 1 {
        block = block.title_bottom(
            Line::styled(" [←] prev  [→] next ", Style::default().fg(Color::DarkGray))
        );
    }

    let points = match run_view.metrics.get(&key) {
        Some(pts) if !pts.is_empty() => pts,
        _ => {
            f.render_widget(Paragraph::new("").block(block), area);
            return;
        }
    };

    let min_x = points.first().map(|(x, _)| *x).unwrap_or(0.0);
    let max_x = points.last().map(|(x, _)| *x).unwrap_or(1.0);
    let min_y = points.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let max_y = points.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);
    let y_range = (max_y - min_y).max(0.001);

    let dataset = Dataset::default()
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Cyan))
        .data(points);

    let chart = Chart::new(vec![dataset])
        .block(block)
        .x_axis(Axis::default()
            .bounds([min_x, max_x])
            .labels(vec![
                Span::raw(fmt_val(min_x)),
                Span::raw(fmt_val(max_x)),
            ]))
        .y_axis(Axis::default()
            .bounds([min_y - y_range * 0.1, max_y + y_range * 0.1])
            .labels(vec![
                Span::raw(fmt_val(min_y)),
                Span::raw(fmt_val(max_y)),
            ]));

    f.render_widget(chart, area);
}

fn render_alert_feed(f: &mut Frame, run_view: &RunViewState, area: Rect) {
    let h = area.height.saturating_sub(2) as usize;
    let total = run_view.alert_feed.len();
    let max_scroll = total.saturating_sub(h);
    let scroll = max_scroll.saturating_sub(run_view.alert_scroll);

    let lines: Vec<Line> = if run_view.alert_feed.is_empty() {
        vec![Line::styled(
            "  Monitoring… alerts will appear here.",
            Style::default().fg(Color::DarkGray),
        )]
    } else {
        run_view.alert_feed.iter()
            .skip(scroll)
            .take(h)
            .map(|alert| {
                Line::styled(
                    format!("{} [{}] {} — {}",
                        alert.prefix(), alert.metric, alert.signal, alert.description),
                    Style::default().fg(alert.color()),
                )
            })
            .collect()
    };

    let border_style = if !run_view.focus_stdout {
        Style::default().fg(Color::White)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(" alerts [Tab to focus] ");
    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_intervention_panel(f: &mut Frame, run_view: &RunViewState, area: Rect) {
    let Some(crit) = &run_view.active_critical else { return; };

    let btn = |label: &str, idx: usize, cursor: usize| {
        let style = if cursor == idx {
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD | Modifier::REVERSED)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        Span::styled(format!("[ {} ]", label), style)
    };

    let desc = format!(
        "{} {} (conf {:.0}%) — {}",
        crit.alert.prefix(), crit.alert.metric,
        crit.alert.confidence * 100.0,
        crit.alert.description,
    );
    let suggestion = crit.alert.suggestions.first().cloned().unwrap_or_default();

    let mut lines = vec![
        Line::styled(desc, Style::default().fg(crit.alert.color()).add_modifier(Modifier::BOLD)),
        Line::styled(format!("  {}", suggestion), Style::default().fg(Color::DarkGray)),
        Line::from(vec![
            Span::raw("  "),
            btn("stop run", 0, crit.cursor),
            Span::raw("   "),
            btn("continue", 1, crit.cursor),
            Span::raw("   "),
            btn("view details", 2, crit.cursor),
            Span::styled("   ← → Tab to navigate, Enter to confirm", Style::default().fg(Color::DarkGray)),
        ]),
    ];

    if crit.detail_open {
        // Show all suggestions (up to 3 lines)
        for (i, s) in crit.alert.suggestions.iter().enumerate().take(3) {
            lines.push(Line::styled(
                format!("  {}. {}", i + 1, s),
                Style::default().fg(Color::White),
            ));
        }
        if crit.alert.suggestions.is_empty() {
            lines.push(Line::styled(
                format!("  confidence {:.0}% — no suggestions available", crit.alert.confidence * 100.0),
                Style::default().fg(Color::DarkGray),
            ));
        }
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Red))
        .title(" intervention ");
    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_input_bar(f: &mut Frame, editor: &LineEditor, area: Rect) {
    let pre: String = editor.buf[..editor.cursor].iter().collect();
    let post: String = editor.buf[editor.cursor..].iter().collect();
    let line = Line::from(vec![
        Span::styled("> ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        Span::raw(pre),
        Span::styled("█", Style::default().fg(Color::White)),
        Span::raw(post),
    ]);
    let block = Block::default().borders(Borders::ALL);
    f.render_widget(Paragraph::new(line).block(block), area);
}

// ---------------------------------------------------------------------------
// History persistence
// ---------------------------------------------------------------------------

fn history_path() -> PathBuf {
    dirs::home_dir().unwrap_or_default().join(HISTORY_FILE)
}

fn load_history() -> Vec<String> {
    let path = history_path();
    if !path.exists() { return Vec::new(); }
    std::fs::read_to_string(path)
        .map(|s| s.lines().map(str::to_owned).collect())
        .unwrap_or_default()
}

fn save_history(history: &[String]) {
    let path = history_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(path, history.join("\n"));
}
