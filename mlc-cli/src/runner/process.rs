use anyhow::Result;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot};
use std::process::Stdio;

#[derive(Debug)]
pub enum RunnerEvent {
    StdoutLine(String),
    StderrLine(String),
    Exited { code: Option<i32> },
}

/// Handle to a running subprocess. Call `kill()` to send SIGTERM.
pub struct SubprocessRunner {
    kill_tx: oneshot::Sender<()>,
}

impl SubprocessRunner {
    /// Spawn `command` via `sh -c`. Lines from stdout/stderr are sent on `event_tx`.
    /// `RunnerEvent::Exited` is sent when the process ends (naturally or via kill).
    pub async fn spawn(
        command: &str,
        cwd: Option<&str>,
        event_tx: mpsc::Sender<RunnerEvent>,
    ) -> Result<Self> {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", command])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .env("PYTHONUNBUFFERED", "1");  // force line-buffered output when piped

        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }

        let mut child = cmd.spawn()?;

        // Spawn stdout reader
        let stdout = child.stdout.take().expect("stdout piped");
        let tx_out = event_tx.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if tx_out.send(RunnerEvent::StdoutLine(line)).await.is_err() {
                    break;
                }
            }
        });

        // Spawn stderr reader
        let stderr = child.stderr.take().expect("stderr piped");
        let tx_err = event_tx.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if tx_err.send(RunnerEvent::StderrLine(line)).await.is_err() {
                    break;
                }
            }
        });

        // The kill channel: sending on kill_tx triggers SIGTERM.
        let (kill_tx, kill_rx) = oneshot::channel::<()>();

        // Supervisor task: waits for natural exit OR kill signal.
        tokio::spawn(async move {
            tokio::select! {
                status = child.wait() => {
                    let code = status.ok().and_then(|s| s.code());
                    event_tx.send(RunnerEvent::Exited { code }).await.ok();
                }
                _ = kill_rx => {
                    child.kill().await.ok();
                    let code = child.wait().await.ok().and_then(|s| s.code());
                    event_tx.send(RunnerEvent::Exited { code }).await.ok();
                }
            }
        });

        Ok(Self { kill_tx })
    }

    /// Send SIGTERM to the process. The supervisor task will emit `RunnerEvent::Exited`.
    pub fn kill(self) {
        // Sending on the oneshot triggers the kill branch in the supervisor task.
        self.kill_tx.send(()).ok();
    }
}
