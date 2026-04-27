use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RunRecord {
    pub id: String,
    pub command: String,
    pub cwd: String,
    pub started_at: String,  // RFC3339
    pub ended_at: Option<String>,
    pub exit_code: Option<i32>,
    pub final_report: Option<serde_json::Value>,
    pub db_path: String,
}

impl RunRecord {
    pub fn new(id: &str, command: &str, cwd: &str, db_path: &str) -> Self {
        Self {
            id: id.to_string(),
            command: command.to_string(),
            cwd: cwd.to_string(),
            started_at: Utc::now().to_rfc3339(),
            ended_at: None,
            exit_code: None,
            final_report: None,
            db_path: db_path.to_string(),
        }
    }
}

pub fn save_run(dir: &Path, record: &RunRecord) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let filename = format!("{}.json", record.id);
    let json = serde_json::to_string_pretty(record)?;
    std::fs::write(dir.join(filename), json)?;
    Ok(())
}

pub fn list_runs(dir: &Path, limit: usize) -> Result<Vec<RunRecord>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut records: Vec<RunRecord> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
        .filter_map(|e| {
            std::fs::read_to_string(e.path()).ok()
                .and_then(|s| serde_json::from_str(&s).ok())
        })
        .collect();

    records.sort_by(|a, b| b.started_at.cmp(&a.started_at));
    records.truncate(limit);
    Ok(records)
}

/// Generate a run ID: timestamp slug based on current time and command.
pub fn new_run_id(command: &str) -> String {
    let ts = Utc::now().format("%Y%m%dT%H%M%S");
    let slug: String = command
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .take(5)
        .collect::<Vec<_>>()
        .join("-");
    format!("{}-{}", ts, slug)
}
