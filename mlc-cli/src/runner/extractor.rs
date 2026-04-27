use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use regex::Regex;
use serde::Serialize;

// ---------------------------------------------------------------------------
// MetricEvent — the narrow waist type crossing into the analysis service
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct MetricEvent {
    pub run_id: String,
    pub key: String,
    pub value: f64,
    pub step: Option<u64>,
    pub ts: f64, // unix timestamp (seconds)
}

// ---------------------------------------------------------------------------
// Line classifier
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum LineClass {
    /// Parsed metric(s) — buffer the line AND send events to the service.
    Metric(Vec<RawMetric>),
    /// stderr or lines containing ERROR/WARN/Traceback — buffer with error styling.
    Error,
    /// tqdm \r overwrite lines — display only, never buffer.
    Progress,
    /// Empty / whitespace — discard.
    Blank,
    /// Regular stdout — buffer as plain text.
    Plain,
}

#[derive(Debug, Clone)]
pub struct RawMetric {
    pub key: String,
    pub value: f64,
    pub step: Option<u64>,
}

/// Classify a single line before buffering or forwarding.
pub fn classify(line: &str, is_stderr: bool) -> LineClass {
    if line.trim().is_empty() {
        return LineClass::Blank;
    }
    // tqdm overwrites use \r
    if line.contains('\r') {
        return LineClass::Progress;
    }
    if is_stderr
        || line.contains("ERROR")
        || line.contains("WARN")
        || line.contains("Traceback")
        || line.contains("Error:")
        || line.contains("Exception:")
    {
        return LineClass::Error;
    }
    let metrics = extract_metrics(line);
    if !metrics.is_empty() {
        return LineClass::Metric(metrics);
    }
    LineClass::Plain
}

// ---------------------------------------------------------------------------
// Metric extraction
// ---------------------------------------------------------------------------

struct Patterns {
    json_obj: Regex,
    epoch_kv: Regex,
    step_kv: Regex,
    simple_kv: Regex,
    tqdm_dict: Regex,
}

fn patterns() -> &'static Patterns {
    static ONCE: OnceLock<Patterns> = OnceLock::new();
    ONCE.get_or_init(|| Patterns {
        // Full-line JSON object
        json_obj: Regex::new(r#"^\s*\{.*\}\s*$"#).unwrap(),
        // Epoch N/M: key=val, key=val ...
        epoch_kv: Regex::new(
            r"(?i)epoch\s+(\d+)[/\s\d]*[:\s,]+((?:[\w_]+\s*=\s*[\d.e+\-]+[\s,]*)+)"
        ).unwrap(),
        // [Step N] key=val or Step N: key=val
        step_kv: Regex::new(
            r"(?i)\[?step\s+(\d+)\]?[:\s,]+((?:[\w_]+\s*=\s*[\d.e+\-]+[\s,]*)+)"
        ).unwrap(),
        // Bare key=val or key: val (float values only, to avoid false matches)
        simple_kv: Regex::new(
            r"([\w_]+)\s*[:=]\s*(\d+\.\d+(?:[eE][+\-]?\d+)?)"
        ).unwrap(),
        // tqdm / HF Trainer dict suffix: {'loss': 1.23, ...} or {"loss": 1.23}
        tqdm_dict: Regex::new(
            r#"\{((?:['"]?[\w_]+['"]?\s*:\s*[\d.e+\-]+,?\s*)+)\}"#
        ).unwrap(),
    })
}

/// Extract all metrics from a single stdout line.
pub fn extract_metrics(line: &str) -> Vec<RawMetric> {
    let p = patterns();
    let mut results = Vec::new();

    // 1. Full JSON object line
    if p.json_obj.is_match(line) {
        // Normalise single-quoted Python dicts → valid JSON
        let normalised = line.trim().replace('\'', "\"");
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&normalised) {
            if let Some(obj) = val.as_object() {
                let step = obj.get("step").or_else(|| obj.get("epoch"))
                    .and_then(|v| v.as_u64());
                for (k, v) in obj {
                    if k == "step" || k == "epoch" { continue; }
                    if let Some(f) = v.as_f64() {
                        results.push(RawMetric { key: k.clone(), value: f, step });
                    }
                }
            }
        }
        if !results.is_empty() { return results; }
    }

    // 2. Epoch prefix: "Epoch 5/50: loss=0.43, val_loss=0.89"
    if let Some(caps) = p.epoch_kv.captures(line) {
        let step = caps.get(1).and_then(|m| m.as_str().parse::<u64>().ok());
        let kv_str = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        let pairs = parse_kv_pairs(kv_str, step);
        if !pairs.is_empty() { return pairs; }
    }

    // 3. Step prefix: "[Step 100] loss=0.43"
    if let Some(caps) = p.step_kv.captures(line) {
        let step = caps.get(1).and_then(|m| m.as_str().parse::<u64>().ok());
        let kv_str = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        let pairs = parse_kv_pairs(kv_str, step);
        if !pairs.is_empty() { return pairs; }
    }

    // 4. tqdm/HF dict suffix: {'loss': 1.23, 'lr': 5e-5}
    if let Some(caps) = p.tqdm_dict.captures(line) {
        let inner = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        let normalised = format!("{{{}}}", inner.replace('\'', "\""));
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&normalised) {
            if let Some(obj) = val.as_object() {
                for (k, v) in obj {
                    if let Some(f) = v.as_f64() {
                        results.push(RawMetric { key: k.clone(), value: f, step: None });
                    }
                }
            }
        }
        if !results.is_empty() { return results; }
    }

    // 5. Simple key=val / key: val scan (floats only)
    for caps in p.simple_kv.captures_iter(line) {
        let key = caps.get(1).map(|m| m.as_str()).unwrap_or("").to_string();
        let val_str = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        // Skip very common non-metric identifiers
        if matches!(key.as_str(), "version" | "pid" | "port" | "line" | "col") { continue; }
        if let Ok(value) = val_str.parse::<f64>() {
            results.push(RawMetric { key, value, step: None });
        }
    }

    results
}

fn parse_kv_pairs(s: &str, step: Option<u64>) -> Vec<RawMetric> {
    let p = patterns();
    p.simple_kv.captures_iter(s).filter_map(|caps| {
        let key = caps.get(1)?.as_str().to_string();
        let val: f64 = caps.get(2)?.as_str().parse().ok()?;
        Some(RawMetric { key, value: val, step })
    }).collect()
}

// ---------------------------------------------------------------------------
// Per-key downsampling rate gate (max 5 samples/second per key)
// ---------------------------------------------------------------------------

pub struct RateGate {
    last_seen: HashMap<String, Instant>,
    min_interval: Duration,
}

impl RateGate {
    pub fn new() -> Self {
        Self {
            last_seen: HashMap::new(),
            min_interval: Duration::from_millis(200),
        }
    }

    /// Returns true if this key should be forwarded to the analysis service.
    pub fn should_pass(&mut self, key: &str) -> bool {
        let now = Instant::now();
        let entry = self.last_seen.entry(key.to_string())
            .or_insert_with(|| now - self.min_interval - Duration::from_millis(1));
        if now.duration_since(*entry) >= self.min_interval {
            *entry = now;
            true
        } else {
            false
        }
    }
}
