use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::runner::MetricEvent;

// ---------------------------------------------------------------------------
// InsightReport — what the analysis service returns
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InsightReport {
    pub run_id: String,
    pub healthy: bool,
    pub alerts: Vec<InsightAlert>,
    pub reasoning: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InsightAlert {
    pub level: AlertLevel,
    pub signal: String,
    pub metric: String,
    pub confidence: f64,
    pub description: String,
    pub suggestions: Vec<String>,
}

// ---------------------------------------------------------------------------
// AnalysisClient — batching HTTP client for the local analysis service
// ---------------------------------------------------------------------------

const BATCH_SIZE: usize = 50;
const FLUSH_INTERVAL_MS: u128 = 200;

pub struct AnalysisClient {
    base_url: String,
    client: reqwest::Client,
    buffer: Vec<MetricEvent>,
    last_flush: Instant,
}

impl AnalysisClient {
    pub fn new(port: u16) -> Self {
        Self {
            base_url: format!("http://127.0.0.1:{}", port),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .expect("http client"),
            buffer: Vec::with_capacity(BATCH_SIZE),
            last_flush: Instant::now(),
        }
    }

    /// Queue a metric event; flushes automatically when batch is full or interval elapsed.
    pub async fn send_event(&mut self, ev: MetricEvent) {
        self.buffer.push(ev);
        if self.buffer.len() >= BATCH_SIZE
            || self.last_flush.elapsed().as_millis() >= FLUSH_INTERVAL_MS
        {
            self.flush().await;
        }
    }

    /// Flush buffered events to the analysis service (fire-and-forget on error).
    pub async fn flush(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        let events = std::mem::take(&mut self.buffer);
        self.last_flush = Instant::now();
        let url = format!("{}/events", self.base_url);
        self.client.post(&url).json(&events).send().await.ok();
    }

    /// Fetch the current InsightReport from the analysis service.
    pub async fn get_insights(&self) -> Result<InsightReport> {
        let url = format!("{}/insights", self.base_url);
        let report = self.client.get(&url).send().await?.json::<InsightReport>().await?;
        Ok(report)
    }

    /// Check if the analysis service is reachable.
    pub async fn health_check(&self) -> bool {
        let url = format!("{}/health", self.base_url);
        self.client.get(&url).send().await.map(|r| r.status().is_success()).unwrap_or(false)
    }
}
