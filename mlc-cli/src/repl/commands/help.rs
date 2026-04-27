use anyhow::Result;
use async_trait::async_trait;
use super::{Command, CommandContext, CommandOutcome};

pub struct HelpCommand {
    entries: Vec<(String, String)>,
}

impl HelpCommand {
    pub fn new(entries: Vec<(String, String)>) -> Self {
        Self { entries }
    }
}

#[async_trait]
impl Command for HelpCommand {
    fn name(&self) -> &str { "/help" }
    fn description(&self) -> &str { "Show this help" }

    async fn run(&self, _ctx: &CommandContext, _args: &str) -> Result<CommandOutcome> {
        let max_len = self.entries.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
        let mut out = String::from("Commands\n\n");
        for (name, desc) in &self.entries {
            out.push_str(&format!("  {:<width$}  {}\n", name, desc, width = max_len));
        }
        out.push_str("\nKeybinds (during a run)\n\n");
        out.push_str("  K        Kill the running process immediately\n");
        out.push_str("  PgUp/Dn  Scroll stdout / alert feed\n");
        out.push_str("  Tab      Switch focus between stdout and alert feed\n");
        out.push_str("  Left/Right / Tab  Navigate intervention panel buttons\n");
        Ok(CommandOutcome::Message(out))
    }
}
