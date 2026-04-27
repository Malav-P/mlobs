use anyhow::Result;
use async_trait::async_trait;
use super::{Command, CommandContext, CommandOutcome};

pub struct ClearCommand;

#[async_trait]
impl Command for ClearCommand {
    fn name(&self) -> &str { "/clear" }
    fn description(&self) -> &str { "Clear the stdout display buffer" }
    async fn run(&self, _ctx: &CommandContext, _args: &str) -> Result<CommandOutcome> {
        Ok(CommandOutcome::Ok)  // The REPL handles clearing the buffer directly
    }
}
