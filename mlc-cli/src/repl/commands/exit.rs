use anyhow::Result;
use async_trait::async_trait;
use super::{Command, CommandContext, CommandOutcome};

pub struct ExitCommand;

#[async_trait]
impl Command for ExitCommand {
    fn name(&self) -> &str { "/exit" }
    fn description(&self) -> &str { "Exit mlc" }
    async fn run(&self, _ctx: &CommandContext, _args: &str) -> Result<CommandOutcome> {
        Ok(CommandOutcome::Exit)
    }
}
