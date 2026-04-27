pub mod clear;
pub mod exit;
pub mod help;

use anyhow::Result;
use async_trait::async_trait;
use indexmap::IndexMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Command context — passed to every command handler
// ---------------------------------------------------------------------------

pub struct CommandContext;

// ---------------------------------------------------------------------------
// Command trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait Command: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    async fn run(&self, ctx: &CommandContext, args: &str) -> Result<CommandOutcome>;
}

pub enum CommandOutcome {
    Ok,
    Exit,
    Message(String),
}

// ---------------------------------------------------------------------------
// CommandRegistry
// ---------------------------------------------------------------------------

pub struct CommandRegistry {
    commands: IndexMap<String, Arc<dyn Command>>,
}

impl CommandRegistry {
    pub fn new() -> Self {
        Self { commands: IndexMap::new() }
    }

    pub fn register(&mut self, cmd: impl Command + 'static) {
        self.commands.insert(cmd.name().to_owned(), Arc::new(cmd));
    }

    pub fn all(&self) -> impl Iterator<Item = &Arc<dyn Command>> {
        self.commands.values()
    }

    pub async fn dispatch(&self, text: &str, ctx: &CommandContext) -> CommandOutcome {
        let mut parts = text.trim().splitn(2, char::is_whitespace);
        let cmd_name = parts.next().unwrap_or("");
        let args = parts.next().unwrap_or("");

        match self.commands.get(cmd_name) {
            None => CommandOutcome::Message(format!("Unknown command: {}. Type /help for commands.", cmd_name)),
            Some(cmd) => match cmd.run(ctx, args).await {
                Ok(outcome) => outcome,
                Err(e) => CommandOutcome::Message(format!("Error: {}", e)),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

pub fn build_registry() -> CommandRegistry {
    let mut r = CommandRegistry::new();
    r.register(clear::ClearCommand);
    r.register(exit::ExitCommand);
    let all: Vec<(String, String)> = r.all()
        .map(|c| (c.name().to_owned(), c.description().to_owned()))
        .collect();
    r.register(help::HelpCommand::new(all));
    r
}
