use gaussrdl::cli::Cli;
use gaussrdl::cli::handlers::CommandHandler;
use clap::Parser;
use gaussrdl::GaussRDLResult;

#[tokio::main]
async fn main() -> GaussRDLResult<()> {
    let cli = Cli::parse();
    
    // Initialize logging
    env_logger::init();
    
    // Handle CLI commands
    CommandHandler::new((&cli).into())?.handle_command(cli.command).await?;
    
    Ok(())
}
