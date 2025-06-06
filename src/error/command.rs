// src/cli/commands.rs - Updated with proper Result types
impl ConnectCommand {
    pub async fn execute(&self) -> gaussrelgt::Result<()> {
        use gaussrelgt::database::DatabaseConnectionManager;
        use console::style;
        
        println!("{}", style("Testing database connection...").cyan());
        
        let conn = DatabaseConnectionManager::new(&self.url, self.db_type).await?;
        let info = conn.get_info().await?;
        
        println!("{}", style("✓ Connection successful!").green());
        println!("Database: {} v{}", info.name, info.version);
        println!("Tables: {}", info.table_count);
        println!("Total rows: {}", info.total_rows);
        
        Ok(())
    }
}
