#!/bin/bash

# Migration script to move GaussRelGT to GaussRDL workspace structure
# This script helps migrate the existing monolithic codebase to the new workspace

set -e

echo "🚀 Starting migration from GaussRelGT to GaussRDL workspace..."

# Create backup of original source
echo "📦 Creating backup of original source..."
cp -r src src_backup_$(date +%Y%m%d_%H%M%S)

# Function to move files from src to appropriate crate
move_to_crate() {
    local source_dir=$1
    local target_crate=$2
    local target_dir=$3
    
    if [ -d "src/$source_dir" ]; then
        echo "📁 Moving $source_dir to $target_crate/$target_dir"
        mkdir -p "$target_crate/src/$target_dir"
        cp -r "src/$source_dir"/* "$target_crate/src/$target_dir/"
    fi
}

# Move core functionality
echo "🔧 Moving core functionality..."
move_to_crate "base" "gaussrdl-core" "base"
move_to_crate "error.rs" "gaussrdl-core" ""
move_to_crate "core" "gaussrdl-core" "core"

# Move data-related functionality
echo "📊 Moving data functionality..."
move_to_crate "datasets" "gaussrdl-data" "datasets"
move_to_crate "data" "gaussrdl-data" "data"
move_to_crate "tasks" "gaussrdl-data" "tasks"
move_to_crate "task" "gaussrdl-data" "task"
move_to_crate "dataset" "gaussrdl-data" "dataset"

# Move graph functionality
echo "🕸️ Moving graph functionality..."
move_to_crate "graph" "gaussrdl-graph" "graph"

# Move model functionality
echo "🧠 Moving model functionality..."
move_to_crate "models" "gaussrdl-models" "models"
move_to_crate "model" "gaussrdl-models" "model"

# Move training functionality
echo "🏋️ Moving training functionality..."
move_to_crate "training" "gaussrdl-training" "training"

# Move metrics functionality
echo "📈 Moving metrics functionality..."
move_to_crate "metrics" "gaussrdl-metrics" "metrics"
move_to_crate "monitoring" "gaussrdl-metrics" "monitoring"

# Move database functionality
echo "🗄️ Moving database functionality..."
move_to_crate "database" "gaussrdl-database" "database"

# Move server functionality
echo "🌐 Moving server functionality..."
move_to_crate "server" "gaussrdl-server" "server"

# Move CLI functionality
echo "💻 Moving CLI functionality..."
move_to_crate "cli" "gaussrdl-cli" "cli"

# Move utility functionality
echo "🔧 Moving utility functionality..."
move_to_crate "utils" "gaussrdl-utils" "utils"
move_to_crate "memory" "gaussrdl-utils" "memory"
move_to_crate "parallel" "gaussrdl-utils" "parallel"
move_to_crate "io" "gaussrdl-utils" "io"
move_to_crate "gpu" "gaussrdl-utils" "gpu"
move_to_crate "distributed" "gaussrdl-utils" "distributed"
move_to_crate "testing" "gaussrdl-utils" "testing"
move_to_crate "di" "gaussrdl-utils" "di"

# Create lib.rs files for each crate if they don't exist
echo "📝 Creating lib.rs files..."

# gaussrdl-data lib.rs
cat > gaussrdl-data/src/lib.rs << 'EOF'
//! Data loading, datasets, and data processing for GaussRDL

pub mod datasets;
pub mod data;
pub mod tasks;
pub mod task;
pub mod dataset;

// Re-export main functionality
pub use datasets::{get_dataset, DatasetRegistry};
pub use tasks::{get_task, TaskRegistry};
EOF

# gaussrdl-graph lib.rs
cat > gaussrdl-graph/src/lib.rs << 'EOF'
//! Graph construction and manipulation for GaussRDL

pub mod graph;
EOF

# gaussrdl-models lib.rs
cat > gaussrdl-models/src/lib.rs << 'EOF'
//! ML models and neural networks for GaussRDL

pub mod models;
pub mod model;
EOF

# gaussrdl-training lib.rs
cat > gaussrdl-training/src/lib.rs << 'EOF'
//! Training infrastructure and optimizers for GaussRDL

pub mod training;
EOF

# gaussrdl-metrics lib.rs
cat > gaussrdl-metrics/src/lib.rs << 'EOF'
//! Evaluation metrics and monitoring for GaussRDL

pub mod metrics;
pub mod monitoring;
EOF

# gaussrdl-database lib.rs
cat > gaussrdl-database/src/lib.rs << 'EOF'
//! Database connectivity and operations for GaussRDL

pub mod database;
EOF

# gaussrdl-server lib.rs
cat > gaussrdl-server/src/lib.rs << 'EOF'
//! HTTP server and API for GaussRDL

pub mod server;
EOF

# gaussrdl-cli lib.rs
cat > gaussrdl-cli/src/lib.rs << 'EOF'
//! Command-line interface for GaussRDL

pub mod cli;
EOF

# gaussrdl-utils lib.rs
cat > gaussrdl-utils/src/lib.rs << 'EOF'
//! Utility functions and helpers for GaussRDL

pub mod utils;
pub mod memory;
pub mod parallel;
pub mod io;
pub mod gpu;
pub mod distributed;
pub mod testing;
pub mod di;
EOF

# Update main.rs to use the new workspace
echo "🔄 Updating main.rs..."
cat > gaussrdl/src/main.rs << 'EOF'
use gaussrdl::cli::cli::Cli;

#[tokio::main]
async fn main() -> gaussrdl::Result<()> {
    let cli = Cli::parse();
    
    // Initialize logging
    env_logger::init();
    
    // Handle CLI commands
    gaussrdl::cli::handlers::handle_commands(cli).await?;
    
    Ok(())
}
EOF

echo "✅ Migration completed!"
echo ""
echo "📋 Next steps:"
echo "1. Review the migrated code in each crate"
echo "2. Fix any import paths and dependencies"
echo "3. Run 'cargo check' to identify compilation issues"
echo "4. Update any remaining references to the old structure"
echo "5. Test the new workspace with 'cargo build'"
echo ""
echo "🔧 To build the workspace:"
echo "   cargo build"
echo ""
echo "🧪 To test the workspace:"
echo "   cargo test"
echo ""
echo "📚 The original source is backed up in src_backup_*" 