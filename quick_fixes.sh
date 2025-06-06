#!/bin/bash
# quick_fixes.sh - Common build issue fixes for GaussRELGT

echo "🔧 GaussRELGT Quick Fixes"
echo "========================="

# Fix 1: Remove benchmark configurations if benchmark files don't exist
echo "🔧 Fix 1: Cleaning up Cargo.toml..."
if [ -f "Cargo.toml" ]; then
    # Create backup
    cp Cargo.toml Cargo.toml.backup
    
    # Remove benchmark sections if bench files don't exist
    if [ ! -f "benches/training_benchmark.rs" ] && [ ! -f "benches/inference_benchmark.rs" ]; then
        # Remove benchmark configurations
        sed -i.tmp '/# Benchmark Configuration/,/harness = false/d' Cargo.toml
        rm -f Cargo.toml.tmp
        echo "✅ Removed benchmark configurations"
    fi
fi

# Fix 2: Update dependencies to compatible versions
echo "🔧 Fix 2: Updating dependencies..."
cat > Cargo.toml << 'EOF'
[package]
name = "gaussrelgt"
version = "0.1.0"
edition = "2021"
authors = ["GaussRELGT Team <team@gaussrelgt.ai>"]
description = "Advanced Relational Graph Transformer CLI for deep learning on relational databases"
license = "MIT OR Apache-2.0"

[dependencies]
# Core ML Framework
candle-core = "0.3"
candle-nn = "0.3"

# CLI Framework
clap = { version = "4.4", features = ["derive"] }
console = "0.15"
indicatif = "0.17"

# Database Connections
sqlx = { version = "0.7", features = ["runtime-tokio-rustls", "postgres", "chrono", "uuid", "json"], optional = true }
surrealdb = { version = "1.0", features = ["kv-mem"], optional = true }

# Async Runtime
tokio = { version = "1.0", features = ["full"] }
async-trait = "0.1"

# Serialization & Config
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
toml = "0.8"

# Error Handling & Logging
anyhow = "1.0"
thiserror = "1.0"
color-eyre = "0.6"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Utilities
uuid = { version = "1.0", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
rand = "0.8"
once_cell = "1.19"

# Optional Features
[features]
default = ["postgres"]
postgres = ["sqlx"]
surrealdb = ["dep:surrealdb"]
cuda = ["candle-core/cuda"]
metal = ["candle-core/metal"]

[[bin]]
name = "gaussrelgt"
path = "src/main.rs"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
EOF

echo "✅ Updated Cargo.toml with minimal dependencies"

# Fix 3: Create missing source file stubs if needed
echo "🔧 Fix 3: Ensuring all source files exist..."

# Check if all module files exist
REQUIRED_FILES=(
    "src/main.rs"
    "src/lib.rs"
    "src/error/mod.rs"
    "src/cli/mod.rs"
    "src/cli/commands.rs"
    "src/database/mod.rs"
    "src/database/connection.rs"
    "src/database/postgres.rs"
    "src/database/surrealdb.rs"
    "src/graph/mod.rs"
    "src/graph/types.rs"
    "src/graph/builder.rs"
    "src/graph/sampler.rs"
    "src/model/mod.rs"
    "src/model/relgt.rs"
    "src/training/mod.rs"
    "src/training/trainer.rs"
    "src/evaluation/mod.rs"
    "src/evaluation/evaluator.rs"
    "src/data/mod.rs"
    "src/data/loader.rs"
    "src/utils/mod.rs"
    "src/utils/logger.rs"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "⚠️  Missing: $file"
        # Create directory if it doesn't exist
        mkdir -p "$(dirname "$file")"
        
        # Create minimal stub
        case "$file" in
            */mod.rs)
                echo "// Module placeholder for $(dirname "$file")" > "$file"
                ;;
            *)
                echo "// Placeholder for $file" > "$file"
                echo "use crate::Result;" >> "$file"
                ;;
        esac
        echo "✅ Created stub: $file"
    fi
done

# Fix 4: Clean build artifacts
echo "🔧 Fix 4: Cleaning build artifacts..."
cargo clean
echo "✅ Build artifacts cleaned"

# Fix 5: Update Rust toolchain if needed
echo "🔧 Fix 5: Checking Rust version..."
RUST_VERSION=$(rustc --version | cut -d' ' -f2)
echo "Current Rust version: $RUST_VERSION"

if rustup check 2>/dev/null; then
    echo "✅ Rust toolchain is up to date"
else
    echo "⚠️  Consider updating Rust toolchain with: rustup update"
fi

# Fix 6: Provide build command suggestions
echo ""
echo "🚀 Suggested build commands:"
echo "============================"
echo "1. Basic check:           cargo check"
echo "2. Debug build:           cargo build"
echo "3. Release build:         cargo build --release"
echo "4. CPU-only build:        cargo build --no-default-features"
echo "5. With PostgreSQL:       cargo build --features postgres"
echo "6. With CUDA (if available): cargo build --features cuda"
echo ""

# Fix 7: Test basic build
echo "🧪 Testing basic build..."
if cargo check --quiet; then
    echo "✅ Basic build check passed!"
    echo ""
    echo "🎉 Quick fixes applied successfully!"
    echo "You can now run: cargo build --release"
else
    echo "❌ Build still has issues. Please check the error messages above."
    echo ""
    echo "Common solutions:"
    echo "1. Make sure all source files from the complete implementation are present"
    echo "2. Check that all dependencies are compatible"
    echo "3. Verify Rust version is 1.70 or higher"
    echo "4. Try: cargo update to update dependencies"
fi
EOF