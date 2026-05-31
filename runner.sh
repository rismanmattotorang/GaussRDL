#!/bin/bash

# GaussRelGT Runner Script
# ========================
# This script sets up and runs both the CLI and Streamlit web interface

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CLI_PORT=8000
STREAMLIT_PORT=8501
LOG_DIR="./logs"
PID_FILE="./gaussrelgt.pid"

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check if Rust/Cargo is installed
    if ! command -v cargo &> /dev/null; then
        print_error "Cargo is not installed. Please install Rust: https://rustup.rs/"
        exit 1
    fi
    
    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8 or later."
        exit 1
    fi
    
    # Check if Streamlit is installed
    if ! python3 -c "import streamlit" &> /dev/null; then
        print_warning "Streamlit is not installed. Installing..."
        python3 -m pip install streamlit plotly pandas || {
            print_warning "Failed to install Streamlit automatically. Please install manually:"
            print_warning "python3 -m pip install streamlit plotly pandas"
        }
    fi
    
    print_success "All dependencies are available"
}

# Setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Set environment variables
    export RUST_LOG=info
    export RUST_BACKTRACE=1
    
    print_success "Environment setup complete"
}

# Build the project
build_project() {
    print_status "Building GaussRelGT project..."
    
    if cargo build --release; then
        print_success "Build completed successfully"
    else
        print_error "Build failed"
        exit 1
    fi
}

# Start the CLI server
start_cli_server() {
    print_status "Starting CLI server on port $CLI_PORT..."
    
    # Check if CLI server is already running
    if pgrep -f "gaussrelgt.*serve" > /dev/null; then
        print_warning "CLI server appears to be already running"
        return 0
    fi
    
    # Start the CLI server in the background
    nohup ./target/release/gaussrelgt serve \
        --host 0.0.0.0 \
        --port "$CLI_PORT" \
        --docs \
        > "$LOG_DIR/cli_server.log" 2>&1 &
    
    CLI_PID=$!
    echo "$CLI_PID" >> "$PID_FILE"
    
    # Wait a moment for the server to start
    sleep 3
    
    # Check if the server started successfully
    if kill -0 "$CLI_PID" 2>/dev/null; then
        print_success "CLI server started with PID $CLI_PID"
        print_status "CLI server accessible at: http://localhost:$CLI_PORT"
    else
        print_error "Failed to start CLI server"
        return 1
    fi
}

# Start the Streamlit app
start_streamlit_app() {
    print_status "Starting Streamlit web interface on port $STREAMLIT_PORT..."
    
    # Check if Streamlit is already running
    if pgrep -f "streamlit.*run" > /dev/null; then
        print_warning "Streamlit app appears to be already running"
        return 0
    fi
    
    # Start Streamlit in the background
    nohup streamlit run streamlit_app.py \
        --server.port "$STREAMLIT_PORT" \
        --server.address 0.0.0.0 \
        --server.headless true \
        --browser.gatherUsageStats false \
        > "$LOG_DIR/streamlit.log" 2>&1 &
    
    STREAMLIT_PID=$!
    echo "$STREAMLIT_PID" >> "$PID_FILE"
    
    # Wait a moment for Streamlit to start
    sleep 5
    
    # Check if Streamlit started successfully
    if kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        print_success "Streamlit app started with PID $STREAMLIT_PID"
        print_status "Web interface accessible at: http://localhost:$STREAMLIT_PORT"
    else
        print_error "Failed to start Streamlit app"
        return 1
    fi
}

# Stop all services
stop_services() {
    print_status "Stopping GaussRelGT services..."
    
    if [ -f "$PID_FILE" ]; then
        while read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                print_status "Stopping process $pid"
                kill "$pid"
                # Wait for graceful shutdown
                sleep 2
                # Force kill if still running
                if kill -0 "$pid" 2>/dev/null; then
                    kill -9 "$pid" 2>/dev/null
                fi
            fi
        done < "$PID_FILE"
        
        rm -f "$PID_FILE"
        print_success "All services stopped"
    else
        print_warning "No PID file found. Attempting to stop known processes..."
        pkill -f "gaussrelgt.*serve" || true
        pkill -f "streamlit.*run.*streamlit_app.py" || true
    fi
}

# Show status
show_status() {
    print_status "GaussRelGT Service Status:"
    echo ""
    
    # Check CLI server
    if pgrep -f "gaussrelgt.*serve" > /dev/null; then
        print_success "✓ CLI server is running on http://localhost:$CLI_PORT"
    else
        print_error "✗ CLI server is not running"
    fi
    
    # Check Streamlit app
    if pgrep -f "streamlit.*run" > /dev/null; then
        print_success "✓ Streamlit app is running on http://localhost:$STREAMLIT_PORT"
    else
        print_error "✗ Streamlit app is not running"
    fi
    
    echo ""
    
    # Show recent logs
    if [ -d "$LOG_DIR" ]; then
        print_status "Recent log entries:"
        echo ""
        
        if [ -f "$LOG_DIR/cli_server.log" ]; then
            echo "CLI Server logs (last 5 lines):"
            tail -n 5 "$LOG_DIR/cli_server.log"
            echo ""
        fi
        
        if [ -f "$LOG_DIR/streamlit.log" ]; then
            echo "Streamlit logs (last 5 lines):"
            tail -n 5 "$LOG_DIR/streamlit.log"
            echo ""
        fi
    fi
}

# Show logs
show_logs() {
    local service="$1"
    
    case "$service" in
        "cli")
            if [ -f "$LOG_DIR/cli_server.log" ]; then
                print_status "CLI Server logs:"
                tail -f "$LOG_DIR/cli_server.log"
            else
                print_error "CLI server log file not found"
            fi
            ;;
        "streamlit"|"web")
            if [ -f "$LOG_DIR/streamlit.log" ]; then
                print_status "Streamlit logs:"
                tail -f "$LOG_DIR/streamlit.log"
            else
                print_error "Streamlit log file not found"
            fi
            ;;
        "all"|*)
            print_status "All logs (use Ctrl+C to exit):"
            if [ -f "$LOG_DIR/cli_server.log" ] && [ -f "$LOG_DIR/streamlit.log" ]; then
                tail -f "$LOG_DIR/cli_server.log" "$LOG_DIR/streamlit.log"
            elif [ -f "$LOG_DIR/cli_server.log" ]; then
                tail -f "$LOG_DIR/cli_server.log"
            elif [ -f "$LOG_DIR/streamlit.log" ]; then
                tail -f "$LOG_DIR/streamlit.log"
            else
                print_error "No log files found"
            fi
            ;;
    esac
}

# Restart services
restart_services() {
    print_status "Restarting GaussRelGT services..."
    stop_services
    sleep 2
    start_all_services
}

# Start all services
start_all_services() {
    check_dependencies
    setup_environment
    build_project
    
    # Start services
    start_cli_server
    start_streamlit_app
    
    echo ""
    print_success "🚀 GaussRelGT is now running!"
    echo ""
    print_status "📊 Web Interface: http://localhost:$STREAMLIT_PORT"
    print_status "🔧 CLI API:       http://localhost:$CLI_PORT"
    print_status "📚 API Docs:      http://localhost:$CLI_PORT/docs"
    echo ""
    print_status "Use './runner.sh status' to check service status"
    print_status "Use './runner.sh stop' to stop all services"
    print_status "Use './runner.sh logs' to view logs"
    echo ""
}

# Quick development mode
dev_mode() {
    print_status "Starting in development mode..."
    
    check_dependencies
    setup_environment
    
    # Build in debug mode for faster compilation
    print_status "Building in debug mode..."
    cargo build
    
    # Start services using debug binary
    print_status "Starting CLI server (debug mode)..."
    nohup ./target/debug/gaussrelgt serve \
        --host 0.0.0.0 \
        --port "$CLI_PORT" \
        --docs \
        > "$LOG_DIR/cli_server_dev.log" 2>&1 &
    
    CLI_PID=$!
    echo "$CLI_PID" >> "$PID_FILE"
    
    sleep 2
    
    # Start Streamlit with auto-reload
    print_status "Starting Streamlit (development mode)..."
    streamlit run streamlit_app.py \
        --server.port "$STREAMLIT_PORT" \
        --server.address 0.0.0.0 \
        --server.runOnSave true \
        --browser.gatherUsageStats false
}

# Cleanup function for graceful shutdown
cleanup() {
    print_status "Received interrupt signal, shutting down..."
    stop_services
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Help function
show_help() {
    echo "GaussRelGT Runner Script"
    echo "========================"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start         Start all services (CLI server + Streamlit app)"
    echo "  stop          Stop all services"
    echo "  restart       Restart all services"
    echo "  status        Show service status"
    echo "  logs [SERVICE] Show logs (all, cli, streamlit/web)"
    echo "  dev           Start in development mode (faster builds, auto-reload)"
    echo "  build         Build the project only"
    echo "  clean         Clean build artifacts and logs"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start              # Start all services"
    echo "  $0 logs cli           # Show CLI server logs"
    echo "  $0 dev                # Start in development mode"
    echo ""
    echo "Default ports:"
    echo "  CLI Server: $CLI_PORT"
    echo "  Streamlit:  $STREAMLIT_PORT"
    echo ""
}

# Clean up build artifacts and logs
clean() {
    print_status "Cleaning up..."
    
    # Stop services first
    stop_services
    
    # Clean build artifacts
    cargo clean
    
    # Clean logs
    if [ -d "$LOG_DIR" ]; then
        rm -rf "$LOG_DIR"
        print_success "Logs cleaned"
    fi
    
    # Remove PID file
    rm -f "$PID_FILE"
    
    print_success "Cleanup complete"
}

# Main command handling
case "$1" in
    "start"|"")
        start_all_services
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        restart_services
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs "$2"
        ;;
    "dev")
        dev_mode
        ;;
    "build")
        check_dependencies
        build_project
        ;;
    "clean")
        clean
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac 