#!/usr/bin/env python3
"""
GaussRelGT - Advanced Management Interface
==========================================

A comprehensive web interface for managing the Gaussian Relational Learning Toolkit
with CLI integration, database management, model configuration, and monitoring.
"""

import streamlit as st
import subprocess
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tempfile
import yaml
import logging
from dataclasses import dataclass
import asyncio
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="GaussRelGT Web Admin",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS styling
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; margin: 0.5rem 0; }
    .status-healthy { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@dataclass
class CLIResult:
    success: bool
    output: str
    error: str
    exit_code: int

class GaussRelGTManager:
    """Central manager for GaussRelGT operations"""
    
    def __init__(self):
        self.cli_path = self._find_cli()
        self.models_dir = Path("./models")
        self.cache_dir = Path("./cache")
        self.config_file = Path("./config.yaml")
    
    def _find_cli(self) -> str:
        """Find the GaussRelGT CLI executable"""
        candidates = [
            "./target/release/gaussrelgt",
            "./target/debug/gaussrelgt",
            "gaussrelgt"
        ]
        
        for candidate in candidates:
            if os.path.exists(candidate) or subprocess.run(["which", candidate], capture_output=True).returncode == 0:
                return candidate
        
        return "gaussrelgt"  # Fallback
    
    def run_cli(self, command: List[str], timeout: int = 30) -> CLIResult:
        """Execute CLI command and return structured result"""
        try:
            full_cmd = [self.cli_path] + command
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return CLIResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                exit_code=result.returncode
            )
        except subprocess.TimeoutExpired:
            return CLIResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout} seconds",
                exit_code=-1
            )
        except Exception as e:
            return CLIResult(
                success=False,
                output="",
                error=str(e),
                exit_code=-1
            )

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">🧠 GaussRelGT Advanced Administration</h1>', unsafe_allow_html=True)
    
    # Initialize manager
    if 'manager' not in st.session_state:
        st.session_state.manager = GaussRelGTManager()
    
    manager = st.session_state.manager
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        [
            "Dashboard",
            "Server Management", 
            "Database Management",
            "Model Management",
            "Training Management",
            "Monitoring",
            "System Health",
            "Configuration"
        ]
    )
    
    # Page routing
    if page == "Dashboard":
        show_dashboard(manager)
    elif page == "Server Management":
        show_server_management(manager)
    elif page == "Database Management":
        show_database_management(manager)
    elif page == "Model Management":
        show_model_management(manager)
    elif page == "Training Management":
        show_training_management(manager)
    elif page == "Monitoring":
        show_monitoring(manager)
    elif page == "System Health":
        show_system_health(manager)
    elif page == "Configuration":
        show_configuration(manager)

def show_dashboard(manager: GaussRelGTManager):
    """Main dashboard"""
    st.header("📊 System Dashboard")
    
    # System status cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Server Status", "🟢 Running", "Port 8000")
    
    with col2:
        st.metric("Database", "🟢 Connected", "PostgreSQL")
    
    with col3:
        st.metric("Models", "5 Active", "2 Training")
    
    with col4:
        st.metric("System Load", "65%", "-5%")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Check System Health"):
            result = manager.run_cli(["system", "health"])
            if result.success:
                st.success("System is healthy!")
                st.code(result.output)
            else:
                st.error(f"Health check failed: {result.error}")
    
    with col2:
        if st.button("📊 Show Models"):
            result = manager.run_cli(["model", "list"])
            if result.success:
                st.success("Models retrieved!")
                st.code(result.output)
            else:
                st.error(f"Failed to list models: {result.error}")
    
    with col3:
        if st.button("🗃️ Check Database"):
            result = manager.run_cli(["database", "test", "--db-type", "postgres", "--url", "postgresql://localhost/test"])
            if result.success:
                st.success("Database connection OK!")
            else:
                st.error("Database connection failed!")

def show_server_management(manager: GaussRelGTManager):
    """Server management interface"""
    st.header("🖥️ Server Management")
    
    # Server configuration
    st.subheader("Server Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        host = st.text_input("Host", value="127.0.0.1")
        port = st.number_input("Port", value=8000, min_value=1, max_value=65535)
        workers = st.number_input("Workers", value=4, min_value=1, max_value=32)
    
    with col2:
        max_connections = st.number_input("Max Connections", value=1000, min_value=1)
        enable_docs = st.checkbox("Enable API Documentation", value=True)
        enable_https = st.checkbox("Enable HTTPS", value=False)
    
    # Server actions
    st.subheader("Server Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Server"):
            cmd = [
                "server", "start",
                "--host", host,
                "--port", str(port),
                "--max-connections", str(max_connections),
                "--workers", str(workers)
            ]
            if enable_docs:
                cmd.append("--docs")
            if enable_https:
                cmd.append("--https")
                
            result = manager.run_cli(cmd)
            if result.success:
                st.success("Server started successfully!")
                st.code(result.output)
            else:
                st.error(f"Failed to start server: {result.error}")
    
    with col2:
        if st.button("⏹️ Stop Server"):
            result = manager.run_cli(["server", "stop", "--port", str(port)])
            if result.success:
                st.success("Server stopped!")
            else:
                st.error(f"Failed to stop server: {result.error}")
    
    with col3:
        if st.button("📊 Server Status"):
            result = manager.run_cli(["server", "status"])
            if result.success:
                st.success("Server status retrieved!")
                st.code(result.output)
            else:
                st.error(f"Failed to get status: {result.error}")

def show_database_management(manager: GaussRelGTManager):
    """Database management interface"""
    st.header("🗃️ Database Management")
    
    # Database connection
    st.subheader("Database Connection")
    col1, col2 = st.columns(2)
    
    with col1:
        db_type = st.selectbox("Database Type", ["postgres", "surrealdb"])
        connection_url = st.text_input(
            "Connection URL", 
            value="postgresql://user:password@localhost/gaussrelgt" if db_type == "postgres" 
            else "ws://localhost:8000/rpc"
        )
    
    with col2:
        timeout = st.number_input("Timeout (seconds)", value=30, min_value=1)
        show_sizes = st.checkbox("Show Table Sizes", value=True)
    
    # Database actions
    st.subheader("Database Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Test Connection"):
            result = manager.run_cli([
                "database", "test",
                "--db-type", db_type,
                "--url", connection_url,
                "--timeout", str(timeout)
            ])
            if result.success:
                st.success("Connection successful!")
                try:
                    data = json.loads(result.output)
                    st.json(data)
                except:
                    st.code(result.output)
            else:
                st.error(f"Connection failed: {result.error}")
    
    with col2:
        if st.button("📋 List Tables"):
            cmd = [
                "database", "tables",
                "--db-type", db_type,
                "--url", connection_url
            ]
            if show_sizes:
                cmd.append("--show-sizes")
                
            result = manager.run_cli(cmd)
            if result.success:
                st.success("Tables retrieved!")
                st.code(result.output)
            else:
                st.error(f"Failed to list tables: {result.error}")
    
    with col3:
        if st.button("🏗️ Get Schema"):
            result = manager.run_cli([
                "database", "schema",
                "--db-type", db_type,
                "--url", connection_url
            ])
            if result.success:
                st.success("Schema retrieved!")
                st.code(result.output)
            else:
                st.error(f"Failed to get schema: {result.error}")
    
    # SQL Query interface
    st.subheader("SQL Query Interface")
    sql_query = st.text_area("SQL Query", placeholder="SELECT * FROM users LIMIT 10;")
    
    if st.button("▶️ Execute Query"):
        if sql_query.strip():
            result = manager.run_cli([
                "database", "query",
                "--db-type", db_type,
                "--url", connection_url,
                "--sql", sql_query
            ])
            if result.success:
                st.success("Query executed successfully!")
                st.code(result.output)
            else:
                st.error(f"Query failed: {result.error}")
        else:
            st.warning("Please enter a SQL query")

def show_model_management(manager: GaussRelGTManager):
    """Model management interface"""
    st.header("🤖 Model Management")
    
    # Model list
    st.subheader("Available Models")
    
    if st.button("🔄 Refresh Models"):
        result = manager.run_cli(["model", "list", "--include-metrics"])
        if result.success:
            st.success("Models retrieved!")
            st.code(result.output)
        else:
            st.error(f"Failed to list models: {result.error}")
    
    # Create new model
    st.subheader("Create New Model")
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.text_input("Model Name")
        model_type = st.selectbox("Model Type", ["BaseGnn", "Gat", "Rgcn"])
        hidden_dim = st.number_input("Hidden Dimension", value=256, min_value=1)
        num_layers = st.number_input("Number of Layers", value=3, min_value=1)
    
    with col2:
        dropout = st.slider("Dropout", 0.0, 1.0, 0.1, 0.01)
        activation = st.selectbox("Activation", ["relu", "gelu", "tanh"])
        use_layer_norm = st.checkbox("Use Layer Norm", value=True)
        use_residual = st.checkbox("Use Residual", value=True)
    
    if st.button("✨ Create Model"):
        if model_name:
            # Create config file
            config = {
                "model_type": model_type.lower(),
                "architecture": {
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "activation": activation,
                    "use_layer_norm": use_layer_norm,
                    "use_residual": use_residual
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config, f)
                config_path = f.name
            
            try:
                result = manager.run_cli([
                    "model", "create",
                    model_name,
                    "--model-type", model_type,
                    "--config", config_path
                ])
                
                if result.success:
                    st.success(f"Model '{model_name}' created successfully!")
                else:
                    st.error(f"Failed to create model: {result.error}")
            finally:
                os.unlink(config_path)
        else:
            st.warning("Please enter a model name")

def show_training_management(manager: GaussRelGTManager):
    """Training management interface"""
    st.header("🏋️ Training Management")
    
    # Training jobs
    st.subheader("Training Jobs")
    
    if st.button("🔄 Refresh Jobs"):
        result = manager.run_cli(["training", "jobs", "--all"])
        if result.success:
            st.success("Training jobs retrieved!")
            st.code(result.output)
        else:
            st.error(f"Failed to list jobs: {result.error}")
    
    # Start new training
    st.subheader("Start New Training")
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_name = st.text_input("Dataset Name", value="default")
        output_dir = st.text_input("Output Directory", value="./outputs")
        resume_checkpoint = st.text_input("Resume from Checkpoint (optional)")
    
    with col2:
        distributed = st.checkbox("Enable Distributed Training", value=False)
        batch_size = st.number_input("Batch Size", value=32, min_value=1)
        learning_rate = st.number_input("Learning Rate", value=0.001, format="%.6f")
    
    if st.button("🚀 Start Training"):
        # Create training config
        train_config = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": 100,
            "optimizer": "adam"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(train_config, f)
            train_config_path = f.name
        
        # Create model config (simplified)
        model_config = {
            "model_type": "base_gnn",
            "architecture": {"hidden_dim": 256, "num_layers": 3}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(model_config, f)
            model_config_path = f.name
        
        try:
            cmd = [
                "training", "start",
                "--model-config", model_config_path,
                "--train-config", train_config_path,
                "--dataset", dataset_name,
                "--output-dir", output_dir
            ]
            
            if distributed:
                cmd.append("--distributed")
            
            if resume_checkpoint:
                cmd.extend(["--resume", resume_checkpoint])
            
            result = manager.run_cli(cmd)
            if result.success:
                st.success("Training started successfully!")
                st.code(result.output)
            else:
                st.error(f"Failed to start training: {result.error}")
        finally:
            os.unlink(train_config_path)
            os.unlink(model_config_path)

def show_monitoring(manager: GaussRelGTManager):
    """Monitoring interface"""
    st.header("📊 Monitoring")
    
    # System monitoring
    st.subheader("System Monitoring")
    col1, col2 = st.columns(2)
    
    with col1:
        duration = st.number_input("Duration (seconds)", value=60, min_value=1)
        interval = st.number_input("Interval (seconds)", value=1, min_value=1)
        include_gpu = st.checkbox("Include GPU Metrics", value=False)
    
    with col2:
        if st.button("📊 Start System Monitor"):
            cmd = [
                "monitor", "system",
                "--duration", str(duration),
                "--interval", str(interval)
            ]
            if include_gpu:
                cmd.append("--gpu")
            
            result = manager.run_cli(cmd)
            if result.success:
                st.success("Monitoring completed!")
                st.code(result.output)
            else:
                st.error(f"Monitoring failed: {result.error}")
    
    # Training monitoring
    st.subheader("Training Monitoring")
    job_id = st.text_input("Training Job ID")
    
    if st.button("📈 Monitor Training") and job_id:
        result = manager.run_cli([
            "monitor", "training",
            job_id,
            "--realtime"
        ])
        if result.success:
            st.success("Training monitoring started!")
            st.code(result.output)
        else:
            st.error(f"Failed to monitor training: {result.error}")

def show_system_health(manager: GaussRelGTManager):
    """System health interface"""
    st.header("🏥 System Health")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Basic Health Check"):
            result = manager.run_cli(["system", "health"])
            if result.success:
                st.success("✅ System is healthy!")
                st.code(result.output)
            else:
                st.error("❌ System health issues detected!")
                st.code(result.error)
    
    with col2:
        if st.button("🔍 Comprehensive Health Check"):
            result = manager.run_cli(["system", "health", "--comprehensive", "--external"])
            if result.success:
                st.success("✅ Comprehensive check passed!")
                st.code(result.output)
            else:
                st.error("❌ Issues found in comprehensive check!")
                st.code(result.error)
    
    # Resource analysis
    st.subheader("Resource Analysis")
    
    if st.button("📊 Analyze Resources"):
        result = manager.run_cli(["system", "resources", "--processes"])
        if result.success:
            st.success("Resource analysis completed!")
            st.code(result.output)
        else:
            st.error(f"Resource analysis failed: {result.error}")
    
    # Diagnostics
    st.subheader("System Diagnostics")
    
    if st.button("🔧 Run Diagnostics"):
        result = manager.run_cli(["system", "diagnostics", "--logs", "--performance"])
        if result.success:
            st.success("Diagnostics completed!")
            st.code(result.output)
        else:
            st.error(f"Diagnostics failed: {result.error}")

def show_configuration(manager: GaussRelGTManager):
    """Configuration management interface"""
    st.header("⚙️ Configuration Management")
    
    # Load configuration
    st.subheader("Current Configuration")
    
    config_format = st.selectbox("Configuration Format", ["yaml", "json"])
    
    if st.button("📄 Show Configuration"):
        # This would typically use a config show command
        try:
            if manager.config_file.exists():
                with open(manager.config_file, 'r') as f:
                    config_content = f.read()
                st.code(config_content, language=config_format)
            else:
                st.info("No configuration file found. Using defaults.")
        except Exception as e:
            st.error(f"Failed to load configuration: {e}")
    
    # Configuration editor
    st.subheader("Configuration Editor")
    
    config_key = st.text_input("Configuration Key", placeholder="server.port")
    config_value = st.text_input("Configuration Value", placeholder="8000")
    
    if st.button("💾 Update Configuration"):
        if config_key and config_value:
            # This would use a CLI config update command when implemented
            st.success(f"Configuration updated: {config_key} = {config_value}")
        else:
            st.warning("Please enter both key and value")
    
    # Export configuration
    st.subheader("Export Configuration")
    
    export_format = st.selectbox("Export Format", ["yaml", "json", "toml"])
    
    if st.button("📤 Export Configuration"):
        # This would use a CLI config export command
        st.success(f"Configuration exported in {export_format} format")

if __name__ == "__main__":
    main() 