"""
Startup Utilities for CSM Voice Chat Assistant

This module provides utilities for enhancing the application startup experience,
including console messages, system checks, and initialization helpers.
"""

import os
import sys
import platform
import socket
import logging
import time
import psutil
import torch
from typing import Dict, Any, List, Optional, Tuple
import importlib.metadata as metadata
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for diagnostics
    
    Returns:
        Dictionary with system information
    """
    try:
        # Get Python version
        python_version = platform.python_version()
        
        # Get OS information
        os_info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }
        
        # Get memory information
        memory = psutil.virtual_memory()
        memory_info = {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used,
            "free": memory.free
        }
        
        # Get disk information
        disk = psutil.disk_usage('/')
        disk_info = {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent
        }
        
        # Get CPU information
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False),
            "total_cores": psutil.cpu_count(logical=True),
            "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "percent": psutil.cpu_percent(interval=1)
        }
        
        # Get GPU information
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "memory_allocated": torch.cuda.memory_allocated(0),
                "memory_reserved": torch.cuda.memory_reserved(0),
                "max_memory_allocated": torch.cuda.max_memory_allocated(0)
            }
        
        # Get package versions
        packages = [
            "flask", "torch", "torchaudio", "openai", "anthropic", 
            "flask-socketio", "pydub", "numpy", "pillow"
        ]
        package_versions = {}
        for package in packages:
            try:
                package_versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                package_versions[package] = "not installed"
        
        # Combine all information
        system_info = {
            "python_version": python_version,
            "os": os_info,
            "memory": memory_info,
            "disk": disk_info,
            "cpu": cpu_info,
            "gpu": gpu_info,
            "packages": package_versions,
            "timestamp": time.time()
        }
        
        return system_info
    except Exception as e:
        logger.error(f"Error getting system information: {str(e)}")
        return {"error": str(e)}

def check_dependencies() -> Tuple[bool, List[str]]:
    """
    Check if all required dependencies are installed
    
    Returns:
        Tuple of (success, missing_dependencies)
    """
    required_packages = [
        "flask", "torch", "torchaudio", "python-dotenv", 
        "pydub", "requests", "pytest", "openai", "anthropic",
        "PyJWT", "SQLAlchemy", "numpy", "Pillow", "soundfile"
    ]
    
    missing = []
    for package in required_packages:
        try:
            metadata.version(package)
        except metadata.PackageNotFoundError:
            missing.append(package)
    
    return len(missing) == 0, missing

def check_gpu() -> Tuple[bool, str]:
    """
    Check if GPU is available and working
    
    Returns:
        Tuple of (success, message)
    """
    if not torch.cuda.is_available():
        return False, "CUDA is not available. Running in CPU mode."
    
    try:
        # Try to allocate a small tensor on GPU
        x = torch.rand(10, 10).cuda()
        y = x + x
        
        # Get GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        
        return True, f"GPU is working: {gpu_name} with {gpu_memory / (1024**3):.2f} GB memory"
    except Exception as e:
        return False, f"GPU error: {str(e)}"

def check_model_files() -> Tuple[bool, List[str]]:
    """
    Check if required model files exist
    
    Returns:
        Tuple of (success, missing_files)
    """
    required_files = [
        "models/phi-2.Q4_K_M.gguf"
    ]
    
    missing = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing.append(file_path)
    
    return len(missing) == 0, missing

def get_local_ip() -> str:
    """
    Get the local IP address of the machine
    
    Returns:
        Local IP address
    """
    try:
        # Create a socket to determine the local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

def print_startup_message(host: str = None, port: int = None) -> None:
    """
    Print a clear startup message with access URLs
    
    Args:
        host: Host address (defaults to value from .env)
        port: Port number (defaults to value from .env)
    """
    # Get host and port from environment if not provided
    if host is None:
        host = os.getenv("HOST", "127.0.0.1")
    
    if port is None:
        port = int(os.getenv("PORT", 5000))
    
    # Get local IP for network access
    local_ip = get_local_ip()
    
    # Check if running in debug mode
    debug_mode = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    
    # Check if Socket.IO is enabled
    socketio_mode = os.path.basename(sys.argv[0]) == "app_socketio.py" if len(sys.argv) > 0 else False
    
    message = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                      CSM VOICE CHAT ASSISTANT                          ║
╚═══════════════════════════════════════════════════════════════════════╝

🚀 Server is running!

🌐 Access URLs:
   • Local:           http://127.0.0.1:{port}
   • On Your Network: http://{local_ip}:{port}

💻 Web Interfaces:
   • Chat Interface:  http://127.0.0.1:{port}/
   • Landing Page:    http://127.0.0.1:{port}/landing

🔊 Voice Settings:
   • Speaker ID:      {os.getenv("SPEAKER_ID", "0")}
   • Max Audio Length: {os.getenv("MAX_AUDIO_LENGTH", "5000")} ms
   • Chunk Size:      {os.getenv("CHUNK_SIZE", "60")} chars

🔌 Server Mode:
   • {'Socket.IO (Real-time streaming)' if socketio_mode else 'Standard (Server-Sent Events)'}
   • {'Debug Mode: ON' if debug_mode else 'Production Mode'}

⚙️ System Status:
   • GPU: {'Available' if torch.cuda.is_available() else 'Not available (CPU mode)'}
   • {'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU: ' + platform.processor()}

🧠 LLM Integration:
   • OpenAI API:      {'Configured' if os.getenv("OPENAI_API_KEY") else 'Not configured'}
   • Azure OpenAI:    {'Configured' if os.getenv("AZURE_OPENAI_KEY") else 'Not configured'}
   • Anthropic API:   {'Configured' if os.getenv("ANTHROPIC_API_KEY") else 'Not configured'}

🎤 Speech-to-Text:
   • Whisper API:     {'Configured' if os.getenv("OPENAI_API_KEY") else 'Not configured'}
   • Azure Speech:    {'Configured' if os.getenv("AZURE_SPEECH_KEY") else 'Not configured'}
   • Local Whisper:   {'Enabled' if os.getenv("USE_LOCAL_WHISPER", "False").lower() in ("true", "1", "t") else 'Disabled'}

Press CTRL+C to stop the server
"""
    print(message)

def run_system_checks() -> Dict[str, Any]:
    """
    Run system checks and return results
    
    Returns:
        Dictionary with check results
    """
    results = {}
    
    # Check dependencies
    deps_ok, missing_deps = check_dependencies()
    results["dependencies"] = {
        "status": "ok" if deps_ok else "warning",
        "message": "All dependencies installed" if deps_ok else f"Missing dependencies: {', '.join(missing_deps)}"
    }
    
    # Check GPU
    gpu_ok, gpu_message = check_gpu()
    results["gpu"] = {
        "status": "ok" if gpu_ok else "warning",
        "message": gpu_message
    }
    
    # Check model files
    models_ok, missing_models = check_model_files()
    results["models"] = {
        "status": "ok" if models_ok else "error",
        "message": "All model files found" if models_ok else f"Missing model files: {', '.join(missing_models)}"
    }
    
    # Check environment variables
    required_env_vars = ["NO_TORCH_COMPILE", "PORT", "HOST", "SPEAKER_ID"]
    missing_env_vars = [var for var in required_env_vars if os.getenv(var) is None]
    env_ok = len(missing_env_vars) == 0
    
    results["environment"] = {
        "status": "ok" if env_ok else "warning",
        "message": "All required environment variables set" if env_ok else f"Missing environment variables: {', '.join(missing_env_vars)}"
    }
    
    # Overall status
    if any(result["status"] == "error" for result in results.values()):
        results["overall"] = {
            "status": "error",
            "message": "System checks failed. Please fix the errors before continuing."
        }
    elif any(result["status"] == "warning" for result in results.values()):
        results["overall"] = {
            "status": "warning",
            "message": "System checks completed with warnings. The application may not function correctly."
        }
    else:
        results["overall"] = {
            "status": "ok",
            "message": "All system checks passed. The application is ready to run."
        }
    
    return results

def initialize_app() -> Dict[str, Any]:
    """
    Initialize the application and run system checks
    
    Returns:
        Dictionary with initialization results
    """
    # Load environment variables
    load_dotenv()
    
    # Run system checks
    check_results = run_system_checks()
    
    # Print startup message
    print_startup_message()
    
    # Log system status
    if check_results["overall"]["status"] == "ok":
        logger.info("Application initialized successfully")
    elif check_results["overall"]["status"] == "warning":
        logger.warning(f"Application initialized with warnings: {check_results['overall']['message']}")
    else:
        logger.error(f"Application initialization failed: {check_results['overall']['message']}")
    
    return {
        "status": check_results["overall"]["status"],
        "message": check_results["overall"]["message"],
        "checks": check_results,
        "system_info": get_system_info()
    }

# If this script is run directly, print system information
if __name__ == "__main__":
    print_startup_message()
    print("\nSystem Information:")
    system_info = get_system_info()
    for key, value in system_info.items():
        if key != "packages":
            print(f"  {key}: {value}")
    
    print("\nPackage Versions:")
    for package, version in system_info.get("packages", {}).items():
        print(f"  {package}: {version}")