#!/bin/bash
echo "CSM Voice Chat Assistant Installer"
echo "=============================="

# Check for Python installation
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH."
    echo "Please install Python 3.8 or higher and try again."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
min_version="3.8"
if [ "$(printf '%s\n' "$min_version" "$python_version" | sort -V | head -n1)" != "$min_version" ]; then
    echo "Error: Python version must be 3.8 or higher. Found: $python_version"
    exit 1
fi

echo "Creating virtual environment..."
python3 -m venv .venv
if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment."
    exit 1
fi

echo "Activating virtual environment..."
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment."
    exit 1
fi

echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies."
    exit 1
fi

# Make the start script executable
chmod +x start.sh

echo
echo "Installation complete!"
echo
echo "To start the application:"
echo "1. Open Terminal in this directory"
echo "2. Run: ./start.sh"
echo
echo "Alternatively, you can run these commands manually:"
echo
echo "  source .venv/bin/activate"
echo "  export NO_TORCH_COMPILE=1"
echo "  python app.py"
echo
echo "Press any key to exit..."
read -n 1 -s
