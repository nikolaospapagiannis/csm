#!/bin/bash
echo "Starting CSM Voice Chat Assistant..."
echo "=============================="

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found."
    echo "Please run ./install.sh first."
    exit 1
fi

source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment."
    echo "Please run ./install.sh first."
    exit 1
fi

# Set required environment variable
export NO_TORCH_COMPILE=1

# Start the application
echo "Starting server..."
echo "The web interface will be available at: http://127.0.0.1:5000"
python app.py
