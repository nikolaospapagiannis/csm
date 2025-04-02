#!/bin/bash
echo "Starting CSM Voice Assistant with Socket.IO..."
echo ""

echo "Setting environment variables..."
export NO_TORCH_COMPILE=1
export CHUNK_SIZE=30
export MAX_AUDIO_LENGTH=2000

echo "Starting server..."
python app_socketio.py
