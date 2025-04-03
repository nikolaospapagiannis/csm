# CSM Voice Chat Assistant

A web-based voice assistant built with [Controllable Speech Model (CSM)](https://github.com/SesameAILabs/csm) for high-quality, natural-sounding speech and a simple text interface.

## Features

- 🗣️ High-quality speech synthesis with CSM
- 🔄 Multiple voice options (8 different voices)
- 💬 Interactive chat interface
- 📱 Modern, responsive UI
- 🎛️ Simple REST API for chat interactions
- 🔌 Optional Socket.IO for real-time streaming (recommended)
- 🤖 LLM integration (OpenAI, Azure OpenAI, Anthropic)
- 🎙️ Voice input with speech-to-text capabilities
- 🐳 Docker containerization for easy deployment
- 🔄 CI/CD pipeline for automated testing and deployment

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch
- CUDA-capable GPU (recommended for faster performance)

### Setup

#### Standard Setup

1. Clone the CSM repository and set up your environment:

```bash
git clone https://github.com/SesameAILabs/csm.git
cd csm
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Linux/Mac
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

#### Docker Setup (Recommended)

1. Clone the repository:

```bash
git clone https://github.com/SesameAILabs/csm.git
cd csm
```

2. Build and run with Docker Compose:

```bash
docker-compose up -d
```

## Running the Application

### Standard Version

Run the application with the standard REST-based interface:

```bash
start.bat  # On Windows
./start.sh  # On Linux/Mac
```

### Socket.IO Version (Recommended)

For a better experience with real-time streaming:

```bash
start_socketio.bat  # On Windows
./start_socketio.sh  # On Linux/Mac
```

Then visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Configuration

The application can be configured using the `.env` file (see `.env.example` for all available options):

```
# Required environment variable for torch
NO_TORCH_COMPILE=1

# Application settings
PORT=5000
HOST=127.0.0.1
DEBUG=True

# Voice settings
SPEAKER_ID=0  # 0-7 for different voices
MAX_AUDIO_LENGTH=5000  # Maximum audio length in milliseconds
CHUNK_SIZE=60  # Text chunk size for speech generation

# LLM Integration (Optional)
# Uncomment and add your API keys to enable LLM integration
# OPENAI_API_KEY=your_openai_key_here
# AZURE_OPENAI_KEY=your_azure_openai_key_here
# AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
# ANTHROPIC_API_KEY=your_anthropic_key_here

# Speech-to-Text (Optional)
# Uncomment and add your API keys to enable speech-to-text
# OPENAI_API_KEY=your_openai_key_here  # For Whisper API
# AZURE_SPEECH_KEY=your_azure_speech_key_here
# AZURE_SPEECH_REGION=your_azure_region_here
# USE_LOCAL_WHISPER=False  # Set to True to use local Whisper model
```

## Usage

1. Open your browser and navigate to http://127.0.0.1:5000
2. You'll be greeted with a welcome message
3. Type a message in the text box and press Send or hit Enter
4. Alternatively, click the microphone button to use voice input
5. The assistant will respond with both text and speech
6. You can change voices using the dropdown menu
7. Click the "Clear Chat" button to start a new conversation

## LLM Integration

The assistant can be enhanced with external LLM APIs for more intelligent responses:

1. OpenAI API (GPT models)
2. Azure OpenAI API
3. Anthropic API (Claude models)

To enable these integrations, add the appropriate API keys to your `.env` file.

## Speech-to-Text

Voice input is supported through multiple speech-to-text services:

1. OpenAI Whisper API
2. Azure Speech Service
3. Local Whisper model (requires additional setup)

To enable voice input, add the appropriate API keys to your `.env` file.

## Docker Deployment

The application can be easily deployed using Docker:

```bash
# Build and run with Docker Compose
docker-compose up -d

# To stop the containers
docker-compose down
```

## CI/CD Pipeline

The project includes a GitHub Actions workflow for continuous integration and deployment:

- Automated testing on multiple Python versions
- Docker image building and publishing
- Deployment automation (configurable)

## Troubleshooting

### Audio Not Playing

Modern browsers require user interaction before playing audio. If you don't hear any voice:
- Click anywhere on the page to enable audio
- Make sure your browser's audio is not muted
- Try reducing the `MAX_AUDIO_LENGTH` in the `.env` file if you encounter errors

### Memory Issues

If you experience out-of-memory errors:
- Reduce `MAX_AUDIO_LENGTH` in the `.env` file (default: 5000ms)
- Use smaller `CHUNK_SIZE` values (default: 60 characters)
- Close other GPU-intensive applications

### Socket.IO Version

If the Socket.IO version isn't working:
- Make sure you have the correct version of Flask-SocketIO: `pip install flask-socketio==5.3.4`
- Check that you're using the correct HTML template for Socket.IO

### Voice Input Not Working

If voice input isn't working:
- Ensure you've granted microphone permissions in your browser
- Check that you've added the required API keys to your `.env` file
- Try using a different browser if issues persist

## Implementation Details

The application is built with:

- **Flask**: Web framework
- **CSM**: Controllable Speech Model for high-quality voice synthesis
- **Socket.IO** (optional): For real-time streaming communication
- **HTML/CSS/JavaScript**: Frontend UI with modern design
- **OpenAI/Azure/Anthropic APIs**: For enhanced response generation
- **Speech-to-Text Services**: For voice input processing
- **Docker**: For containerization and easy deployment
- **GitHub Actions**: For CI/CD pipeline

The system processes text in small chunks to provide a more responsive experience and handles errors gracefully to prevent crashes during speech generation.

## License

This project follows the CSM license terms (Apache 2.0 License) as per the original CSM repository.
