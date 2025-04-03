import os
import sys
import torch
import torchaudio
import time
import json
import re
import random
import traceback
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
from generator import load_csm_1b
import base64
import io
from openai_integration import get_llm_integration
from speech_to_text import process_audio_data
from dotenv import load_dotenv
from startup_utils import print_startup_message, initialize_app, get_system_info
from conversation_memory import get_conversation_memory
from analytics import track_user_message, track_assistant_response, track_session_start, track_session_end

# Load environment variables from .env file if it exists
load_dotenv()

# Set environment variable
os.environ["NO_TORCH_COMPILE"] = "1"

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Initialize global variables
generator = None
conversation_memory = get_conversation_memory()
llm_integration = None
speaker_id = int(os.getenv('SPEAKER_ID', 0))  # Get from environment or default to 0
session_start_times = {}  # Track session start times for analytics

# Simple responses for common questions  
CANNED_RESPONSES = {  
    "hello": ["Hello! How can I help you today?", "Hi there! What can I do for you?", "Greetings! How may I assist you?"],  
    "how are you": ["I'm doing well, thanks for asking! How about you?", "I'm functioning perfectly! How can I help?"],  
    "thank": ["You're welcome!", "Happy to help!", "My pleasure!"],  
    "bye": ["Goodbye! Have a great day!", "See you later!", "Take care!"],  
    "help": ["I can provide information, answer questions, or just chat. What would you like to know?"],  
    "weather": ["I don't have real-time weather data, but I'd be happy to discuss other topics."],  
    "name": ["I'm an AI voice assistant. You can call me Assistant."],  
    "what can you do": ["I can chat with you, answer questions, and provide information on various topics."],  
    "who created you": ["I was created as a voice assistant demo using Sesame AI's Controllable Speech Model for voice synthesis."],  
    "llm": ["LLMs or Large Language Models are AI systems trained on vast amounts of text data. They can understand and generate human-like text for various applications."],  
    "psd2": ["PSD2 stands for Payment Services Directive 2. It's a European regulation for electronic payment services designed to increase security, innovation, and competition in the banking sector."]  
}  

def load_models():  
    """Load the CSM model for voice"""  
    global generator  
    
    # Load CSM model for voice  
    if torch.cuda.is_available():  
        device = "cuda"  
    else:  
        device = "cpu"  

    print(f"Loading CSM on {device}...")  
    try:  
        generator = load_csm_1b(device=device)  
        
        # Initialize LLM integration
        global llm_integration
        llm_integration = get_llm_integration()
        print("CSM loaded successfully!")  
        
        # Test voice generation (following official example)  
        test_text = "This is a test."  # Smaller test for faster startup
        test_audio = generator.generate(  
            text=test_text,
            speaker=speaker_id,
            context=[],
            max_audio_length_ms=3000,  # Smaller test for faster startup
        )  
        print(f"Voice test successful! Generated {test_audio.shape[0]} samples")  
        
        return True  
    except Exception as e:  
        print(f"Error loading CSM model: {str(e)}")  
        traceback.print_exc()  
        return False  

def generate_response(message, user_id=None):  
    """Generate a response based on the input message"""  
    message = message.lower()  
    
    # Try to use LLM integration if available
    global llm_integration
    if llm_integration is not None:
        # Get conversation history from memory
        history = conversation_memory.get_history(user_id=user_id, limit=10)
        
        # Format history for LLM
        formatted_history = []
        for msg in history:
            formatted_history.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Generate response using LLM
        llm_response = llm_integration.generate_response(message, formatted_history)
        if llm_response:
            print(f"Using LLM response: {llm_response[:50]}...")
            return llm_response
    
    # Special case for welcome  
    if message == "welcome":  
        return "Welcome! I'm your voice assistant powered by Sesame AI's Controllable Speech Model. How can I help you today?"  
    
    # Check for keyword matches in canned responses  
    for keyword, responses in CANNED_RESPONSES.items():  
        if keyword in message:  
            return random.choice(responses)  
    
    # Handle questions  
    if "?" in message:  
        if any(word in message for word in ["what", "who", "how", "why", "when", "where"]):  
            return f"That's an interesting question about {message.strip('?')}. While I don't have external knowledge, I'd be happy to discuss this topic based on what you share."  
    
    # Handle greetings or statements  
    if len(message.split()) <= 3:  
        return "I'm listening. Please tell me more."  
    
    # Default responses based on message length  
    if len(message) < 20:  
        return "I understand. Could you elaborate a bit more so I can better assist you?"  
    else:  
        return f"I understand you're interested in {' '.join(message.split()[0:3])}. While I have limited knowledge, I'm here to chat and assist however I can."  

@app.route('/')  
def index():  
    return render_template('index_socketio.html')  

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/favicon.ico')  
def favicon():  
    # Return an empty response to prevent 404 errors  
    return Response("", content_type="image/x-icon")  

@socketio.on('connect')
def handle_connect():
    user_id = request.sid
    print(f"Client connected: {user_id}")
    
    # Track session start time for analytics
    session_start_times[user_id] = time.time()
    
    # Track session start event
    track_session_start(user_id)

@socketio.on('disconnect')
def handle_disconnect():
    user_id = request.sid
    print(f"Client disconnected: {user_id}")
    
    # Calculate session duration for analytics
    if user_id in session_start_times:
        session_duration = time.time() - session_start_times[user_id]
        track_session_end(session_duration, user_id)
        del session_start_times[user_id]

@socketio.on('user_message')
def handle_user_message(data):
    global speaker_id
    
    user_input = data.get('message', '')
    if not user_input:
        emit('error', {'content': 'No message provided'})
        return
    
    # Get user ID from session
    user_id = request.sid
    
    # Track user message for analytics
    track_user_message(user_input, user_id)
    
    # Add message to conversation memory
    conversation_memory.add_message("user", user_input, user_id=user_id)
    
    # Send acknowledgment that we're processing
    emit('message_received', {'status': 'processing'})
    
    try:
        # Record start time for response generation
        start_time = time.time()
        
        # Generate a response
        response = generate_response(user_input, user_id)
        
        # Add assistant message to conversation memory
        conversation_memory.add_message("assistant", response, user_id=user_id)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Track assistant response for analytics
        track_assistant_response(response, processing_time, user_id)
        
        # Send the full text immediately for UI display
        emit('response_text', {'text': response})
        
        # Generate natural speech chunks based on linguistic boundaries
        chunks = get_natural_speech_chunks(response)
        
        # Process each chunk for speech generation
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            
            # Generate speech for this chunk
            audio_data = text_to_speech(chunk)
            if audio_data:  # Only send if audio was generated successfully
                emit('speech_chunk', {'text': chunk.strip(), 'audio': audio_data})
            
            # Small delay for streaming effect
            time.sleep(0.05)
        
        # Mark processing as complete
        emit('processing_complete')
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        error_message = "I'm having trouble processing that right now."
        emit('error', {'text': error_message, 'details': str(e)})

@socketio.on('audio_message')
def handle_audio_message(data):
    """Handle audio input from the client"""
    global speaker_id
    
    audio_data = data.get('audio', '')
    if not audio_data:
        emit('error', {'content': 'No audio data provided'})
        return
    
    # Get user ID from session
    user_id = request.sid
    
    # Process the audio data to get the transcription
    transcription = process_audio_data(audio_data)
    
    if not transcription:
        emit('error', {'content': 'Failed to transcribe audio'})
        return
    
    # Send the transcription back to the client
    emit('transcription', {'text': transcription})
    
    # Track user message for analytics
    track_user_message(transcription, user_id)
    
    # Add message to conversation memory
    conversation_memory.add_message("user", transcription, user_id=user_id)
    
    # Send acknowledgment that we're processing
    emit('message_received', {'status': 'processing'})
    
    try:
        # Record start time for response generation
        start_time = time.time()
        
        # Generate a response
        response = generate_response(transcription, user_id)
        
        # Add assistant message to conversation memory
        conversation_memory.add_message("assistant", response, user_id=user_id)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Track assistant response for analytics
        track_assistant_response(response, processing_time, user_id)
        
        # Send the full text immediately for UI display
        emit('response_text', {'text': response})
        
        # Generate natural speech chunks based on linguistic boundaries
        chunks = get_natural_speech_chunks(response)
        
        # Process each chunk for speech generation
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            
            # Generate speech for this chunk
            audio_data = text_to_speech(chunk)
            if audio_data:  # Only send if audio was generated successfully
                emit('speech_chunk', {'text': chunk.strip(), 'audio': audio_data})
            
            # Small delay for streaming effect
            time.sleep(0.05)
        
        # Mark processing as complete
        emit('processing_complete')
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        error_message = "I'm having trouble processing that right now."
        emit('error', {'text': error_message, 'details': str(e)})

def get_natural_speech_chunks(text):
    """Split text into natural speech chunks based on linguistic boundaries"""
    # Get max chunk size from environment or use default
    max_chunk_size = int(os.getenv('CHUNK_SIZE', 30))  # Even smaller chunks for stability
    
    # First try to split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Process each sentence to avoid too long segments
    chunks = []
    for sentence in sentences:
        # For longer sentences, break on commas and other pauses
        if len(sentence) > max_chunk_size:
            clause_chunks = re.split(r'(?<=[,;:])\s+', sentence)
            for clause in clause_chunks:
                # Further break down if still too long
                if len(clause) > max_chunk_size:
                    words = clause.split()
                    if len(words) > 3:  # Only split if at least 3 words
                        midpoint = len(words) // 2
                        chunks.append(" ".join(words[:midpoint]))
                        chunks.append(" ".join(words[midpoint:]))
                    else:
                        chunks.append(clause)
                else:
                    chunks.append(clause)
        else:
            chunks.append(sentence)
    
    return chunks

def text_to_speech(text):  
    """Convert text to speech with CSM"""  
    global speaker_id, generator
    
    # Skip empty text  
    if not text.strip():  
        return ""  
    
    try:  
        print(f"Generating speech for: '{text}'")
        
        # Clean up CUDA memory before each generation to avoid memory leaks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Use shorter max_audio_length for stability
        max_length = int(os.getenv('MAX_AUDIO_LENGTH', 2000))
        
        # Generate audio using CSM - new generation for each chunk to avoid errors
        audio = generator.generate(  
            text=text,
            speaker=speaker_id,
            context=[],  # Always use empty context
            max_audio_length_ms=max_length
        )
        
        print(f"Speech generated: {audio.shape[0]} samples")  
        
        # Convert audio tensor to WAV format in memory  
        buffer = io.BytesIO()  
        torchaudio.save(buffer, audio.unsqueeze(0).cpu(), generator.sample_rate, format="wav")  
        buffer.seek(0)  
        
        # Encode as base64 for transmission  
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')  
        return audio_base64  
    except Exception as e:  
        print(f"Text-to-speech error: {str(e)}")  
        traceback.print_exc()  
        return ""  

@app.route('/clear-history', methods=['POST'])  
def clear_history():  
    """Clear conversation history"""  
    # Get user ID from session or use default
    user_id = request.cookies.get('user_id', 'anonymous')
    
    # Clear conversation memory
    conversation_memory.clear_history(user_id=user_id)
    
    return jsonify({"status": "success", "message": "Conversation history cleared"})  

@app.route('/change-voice', methods=['POST'])
def change_voice():
    """Change the voice speaker ID"""
    global speaker_id
    
    data = request.json
    new_speaker_id = data.get('speaker_id')
    
    if new_speaker_id is not None and 0 <= new_speaker_id <= 7:
        speaker_id = new_speaker_id
        return jsonify({"status": "success", "message": f"Voice changed to speaker {speaker_id}"})
    else:
        return jsonify({"status": "error", "message": "Invalid speaker ID. Must be between 0 and 7."})

@app.route('/status', methods=['GET'])
def status():
    """Get system status information"""
    system_info = get_system_info()
    
    # Add application-specific information
    system_info["app"] = {
        "conversation_memory": conversation_memory.get_summary(),
        "llm_integration": llm_integration.__class__.__name__ if llm_integration else "None",
        "speaker_id": speaker_id,
        "uptime": time.time() - system_info.get("timestamp", time.time()),
        "active_sessions": len(session_start_times)
    }
    
    return jsonify(system_info)

if __name__ == '__main__':
    # Initialize the application
    init_result = initialize_app()
    
    if init_result["status"] == "error":
        print(f"Error initializing application: {init_result['message']}")
        sys.exit(1)
    
    # Load models before starting the server
    if load_models():
        # Get configuration from environment variables
        debug_mode = os.getenv('DEBUG', 'True').lower() in ('true', '1', 't')
        host = os.getenv('HOST', '127.0.0.1')
        port = int(os.getenv('PORT', 5000))
        
        # Print startup message
        print_startup_message(host, port)
        
        # Run the server
        socketio.run(app, debug=debug_mode, host=host, port=port)
    else:
        print("Failed to load required models. Exiting.")
