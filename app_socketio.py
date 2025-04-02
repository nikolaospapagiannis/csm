import os
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
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Set environment variable
os.environ["NO_TORCH_COMPILE"] = "1"

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Initialize global variables
generator = None
conversation_history = []
speaker_id = int(os.getenv('SPEAKER_ID', 0))  # Get from environment or default to 0

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

def generate_response(message):  
    """Generate a simple response based on the input message"""  
    message = message.lower()  
    
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

@app.route('/favicon.ico')  
def favicon():  
    # Return an empty response to prevent 404 errors  
    return Response("", content_type="image/x-icon")  

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    
@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

@socketio.on('user_message')
def handle_user_message(data):
    global speaker_id
    
    user_input = data.get('message', '')
    if not user_input:
        emit('error', {'content': 'No message provided'})
        return
    
    # Update conversation history with user message
    conversation_history.append({"role": "user", "content": user_input})
    
    # Send acknowledgment that we're processing
    emit('message_received', {'status': 'processing'})
    
    try:
        # Generate a response
        response = generate_response(user_input)
        
        # Add assistant message to history
        assistant_msg = {"role": "assistant", "content": response}
        conversation_history.append(assistant_msg)
        
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
    global conversation_history
    conversation_history = []
    return jsonify({"status": "success", "message": "Conversation history cleared"})  

@app.route('/change-voice', methods=['POST'])
def change_voice():
    """Change the voice speaker ID"""
    global speaker_id
    
    data = request.json
    new_speaker_id = data.get('speaker_id')
    
    if new_speaker_id is not None and 0 <= new_speaker_id <= 13:
        speaker_id = new_speaker_id
        return jsonify({"status": "success", "message": f"Voice changed to speaker {speaker_id}"})
    else:
        return jsonify({"status": "error", "message": "Invalid speaker ID. Must be between 0 and 13."})

if __name__ == '__main__':
    # Load models before starting the server
    if load_models():
        # Get configuration from environment variables
        debug_mode = os.getenv('DEBUG', 'True').lower() in ('true', '1', 't')
        host = os.getenv('HOST', '127.0.0.1')
        port = int(os.getenv('PORT', 5000))
        
        print(f"Starting server with voice ID {speaker_id}...")
        print(f"The web interface will be available at: http://{host}:{port}")
        socketio.run(app, debug=debug_mode, host=host, port=port)
    else:
        print("Failed to load required models. Exiting.")
