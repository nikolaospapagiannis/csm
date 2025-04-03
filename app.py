import os
import torch
import torchaudio
import time
import json
import re
import random
import traceback
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_socketio import SocketIO, emit
from generator import load_csm_1b
import base64
import io
from openai_integration import get_llm_integration
from dotenv import load_dotenv
from startup_utils import print_startup_message, initialize_app, get_system_info
from speech_to_text import process_audio_data
from conversation_memory import get_conversation_memory

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
latest_audio = None  # Store most recent audio for context

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
        test_text = "This is a test of the speech system."
        test_audio = generator.generate(  
            text=test_text,
            speaker=speaker_id,
            context=[],
            max_audio_length_ms=5000,  
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
        history = conversation_memory.get_history(limit=10)
        
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
    return render_template('index.html')

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/favicon.ico')  
def favicon():  
    # Return an empty response to prevent 404 errors  
    return Response("", content_type="image/x-icon")  

@app.route('/chat', methods=['POST'])  
def chat():  
    data = request.json  
    user_input = data.get('message', '')  
    
    # Get user ID from session or use default
    user_id = request.cookies.get('user_id', 'anonymous')
    
    # Add message to conversation memory
    conversation_memory.add_message("user", user_input)
    
    return jsonify({"status": "processing"})  

@app.route('/stream', methods=['GET'])  
def stream():  
    user_input = request.args.get('message', '')  
    
    @stream_with_context  
    def generate():  
        global speaker_id, latest_audio  
        
        # Get user ID from session or use default
        user_id = request.cookies.get('user_id', 'anonymous')
        
        # Get the last user message if not provided  
        if not user_input:
            history = conversation_memory.get_history(limit=1)
            last_user_msg = history[0]["content"] if history and history[0]["role"] == "user" else None
        else:  
            last_user_msg = user_input  
            
        if not last_user_msg:  
            yield f"data: {json.dumps({'type': 'error', 'content': 'No user message found'})}\n\n"  
            return  
        
        try:  
            # Generate a response  
            response = generate_response(last_user_msg, user_id)  
            
            # Add assistant message to conversation memory
            conversation_memory.add_message("assistant", response)
            
            # Send the full text immediately for UI display  
            yield f"data: {json.dumps({'type': 'text', 'text': response})}\n\n"  
            
            # Generate natural speech chunks based on linguistic boundaries  
            chunks = get_natural_speech_chunks(response)  
            
            # Process each chunk for speech generation  
            for i, chunk in enumerate(chunks):  
                if not chunk.strip():  
                    continue  
                
                # Generate speech for this chunk  
                audio_data = text_to_speech(chunk, i==0)  
                if audio_data:  # Only send if audio was generated successfully  
                    yield f"data: {json.dumps({'type': 'speech', 'text': chunk.strip(), 'audio': audio_data})}\n\n"  
                
                # Short delay for streaming  
                time.sleep(0.05)  
            
            # Mark processing as complete  
            yield f"data: {json.dumps({'type': 'done'})}\n\n"  
            
        except Exception as e:  
            print(f"Error: {str(e)}")  
            traceback.print_exc()  
            error_message = "I'm having trouble processing that right now."  
            yield f"data: {json.dumps({'type': 'text', 'text': error_message})}\n\n"  
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"  
    
    return Response(generate(), mimetype='text/event-stream')  


@socketio.on('connect')
def handle_connect():
    print("Client connected")
    
@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

@socketio.on('user_message')
def handle_user_message(data):
    global speaker_id, latest_audio
    
    user_input = data.get('message', '')
    if not user_input:
        emit('error', {'content': 'No message provided'})
        return
    
    # Get user ID from session or use default
    user_id = request.sid
    
    # Add message to conversation memory
    conversation_memory.add_message("user", user_input)
    
    # Send acknowledgment that we're processing
    emit('message_received', {'status': 'processing'})
    
    try:
        # Generate a response
        response = generate_response(user_input, user_id)
        
        # Add assistant message to conversation memory
        conversation_memory.add_message("assistant", response)
        
        # Send the full text immediately for UI display
        emit('response_text', {'text': response})
        
        # Generate natural speech chunks based on linguistic boundaries
        chunks = get_natural_speech_chunks(response)
        
        # Process each chunk for speech generation
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            
            # Generate speech for this chunk
            audio_data = text_to_speech(chunk, i==0)
            if audio_data:  # Only send if audio was generated successfully
                emit('speech_chunk', {'text': chunk.strip(), 'audio': audio_data})
            
        # Mark processing as complete
        emit('processing_complete')
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        error_message = "I'm having trouble processing that right now."
        emit('error', {'text': error_message, 'details': str(e)})

@socketio.on('audio_message')
def handle_audio_message(data):
    global speaker_id, latest_audio
    
    audio_data = data.get('audio', '')
    if not audio_data:
        emit('error', {'content': 'No audio data provided'})
        return
    
    # Get user ID from session or use default
    user_id = request.sid
    
    # Send acknowledgment that we're processing
    emit('message_received', {'status': 'processing'})
    
    try:
        # Process audio data to get transcription
        transcription = process_audio_data(audio_data)
        
        if not transcription:
            emit('error', {'content': 'Could not transcribe audio'})
            return
        
        # Send transcription back to client
        emit('transcription', {'text': transcription})
        
        # Add message to conversation memory
        conversation_memory.add_message("user", transcription)
        
        # Generate a response
        response = generate_response(transcription, user_id)
        
        # Add assistant message to conversation memory
        conversation_memory.add_message("assistant", response)
        
        # Send the full text immediately for UI display
        emit('response_text', {'text': response})
        
        # Generate natural speech chunks based on linguistic boundaries
        chunks = get_natural_speech_chunks(response)
        
        # Process each chunk for speech generation
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            
            # Generate speech for this chunk
            audio_data = text_to_speech(chunk, i==0)
            if audio_data:  # Only send if audio was generated successfully
                emit('speech_chunk', {'text': chunk.strip(), 'audio': audio_data})
            
        # Mark processing as complete
        emit('processing_complete')
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        error_message = "I'm having trouble processing that right now."
        emit('error', {'text': error_message, 'details': str(e)})

def get_natural_speech_chunks(text):
    """Split text into natural speech chunks based on linguistic boundaries"""
    # Get max chunk size from environment or use default
    max_chunk_size = int(os.getenv('CHUNK_SIZE', 60))
    
    # First try to split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    for sentence in sentences:
        # For longer sentences, break on commas and other pauses
        if len(sentence) > max_chunk_size:
            clause_chunks = re.split(r'(?<=[,;:])\s+', sentence)
            chunks.extend(clause_chunks)
        else:
            chunks.append(sentence)
    
    # Ensure no chunk is too long
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chunk_size + 20:  # Allow a little buffer
            # Break long chunks at spaces, aiming for smaller character segments
            words = chunk.split()
            current_chunk = ""
            for word in words:
                if len(current_chunk) + len(word) + 1 > max_chunk_size:
                    final_chunks.append(current_chunk)
                    current_chunk = word
                else:
                    if current_chunk:
                        current_chunk += " " + word
                    else:
                        current_chunk = word
            if current_chunk:
                final_chunks.append(current_chunk)
        else:
            final_chunks.append(chunk)
    
    # Further split any chunks that might still be too long
    result = []
    for chunk in final_chunks:
        if len(chunk) > max_chunk_size:
            # Split at a reasonable word boundary if possible
            words = chunk.split()
            midpoint = len(words) // 2
            part1 = " ".join(words[:midpoint])
            part2 = " ".join(words[midpoint:])
            result.append(part1)
            result.append(part2)
        else:
            result.append(chunk)
    
    return result

def text_to_speech(text, is_first_chunk=False):  
    """Convert text to speech with CSM"""  
    global speaker_id, generator
    
    # Skip empty text  
    if not text.strip():  
        return ""  
    
    try:  
        print(f"Generating speech for: '{text}'")
        
        # Reset the generator for each chunk to avoid KV cache issues
        if hasattr(generator, '_model') and hasattr(generator._model, 'reset_kv_cache'):
            generator._model.reset_kv_cache()
        
        # Use shorter max_audio_length for stability
        max_length = min(int(os.getenv('MAX_AUDIO_LENGTH', 5000)), 5000)
        
        # Generate audio using CSM
        audio = generator.generate(  
            text=text,
            speaker=speaker_id,
            context=[],  # Empty context is always used
            max_audio_length_ms=max_length
        )
        
        print(f"Speech generated: {audio.shape} samples")  
        
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
    global latest_audio  
    
    # Get user ID from session or use default
    user_id = request.cookies.get('user_id', 'anonymous')
    
    # Clear conversation memory
    conversation_memory.clear_history()
    
    latest_audio = None  # Reset audio context too  
    return jsonify({"status": "success", "message": "Conversation history cleared"})  

@app.route('/change-voice', methods=['POST'])
def change_voice():
    """Change the voice speaker ID"""
    global speaker_id, latest_audio
    
    data = request.json
    new_speaker_id = data.get('speaker_id')
    
    if new_speaker_id is not None and 0 <= new_speaker_id <= 7:
        speaker_id = new_speaker_id
        latest_audio = None  # Reset audio context for the new voice
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
        "uptime": time.time() - system_info.get("timestamp", time.time())
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
