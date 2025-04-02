"""
This script fixes the Flask app to properly support Socket.IO
It should be run to update the existing app.py file
"""

import os
import sys
import re

def fix_socketio():
    """Update app.py to correctly implement Socket.IO"""
    # Check if app.py exists
    if not os.path.exists('app.py'):
        print("Error: app.py not found!")
        return False
    
    # Read the current content
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Add required imports if missing
    if 'from flask_socketio import SocketIO' not in content:
        content = re.sub(
            r'from flask import (.*)',
            r'from flask import \1\nfrom flask_socketio import SocketIO, emit',
            content
        )
    
    # Update app initialization
    content = re.sub(
        r'app = Flask\(__name__\)',
        r'app = Flask(__name__)\nsocketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")',
        content
    )
    
    # Update the main entry point
    content = re.sub(
        r'app\.run\(debug=(.*), host=(.*), port=(.*), threaded=(.*)\)',
        r'socketio.run(app, debug=\1, host=\2, port=\3)',
        content
    )
    
    # Replace standard event handling with Socket.IO
    # Find the stream route implementation
    stream_route_match = re.search(r'@app\.route\(\'/stream\'.*?def stream\(\):.*?return Response\(generate\(\), mimetype=\'text/event-stream\'\)(.*?)(\n\n|\Z)', content, re.DOTALL)
    
    if stream_route_match:
        # Create SocketIO event handlers for the chat
        socketio_handler = """
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
"""
        # Insert the SocketIO event handlers after the original stream route
        content = content.replace(stream_route_match.group(0), stream_route_match.group(0) + socketio_handler)
    
    # Update the client-side code as well if templates/index.html exists
    if os.path.exists('templates/index.html'):
        try:
            with open('templates/index.html', 'r') as f:
                html_content = f.read()
            
            # Update Socket.IO client connection
            if "<script src=\"https://cdn.socket.io/4.6.0/socket.io.min.js\"></script>" not in html_content:
                html_content = html_content.replace(
                    "</head>",
                    "<script src=\"https://cdn.socket.io/4.6.0/socket.io.min.js\"></script>\n</head>"
                )
            
            # Update the JavaScript to use Socket.IO
            script_update = """
<script>
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const statusIndicator = document.getElementById('status-indicator');
    const clearButton = document.getElementById('clear-button');
    const voiceSelect = document.getElementById('voice-select');
    const notification = document.getElementById('notification');
    
    let isProcessing = false;
    let audioQueue = [];
    let isPlaying = false;
    let currentAssistantDiv = null;
    let socket = null;
    
    // Connect to Socket.IO
    function connectSocket() {
        socket = io();
        
        socket.on('connect', () => {
            console.log('Connected to server');
            showNotification('Connected to server', false);
        });
        
        socket.on('disconnect', () => {
            console.log('Disconnected from server');
            showNotification('Disconnected from server', true);
        });
        
        socket.on('message_received', (data) => {
            console.log('Message received:', data);
            statusIndicator.innerHTML = 'AI is thinking <div class="typing-animation"><span></span><span></span><span></span></div>';
        });
        
        socket.on('response_text', (data) => {
            // Update the text of the current assistant message
            if (currentAssistantDiv) {
                currentAssistantDiv.textContent = data.text;
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        });
        
        socket.on('speech_chunk', (data) => {
            // Add audio to the queue and play if there's audio data
            if (data.audio) {
                const audio = new Audio('data:audio/wav;base64,' + data.audio);
                audioQueue.push(audio);
                
                if (!isPlaying) {
                    playNextAudio();
                }
            }
        });
        
        socket.on('processing_complete', () => {
            isProcessing = false;
            statusIndicator.textContent = '';
        });
        
        socket.on('error', (data) => {
            console.error('Error from server:', data);
            statusIndicator.textContent = '';
            if (currentAssistantDiv && currentAssistantDiv.textContent === '') {
                currentAssistantDiv.textContent = data.text || 'Sorry, an error occurred.';
            }
            showNotification(data.text || 'Error occurred', true);
            isProcessing = false;
        });
    }
    
    // Set up event listeners
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    clearButton.addEventListener('click', clearChat);
    
    // Voice selection handler
    voiceSelect.addEventListener('change', function() {
        const voiceId = parseInt(this.value);
        
        fetch('/change-voice', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ speaker_id: voiceId })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                showNotification(`Voice changed to Voice ${voiceId + 1}`, false);
            } else {
                showNotification(data.message || 'Error changing voice', true);
            }
        })
        .catch(error => {
            console.error('Error changing voice:', error);
            showNotification('Failed to change voice', true);
        });
    });
    
    function clearChat() {
        fetch('/clear-history', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                chatContainer.innerHTML = '';
                showWelcomeMessage();
                showNotification('Chat history cleared', false);
            }
        })
        .catch(error => {
            console.error('Error clearing chat:', error);
        });
    }
    
    function showNotification(message, isError) {
        notification.textContent = message;
        notification.className = 'notification' + (isError ? ' error' : '');
        notification.classList.add('show');
        
        setTimeout(() => {
            notification.classList.remove('show');
        }, 3000);
    }
    
    function sendMessage() {
        if (isProcessing) return;
        
        const message = userInput.value.trim();
        if (!message) return;
        
        // Display user message
        appendMessage(message, 'user');
        userInput.value = '';
        
        isProcessing = true;
        
        // Create assistant message container
        const assistantContainer = document.createElement('div');
        assistantContainer.style.width = '100%';
        assistantContainer.style.display = 'inline-block';
        assistantContainer.style.clear = 'both';
        chatContainer.appendChild(assistantContainer);
        
        currentAssistantDiv = document.createElement('div');
        currentAssistantDiv.className = 'message assistant';
        assistantContainer.appendChild(currentAssistantDiv);
        
        // Add a clearfix element after the message
        const clearfix = document.createElement('div');
        clearfix.className = 'clearfix';
        assistantContainer.appendChild(clearfix);
        
        chatContainer.scrollTop = chatContainer.scrollHeight;
        
        // Send message via Socket.IO
        socket.emit('user_message', { message: message });
    }
    
    function playNextAudio() {
        if (audioQueue.length === 0) {
            isPlaying = false;
            
            // If we're done playing audio and done processing, clear status
            if (!isProcessing) {
                statusIndicator.textContent = '';
            }
            return;
        }
        
        isPlaying = true;
        const audio = audioQueue.shift();
        
        audio.onended = function() {
            // Immediate transition for fluent speech
            playNextAudio();
        };
        
        audio.play().catch(error => {
            console.error('Audio playback error:', error);
            playNextAudio(); // Try next audio if one fails
        });
    }
    
    function appendMessage(text, sender) {
        const messageContainer = document.createElement('div');
        messageContainer.style.width = '100%';
        messageContainer.style.display = 'inline-block';
        messageContainer.style.clear = 'both';
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        messageDiv.textContent = text;
        
        messageContainer.appendChild(messageDiv);
        
        // Add a clearfix element after the message
        const clearfix = document.createElement('div');
        clearfix.className = 'clearfix';
        messageContainer.appendChild(clearfix);
        
        chatContainer.appendChild(messageContainer);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    // Display a welcome message on page load
    function showWelcomeMessage() {
        const welcomeText = "Welcome! I'm your voice assistant powered by Sesame AI's Controllable Speech Model. How can I help you today?";
        
        // Create assistant message container
        const assistantContainer = document.createElement('div');
        assistantContainer.style.width = '100%';
        assistantContainer.style.display = 'inline-block';
        assistantContainer.style.clear = 'both';
        chatContainer.appendChild(assistantContainer);
        
        currentAssistantDiv = document.createElement('div');
        currentAssistantDiv.className = 'message assistant';
        currentAssistantDiv.textContent = welcomeText;
        assistantContainer.appendChild(currentAssistantDiv);
        
        // Add a clearfix element after the message
        const clearfix = document.createElement('div');
        clearfix.className = 'clearfix';
        assistantContainer.appendChild(clearfix);
        
        // Send welcome message via Socket.IO - with a small delay
        setTimeout(() => {
            socket.emit('user_message', { message: "welcome" });
        }, 500);
    }
    
    // Initialize
    window.addEventListener('load', function() {
        // Connect to Socket.IO
        connectSocket();
        
        // Focus on input field
        userInput.focus();
        
        // Show welcome message after a brief delay to ensure socket connection
        setTimeout(() => {
            showWelcomeMessage();
        }, 300);
    });
</script>
"""
            
            # Replace the existing script
            html_content = re.sub(
                r'<script>.*?</script>',
                script_update,
                html_content,
                flags=re.DOTALL
            )
            
            with open('templates/index.html', 'w') as f:
                f.write(html_content)
            
            print("Updated templates/index.html with Socket.IO client code")
            
        except Exception as e:
            print(f"Error updating templates/index.html: {str(e)}")
    
    # Write the updated app.py
    with open('app.py', 'w') as f:
        f.write(content)
    
    print("Successfully updated app.py with improved Socket.IO support")
    return True

if __name__ == "__main__":
    fix_socketio()
