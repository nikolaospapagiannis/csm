"""
Speech-to-Text Module for CSM Voice Chat Assistant

This module provides speech-to-text functionality to enable voice input
for the assistant, supporting multiple providers including OpenAI Whisper,
Azure Speech Service, and local Whisper model.
"""

import os
import json
import logging
import tempfile
import base64
import io
from typing import Optional, Dict, Any, List
import requests
import torch
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SpeechToTextService:
    """Base class for speech-to-text services"""
    
    def __init__(self):
        """Initialize the speech-to-text service"""
        self.is_available = False
    
    def transcribe(self, audio_data: bytes) -> Optional[str]:
        """
        Transcribe audio data to text
        
        Args:
            audio_data: Raw audio data in bytes
            
        Returns:
            Transcribed text or None if transcription fails
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def is_enabled(self) -> bool:
        """Check if the service is enabled"""
        return self.is_available


class WhisperAPIService(SpeechToTextService):
    """OpenAI Whisper API service for speech-to-text"""
    
    def __init__(self):
        """Initialize the Whisper API service"""
        super().__init__()
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
        self.model = os.getenv('WHISPER_MODEL', 'whisper-1')
        
        # Check if API key is available
        self.is_available = self.api_key is not None and self.api_key != ""
        
        if self.is_available:
            logger.info(f"Whisper API service initialized with model: {self.model}")
        else:
            logger.warning("OpenAI API key not found. Whisper API service will be disabled.")
    
    def is_enabled(self) -> bool:
        """Check if the service is enabled"""
        return self.is_available and self.api_key != ""
    
    def transcribe(self, audio_data: bytes) -> Optional[str]:
        """
        Transcribe audio data using OpenAI's Whisper API
        
        Args:
            audio_data: Raw audio data in bytes
            
        Returns:
            Transcribed text or None if transcription fails
        """
        if not self.is_enabled():
            logger.warning("Whisper API service is not enabled. Skipping transcription.")
            return None
        
        try:
            # Prepare the API request
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Create a temporary file for the audio data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                temp_file.write(audio_data)
                temp_file.flush()
                
                # Open the file in binary mode
                with open(temp_file.name, "rb") as audio_file:
                    # Make the API call
                    response = requests.post(
                        f"{self.api_base}/audio/transcriptions",
                        headers=headers,
                        files={"file": audio_file},
                        data={"model": self.model},
                        timeout=30
                    )
            
            # Parse the response
            if response.status_code == 200:
                response_data = response.json()
                transcription = response_data.get("text", "").strip()
                
                if transcription:
                    logger.info(f"Audio transcribed successfully: {transcription[:50]}...")
                    return transcription
                else:
                    logger.warning("Empty transcription returned from Whisper API")
                    return None
            else:
                logger.error(f"Whisper API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error transcribing audio with Whisper API: {str(e)}")
            return None


class AzureSpeechService(SpeechToTextService):
    """Azure Speech Service for speech-to-text"""
    
    def __init__(self):
        """Initialize the Azure Speech Service"""
        super().__init__()
        self.api_key = os.getenv('AZURE_SPEECH_KEY')
        self.region = os.getenv('AZURE_SPEECH_REGION')
        self.language = os.getenv('AZURE_SPEECH_LANGUAGE', 'en-US')
        
        # Check if API key and region are available
        self.is_available = (self.api_key is not None and self.api_key != "" and 
                            self.region is not None and self.region != "")
        
        if self.is_available:
            logger.info(f"Azure Speech Service initialized with region: {self.region}")
        else:
            logger.warning("Azure Speech API key or region not found. Azure Speech Service will be disabled.")
    
    def is_enabled(self) -> bool:
        """Check if the service is enabled"""
        return self.is_available and self.api_key != "" and self.region != ""
    
    def transcribe(self, audio_data: bytes) -> Optional[str]:
        """
        Transcribe audio data using Azure Speech Service
        
        Args:
            audio_data: Raw audio data in bytes
            
        Returns:
            Transcribed text or None if transcription fails
        """
        if not self.is_enabled():
            logger.warning("Azure Speech Service is not enabled. Skipping transcription.")
            return None
        
        try:
            # Prepare the API request
            url = f"https://{self.region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
            headers = {
                "Ocp-Apim-Subscription-Key": self.api_key,
                "Content-Type": "audio/wav",
                "Accept": "application/json"
            }
            
            params = {
                "language": self.language,
                "format": "detailed"
            }
            
            # Make the API call
            response = requests.post(
                url,
                headers=headers,
                params=params,
                data=audio_data,
                timeout=30
            )
            
            # Parse the response
            if response.status_code == 200:
                response_data = response.json()
                
                # Extract the recognized text
                if response_data.get("RecognitionStatus") == "Success":
                    transcription = response_data.get("DisplayText", "").strip()
                    
                    if transcription:
                        logger.info(f"Audio transcribed successfully with Azure: {transcription[:50]}...")
                        return transcription
                    else:
                        logger.warning("Empty transcription returned from Azure Speech Service")
                        return None
                else:
                    logger.warning(f"Azure Speech recognition failed: {response_data.get('RecognitionStatus')}")
                    return None
            else:
                logger.error(f"Azure Speech Service error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error transcribing audio with Azure Speech Service: {str(e)}")
            return None


class LocalWhisperService(SpeechToTextService):
    """Local Whisper model service for speech-to-text"""
    
    def __init__(self):
        """Initialize the Local Whisper service"""
        super().__init__()
        self.model = None
        self.model_size = os.getenv('LOCAL_WHISPER_MODEL', 'base')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check if we should use the local model
        use_local = os.getenv('USE_LOCAL_WHISPER', 'False').lower() in ('true', '1', 't')
        self.is_available = use_local
        
        if self.is_available:
            logger.info(f"Local Whisper service will be initialized with model: {self.model_size} on {self.device}")
        else:
            logger.info("Local Whisper service is disabled. Set USE_LOCAL_WHISPER=True to enable.")
    
    def is_enabled(self) -> bool:
        """Check if the service is enabled"""
        return self.is_available
    
    def load_model(self) -> bool:
        """
        Load the Whisper model
        
        Returns:
            True if model was loaded successfully, False otherwise
        """
        try:
            # Import here to avoid dependency issues if not using local whisper
            import whisper
            
            # Load the model
            self.model = whisper.load_model(self.model_size, device=self.device)
            
            logger.info(f"Local Whisper model '{self.model_size}' loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Error loading local Whisper model: {str(e)}")
            self.is_available = False
            return False
    
    def transcribe(self, audio_data: bytes) -> Optional[str]:
        """
        Transcribe audio data using local Whisper model
        
        Args:
            audio_data: Raw audio data in bytes
            
        Returns:
            Transcribed text or None if transcription fails
        """
        if not self.is_enabled():
            logger.warning("Local Whisper service is not enabled. Skipping transcription.")
            return None
        
        try:
            # Load model if not already loaded
            if self.model is None:
                if not self.load_model():
                    return None
            
            # Import here to avoid dependency issues if not using local whisper
            import numpy as np
            import soundfile as sf
            
            # Convert audio bytes to numpy array
            with io.BytesIO(audio_data) as audio_io:
                audio_array, sample_rate = sf.read(audio_io)
                
                # Convert to mono if stereo
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.mean(axis=1)
            
            # Transcribe audio
            result = self.model.transcribe(audio_array, fp16=False)
            transcription = result.get("text", "").strip()
            
            if transcription:
                logger.info(f"Audio transcribed successfully with local Whisper: {transcription[:50]}...")
                return transcription
            else:
                logger.warning("Empty transcription returned from local Whisper model")
                return None
        except Exception as e:
            logger.error(f"Error transcribing audio with local Whisper model: {str(e)}")
            return None


def get_speech_to_text_service() -> SpeechToTextService:
    """
    Get the appropriate speech-to-text service based on available options.
    Prioritizes: Whisper API > Azure Speech > Local Whisper
    
    Returns:
        An instance of the appropriate speech-to-text service
    """
    # Try Whisper API first
    whisper_service = WhisperAPIService()
    if whisper_service.is_enabled():
        return whisper_service
    
    # Try Azure Speech next
    azure_service = AzureSpeechService()
    if azure_service.is_enabled():
        return azure_service
    
    # Try Local Whisper last
    local_service = LocalWhisperService()
    if local_service.is_enabled():
        return local_service
    
    # No service available, return a disabled service
    logger.warning("No speech-to-text service available.")
    return SpeechToTextService()


def process_audio_data(audio_base64: str) -> Optional[str]:
    """
    Process audio data from the web interface
    
    Args:
        audio_base64: Base64-encoded audio data
        
    Returns:
        Transcribed text or None if transcription fails
    """
    try:
        # Decode base64 audio data
        audio_data = base64.b64decode(audio_base64)
        
        # Get the appropriate speech-to-text service
        service = get_speech_to_text_service()
        
        # Transcribe the audio
        return service.transcribe(audio_data)
        
    except Exception as e:
        logger.error(f"Error processing audio data: {str(e)}")
        return None