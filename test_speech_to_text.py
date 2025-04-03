"""
Unit tests for the speech-to-text module.
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import os
import base64
import tempfile
from speech_to_text import (
    SpeechToTextService, 
    WhisperAPIService, 
    AzureSpeechService, 
    LocalWhisperService, 
    get_speech_to_text_service,
    process_audio_data
)

class TestSpeechToTextBase(unittest.TestCase):
    """Test cases for the base SpeechToTextService class."""
    
    def test_initialization(self):
        """Test initialization of base service."""
        service = SpeechToTextService()
        self.assertFalse(service.is_available)
        self.assertFalse(service.is_enabled())
    
    def test_transcribe_not_implemented(self):
        """Test that transcribe method raises NotImplementedError."""
        service = SpeechToTextService()
        with self.assertRaises(NotImplementedError):
            service.transcribe(b"test audio data")


class TestWhisperAPIService(unittest.TestCase):
    """Test cases for the WhisperAPIService class."""
    
    def setUp(self):
        """Set up test environment."""
        # Save original environment variables
        self.original_env = {}
        for key in ['OPENAI_API_KEY', 'OPENAI_API_BASE', 'WHISPER_MODEL']:
            self.original_env[key] = os.environ.get(key)
        
        # Set test environment variables
        os.environ['OPENAI_API_KEY'] = 'test_api_key'
        os.environ['OPENAI_API_BASE'] = 'https://api.openai.com/v1'
        os.environ['WHISPER_MODEL'] = 'whisper-1'
    
    def tearDown(self):
        """Tear down test environment."""
        # Restore original environment variables
        for key, value in self.original_env.items():
            if value is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = value
    
    def test_initialization(self):
        """Test initialization of WhisperAPIService."""
        service = WhisperAPIService()
        self.assertTrue(service.is_available)
        self.assertEqual(service.api_key, 'test_api_key')
        self.assertEqual(service.api_base, 'https://api.openai.com/v1')
        self.assertEqual(service.model, 'whisper-1')
    
    def test_is_enabled(self):
        """Test is_enabled method."""
        service = WhisperAPIService()
        self.assertTrue(service.is_enabled())
        
        # Test with no API key
        with patch.dict(os.environ, {'OPENAI_API_KEY': ''}):
            service = WhisperAPIService()
            self.assertFalse(service.is_enabled())
    
    @patch('requests.post')
    def test_transcribe_success(self, mock_post):
        """Test successful transcription."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "This is a test transcription."
        }
        mock_post.return_value = mock_response
        
        service = WhisperAPIService()
        
        # Create a simple audio file for testing
        audio_data = b"test audio data"
        
        response = service.transcribe(audio_data)
        
        self.assertEqual(response, "This is a test transcription.")
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_transcribe_error(self, mock_post):
        """Test error handling in transcription."""
        # Mock error API response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Error message"
        mock_post.return_value = mock_response
        
        service = WhisperAPIService()
        response = service.transcribe(b"test audio data")
        
        self.assertIsNone(response)
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_transcribe_exception(self, mock_post):
        """Test exception handling in transcription."""
        # Mock exception during API call
        mock_post.side_effect = Exception("Test exception")
        
        service = WhisperAPIService()
        response = service.transcribe(b"test audio data")
        
        self.assertIsNone(response)
        mock_post.assert_called_once()


class TestAzureSpeechService(unittest.TestCase):
    """Test cases for the AzureSpeechService class."""
    
    def setUp(self):
        """Set up test environment."""
        # Save original environment variables
        self.original_env = {}
        for key in ['AZURE_SPEECH_KEY', 'AZURE_SPEECH_REGION', 'AZURE_SPEECH_LANGUAGE']:
            self.original_env[key] = os.environ.get(key)
        
        # Set test environment variables
        os.environ['AZURE_SPEECH_KEY'] = 'test_api_key'
        os.environ['AZURE_SPEECH_REGION'] = 'westus'
        os.environ['AZURE_SPEECH_LANGUAGE'] = 'en-US'
    
    def tearDown(self):
        """Tear down test environment."""
        # Restore original environment variables
        for key, value in self.original_env.items():
            if value is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = value
    
    def test_initialization(self):
        """Test initialization of AzureSpeechService."""
        service = AzureSpeechService()
        self.assertTrue(service.is_available)
        self.assertEqual(service.api_key, 'test_api_key')
        self.assertEqual(service.region, 'westus')
        self.assertEqual(service.language, 'en-US')
    
    def test_is_enabled(self):
        """Test is_enabled method."""
        service = AzureSpeechService()
        self.assertTrue(service.is_enabled())
        
        # Test with no API key
        with patch.dict(os.environ, {'AZURE_SPEECH_KEY': ''}):
            service = AzureSpeechService()
            self.assertFalse(service.is_enabled())
        
        # Test with no region
        with patch.dict(os.environ, {'AZURE_SPEECH_REGION': ''}):
            service = AzureSpeechService()
            self.assertFalse(service.is_enabled())
    
    @patch('requests.post')
    def test_transcribe_success(self, mock_post):
        """Test successful transcription."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "DisplayText": "This is a test transcription."
        }
        mock_post.return_value = mock_response
        
        service = AzureSpeechService()
        
        # Create a simple audio file for testing
        audio_data = b"test audio data"
        
        response = service.transcribe(audio_data)
        
        self.assertEqual(response, "This is a test transcription.")
        mock_post.assert_called_once()


class TestLocalWhisperService(unittest.TestCase):
    """Test cases for the LocalWhisperService class."""
    
    def setUp(self):
        """Set up test environment."""
        # Save original environment variables
        self.original_env = {}
        for key in ['USE_LOCAL_WHISPER', 'LOCAL_WHISPER_MODEL']:
            self.original_env[key] = os.environ.get(key)
        
        # Set test environment variables
        os.environ['USE_LOCAL_WHISPER'] = 'True'
        os.environ['LOCAL_WHISPER_MODEL'] = 'base'
    
    def tearDown(self):
        """Tear down test environment."""
        # Restore original environment variables
        for key, value in self.original_env.items():
            if value is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = value
    
    def test_initialization(self):
        """Test initialization of LocalWhisperService."""
        service = LocalWhisperService()
        self.assertTrue(service.is_available)
        self.assertEqual(service.model_size, 'base')
        self.assertIn(service.device, ['cuda', 'cpu'])
        self.assertIsNone(service.model)
    
    def test_is_enabled(self):
        """Test is_enabled method."""
        service = LocalWhisperService()
        self.assertTrue(service.is_enabled())
        
        # Test with USE_LOCAL_WHISPER set to False
        with patch.dict(os.environ, {'USE_LOCAL_WHISPER': 'False'}):
            service = LocalWhisperService()
            self.assertFalse(service.is_enabled())
    
    @patch('whisper.load_model')
    def test_load_model(self, mock_load_model):
        """Test loading the Whisper model."""
        # Mock the whisper.load_model function
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        service = LocalWhisperService()
        result = service.load_model()
        
        self.assertTrue(result)
        self.assertEqual(service.model, mock_model)
        mock_load_model.assert_called_once_with('base', device=service.device)
    
    @patch('whisper.load_model')
    def test_load_model_exception(self, mock_load_model):
        """Test exception handling when loading the model."""
        # Mock an exception when loading the model
        mock_load_model.side_effect = Exception("Test exception")
        
        service = LocalWhisperService()
        result = service.load_model()
        
        self.assertFalse(result)
        self.assertFalse(service.is_available)
        self.assertIsNone(service.model)
        mock_load_model.assert_called_once()


class TestSpeechToTextFactory(unittest.TestCase):
    """Test cases for the speech-to-text factory function."""
    
    def setUp(self):
        """Set up test environment."""
        # Save original environment variables
        self.original_env = {}
        for key in ['OPENAI_API_KEY', 'AZURE_SPEECH_KEY', 'AZURE_SPEECH_REGION', 'USE_LOCAL_WHISPER']:
            self.original_env[key] = os.environ.get(key)
        
        # Clear all API keys
        for key in ['OPENAI_API_KEY', 'AZURE_SPEECH_KEY', 'AZURE_SPEECH_REGION', 'USE_LOCAL_WHISPER']:
            if key in os.environ:
                del os.environ[key]
    
    def tearDown(self):
        """Tear down test environment."""
        # Restore original environment variables
        for key, value in self.original_env.items():
            if value is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = value
    
    def test_get_whisper_api_service(self):
        """Test getting WhisperAPIService."""
        os.environ['OPENAI_API_KEY'] = 'test_api_key'
        
        service = get_speech_to_text_service()
        
        self.assertIsInstance(service, WhisperAPIService)
    
    def test_get_azure_speech_service(self):
        """Test getting AzureSpeechService."""
        os.environ['AZURE_SPEECH_KEY'] = 'test_api_key'
        os.environ['AZURE_SPEECH_REGION'] = 'westus'
        
        service = get_speech_to_text_service()
        
        self.assertIsInstance(service, AzureSpeechService)
    
    def test_get_local_whisper_service(self):
        """Test getting LocalWhisperService."""
        os.environ['USE_LOCAL_WHISPER'] = 'True'
        
        service = get_speech_to_text_service()
        
        self.assertIsInstance(service, LocalWhisperService)
    
    def test_get_base_service(self):
        """Test getting base service when no API keys are available."""
        service = get_speech_to_text_service()
        
        self.assertIsInstance(service, SpeechToTextService)
    
    def test_service_priority(self):
        """Test service priority (Whisper API > Azure Speech > Local Whisper)."""
        # Set all API keys
        os.environ['OPENAI_API_KEY'] = 'test_api_key'
        os.environ['AZURE_SPEECH_KEY'] = 'test_api_key'
        os.environ['AZURE_SPEECH_REGION'] = 'westus'
        os.environ['USE_LOCAL_WHISPER'] = 'True'
        
        service = get_speech_to_text_service()
        
        # Should prioritize Whisper API
        self.assertIsInstance(service, WhisperAPIService)
        
        # Remove OpenAI key
        del os.environ['OPENAI_API_KEY']
        service = get_speech_to_text_service()
        
        # Should now use Azure Speech
        self.assertIsInstance(service, AzureSpeechService)
        
        # Remove Azure keys
        del os.environ['AZURE_SPEECH_KEY']
        del os.environ['AZURE_SPEECH_REGION']
        service = get_speech_to_text_service()
        
        # Should now use Local Whisper
        self.assertIsInstance(service, LocalWhisperService)


class TestProcessAudioData(unittest.TestCase):
    """Test cases for the process_audio_data function."""
    
    @patch('speech_to_text.get_speech_to_text_service')
    def test_process_audio_data_success(self, mock_get_service):
        """Test successful audio data processing."""
        # Mock the speech-to-text service
        mock_service = MagicMock()
        mock_service.transcribe.return_value = "This is a test transcription."
        mock_get_service.return_value = mock_service
        
        # Create a base64-encoded audio string
        audio_base64 = base64.b64encode(b"test audio data").decode('utf-8')
        
        result = process_audio_data(audio_base64)
        
        self.assertEqual(result, "This is a test transcription.")
        mock_service.transcribe.assert_called_once_with(b"test audio data")
    
    @patch('speech_to_text.get_speech_to_text_service')
    def test_process_audio_data_failure(self, mock_get_service):
        """Test handling of transcription failure."""
        # Mock the speech-to-text service
        mock_service = MagicMock()
        mock_service.transcribe.return_value = None
        mock_get_service.return_value = mock_service
        
        # Create a base64-encoded audio string
        audio_base64 = base64.b64encode(b"test audio data").decode('utf-8')
        
        result = process_audio_data(audio_base64)
        
        self.assertIsNone(result)
        mock_service.transcribe.assert_called_once_with(b"test audio data")
    
    @patch('speech_to_text.get_speech_to_text_service')
    def test_process_audio_data_exception(self, mock_get_service):
        """Test exception handling in audio data processing."""
        # Mock an exception during transcription
        mock_get_service.side_effect = Exception("Test exception")
        
        # Create a base64-encoded audio string
        audio_base64 = base64.b64encode(b"test audio data").decode('utf-8')
        
        result = process_audio_data(audio_base64)
        
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()