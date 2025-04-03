"""
Unit tests for the OpenAI integration module.
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import os
from openai_integration import OpenAIIntegration, AzureOpenAIIntegration, AnthropicIntegration, get_llm_integration

class TestOpenAIIntegration(unittest.TestCase):
    """Test cases for the OpenAI integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Save original environment variables
        self.original_env = {}
        for key in ['OPENAI_API_KEY', 'OPENAI_MODEL', 'OPENAI_MAX_TOKENS', 'OPENAI_TEMPERATURE']:
            self.original_env[key] = os.environ.get(key)
        
        # Set test environment variables
        os.environ['OPENAI_API_KEY'] = 'test_api_key'
        os.environ['OPENAI_MODEL'] = 'gpt-3.5-turbo'
        os.environ['OPENAI_MAX_TOKENS'] = '100'
        os.environ['OPENAI_TEMPERATURE'] = '0.7'
    
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
        """Test initialization of OpenAI integration."""
        integration = OpenAIIntegration()
        self.assertTrue(integration.is_available)
        self.assertEqual(integration.api_key, 'test_api_key')
        self.assertEqual(integration.model, 'gpt-3.5-turbo')
        self.assertEqual(integration.max_tokens, 100)
        self.assertEqual(integration.temperature, 0.7)
    
    def test_is_enabled(self):
        """Test is_enabled method."""
        integration = OpenAIIntegration()
        self.assertTrue(integration.is_enabled())
        
        # Test with no API key
        with patch.dict(os.environ, {'OPENAI_API_KEY': ''}):
            integration = OpenAIIntegration()
            self.assertFalse(integration.is_enabled())
    
    @patch('requests.post')
    def test_generate_response_success(self, mock_post):
        """Test successful response generation."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This is a test response."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        integration = OpenAIIntegration()
        conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        response = integration.generate_response("How are you?", conversation_history)
        
        self.assertEqual(response, "This is a test response.")
        mock_post.assert_called_once()
        
        # Check that the correct data was sent
        call_args = mock_post.call_args
        payload = json.loads(call_args[1]['data'])
        self.assertEqual(payload['model'], 'gpt-3.5-turbo')
        self.assertEqual(payload['max_tokens'], 100)
        self.assertEqual(payload['temperature'], 0.7)
        
        # Check that the messages were formatted correctly
        messages = payload['messages']
        self.assertEqual(len(messages), 4)  # system + 2 history + 1 current
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[1]['role'], 'user')
        self.assertEqual(messages[1]['content'], 'Hello')
        self.assertEqual(messages[2]['role'], 'assistant')
        self.assertEqual(messages[2]['content'], 'Hi there!')
        self.assertEqual(messages[3]['role'], 'user')
        self.assertEqual(messages[3]['content'], 'How are you?')
    
    @patch('requests.post')
    def test_generate_response_error(self, mock_post):
        """Test error handling in response generation."""
        # Mock error API response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Error message"
        mock_post.return_value = mock_response
        
        integration = OpenAIIntegration()
        response = integration.generate_response("Test message", [])
        
        self.assertIsNone(response)
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_generate_response_exception(self, mock_post):
        """Test exception handling in response generation."""
        # Mock exception during API call
        mock_post.side_effect = Exception("Test exception")
        
        integration = OpenAIIntegration()
        response = integration.generate_response("Test message", [])
        
        self.assertIsNone(response)
        mock_post.assert_called_once()


class TestAzureOpenAIIntegration(unittest.TestCase):
    """Test cases for the Azure OpenAI integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Save original environment variables
        self.original_env = {}
        for key in ['AZURE_OPENAI_KEY', 'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_DEPLOYMENT', 'AZURE_OPENAI_API_VERSION']:
            self.original_env[key] = os.environ.get(key)
        
        # Set test environment variables
        os.environ['AZURE_OPENAI_KEY'] = 'test_api_key'
        os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://test-endpoint.openai.azure.com'
        os.environ['AZURE_OPENAI_DEPLOYMENT'] = 'gpt-35-turbo'
        os.environ['AZURE_OPENAI_API_VERSION'] = '2023-05-15'
    
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
        """Test initialization of Azure OpenAI integration."""
        integration = AzureOpenAIIntegration()
        self.assertTrue(integration.is_available)
        self.assertEqual(integration.api_key, 'test_api_key')
        self.assertEqual(integration.api_base, 'https://test-endpoint.openai.azure.com')
        self.assertEqual(integration.deployment_name, 'gpt-35-turbo')
        self.assertEqual(integration.api_version, '2023-05-15')
    
    def test_is_enabled(self):
        """Test is_enabled method."""
        integration = AzureOpenAIIntegration()
        self.assertTrue(integration.is_enabled())
        
        # Test with no API key
        with patch.dict(os.environ, {'AZURE_OPENAI_KEY': ''}):
            integration = AzureOpenAIIntegration()
            self.assertFalse(integration.is_enabled())
        
        # Test with no endpoint
        with patch.dict(os.environ, {'AZURE_OPENAI_ENDPOINT': ''}):
            integration = AzureOpenAIIntegration()
            self.assertFalse(integration.is_enabled())


class TestAnthropicIntegration(unittest.TestCase):
    """Test cases for the Anthropic integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Save original environment variables
        self.original_env = {}
        for key in ['ANTHROPIC_API_KEY', 'ANTHROPIC_MODEL', 'ANTHROPIC_MAX_TOKENS', 'ANTHROPIC_TEMPERATURE']:
            self.original_env[key] = os.environ.get(key)
        
        # Set test environment variables
        os.environ['ANTHROPIC_API_KEY'] = 'test_api_key'
        os.environ['ANTHROPIC_MODEL'] = 'claude-2'
        os.environ['ANTHROPIC_MAX_TOKENS'] = '100'
        os.environ['ANTHROPIC_TEMPERATURE'] = '0.7'
    
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
        """Test initialization of Anthropic integration."""
        integration = AnthropicIntegration()
        self.assertTrue(integration.is_available)
        self.assertEqual(integration.api_key, 'test_api_key')
        self.assertEqual(integration.model, 'claude-2')
        self.assertEqual(integration.max_tokens, 100)
        self.assertEqual(integration.temperature, 0.7)
    
    def test_is_enabled(self):
        """Test is_enabled method."""
        integration = AnthropicIntegration()
        self.assertTrue(integration.is_enabled())
        
        # Test with no API key
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': ''}):
            integration = AnthropicIntegration()
            self.assertFalse(integration.is_enabled())
    
    @patch('requests.post')
    def test_generate_response_success(self, mock_post):
        """Test successful response generation."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "completion": "This is a test response."
        }
        mock_post.return_value = mock_response
        
        integration = AnthropicIntegration()
        conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        response = integration.generate_response("How are you?", conversation_history)
        
        self.assertEqual(response, "This is a test response.")
        mock_post.assert_called_once()


class TestLLMIntegrationFactory(unittest.TestCase):
    """Test cases for the LLM integration factory function."""
    
    def setUp(self):
        """Set up test environment."""
        # Save original environment variables
        self.original_env = {}
        for key in ['OPENAI_API_KEY', 'AZURE_OPENAI_KEY', 'AZURE_OPENAI_ENDPOINT', 'ANTHROPIC_API_KEY']:
            self.original_env[key] = os.environ.get(key)
        
        # Clear all API keys
        for key in ['OPENAI_API_KEY', 'AZURE_OPENAI_KEY', 'AZURE_OPENAI_ENDPOINT', 'ANTHROPIC_API_KEY']:
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
    
    def test_get_openai_integration(self):
        """Test getting OpenAI integration."""
        os.environ['OPENAI_API_KEY'] = 'test_api_key'
        
        integration = get_llm_integration()
        
        self.assertIsInstance(integration, OpenAIIntegration)
    
    def test_get_azure_openai_integration(self):
        """Test getting Azure OpenAI integration."""
        os.environ['AZURE_OPENAI_KEY'] = 'test_api_key'
        os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://test-endpoint.openai.azure.com'
        
        integration = get_llm_integration()
        
        self.assertIsInstance(integration, AzureOpenAIIntegration)
    
    def test_get_anthropic_integration(self):
        """Test getting Anthropic integration."""
        os.environ['ANTHROPIC_API_KEY'] = 'test_api_key'
        
        integration = get_llm_integration()
        
        self.assertIsInstance(integration, AnthropicIntegration)
    
    def test_get_no_integration(self):
        """Test getting no integration when no API keys are available."""
        integration = get_llm_integration()
        
        self.assertIsNone(integration)
    
    def test_integration_priority(self):
        """Test integration priority (OpenAI > Azure > Anthropic)."""
        # Set all API keys
        os.environ['OPENAI_API_KEY'] = 'test_api_key'
        os.environ['AZURE_OPENAI_KEY'] = 'test_api_key'
        os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://test-endpoint.openai.azure.com'
        os.environ['ANTHROPIC_API_KEY'] = 'test_api_key'
        
        integration = get_llm_integration()
        
        # Should prioritize OpenAI
        self.assertIsInstance(integration, OpenAIIntegration)
        
        # Remove OpenAI key
        del os.environ['OPENAI_API_KEY']
        integration = get_llm_integration()
        
        # Should now use Azure
        self.assertIsInstance(integration, AzureOpenAIIntegration)
        
        # Remove Azure keys
        del os.environ['AZURE_OPENAI_KEY']
        del os.environ['AZURE_OPENAI_ENDPOINT']
        integration = get_llm_integration()
        
        # Should now use Anthropic
        self.assertIsInstance(integration, AnthropicIntegration)


if __name__ == '__main__':
    unittest.main()