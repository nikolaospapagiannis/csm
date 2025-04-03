"""
OpenAI Integration Module for CSM Voice Chat Assistant

This module provides integration with OpenAI's API to enhance the assistant's
responses with more intelligent and contextually relevant content.
"""

import os
import json
import logging
import requests
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class OpenAIIntegration:
    """
    Integration with OpenAI's API for enhanced text generation.
    """
    
    def __init__(self):
        """Initialize the OpenAI integration"""
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        self.max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', 150))
        self.temperature = float(os.getenv('OPENAI_TEMPERATURE', 0.7))
        self.api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
        
        # Check if API key is available
        self.is_available = self.api_key is not None and self.api_key != ""
        
        if self.is_available:
            logger.info(f"OpenAI integration initialized with model: {self.model}")
        else:
            logger.warning("OpenAI API key not found. Integration will be disabled.")
    
    def generate_response(self, user_message: str, conversation_history: List[Dict[str, str]]) -> Optional[str]:
        """
        Generate a response using OpenAI's API.
        
        Args:
            user_message: The user's message
            conversation_history: List of previous messages in the conversation
            
        Returns:
            Generated response text or None if API call fails
        """
        if not self.is_available:
            logger.warning("OpenAI integration is not available. Skipping API call.")
            return None
        
        try:
            # Prepare the messages for the API call
            messages = []
            
            # Add system message
            messages.append({
                "role": "system", 
                "content": "You are a helpful, friendly voice assistant. Provide concise, informative responses."
            })
            
            # Add conversation history (limited to last 10 messages to avoid token limits)
            for msg in conversation_history[-10:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Add the current user message if not already in history
            if not conversation_history or conversation_history[-1]["role"] != "user" or conversation_history[-1]["content"] != user_message:
                messages.append({
                    "role": "user",
                    "content": user_message
                })
            
            # Make the API call
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=10
            )
            
            # Parse the response
            if response.status_code == 200:
                response_data = response.json()
                generated_text = response_data["choices"][0]["message"]["content"].strip()
                logger.info(f"OpenAI response generated: {len(generated_text)} chars")
                return generated_text
            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {str(e)}")
            return None
    
    def is_enabled(self) -> bool:
        """Check if the OpenAI integration is enabled"""
        return self.is_available


class AzureOpenAIIntegration(OpenAIIntegration):
    """
    Integration with Azure OpenAI's API for enhanced text generation.
    """
    
    def __init__(self):
        """Initialize the Azure OpenAI integration"""
        super().__init__()
        self.api_key = os.getenv('AZURE_OPENAI_KEY')
        self.api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-35-turbo')
        self.api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2023-05-15')
        
        # Check if API key and endpoint are available
        self.is_available = (self.api_key is not None and self.api_key != "" and 
                            self.api_base is not None and self.api_base != "")
        
        if self.is_available:
            logger.info(f"Azure OpenAI integration initialized with deployment: {self.deployment_name}")
        else:
            logger.warning("Azure OpenAI API key or endpoint not found. Integration will be disabled.")
    
    def generate_response(self, user_message: str, conversation_history: List[Dict[str, str]]) -> Optional[str]:
        """
        Generate a response using Azure OpenAI's API.
        
        Args:
            user_message: The user's message
            conversation_history: List of previous messages in the conversation
            
        Returns:
            Generated response text or None if API call fails
        """
        if not self.is_available:
            logger.warning("Azure OpenAI integration is not available. Skipping API call.")
            return None
        
        try:
            # Prepare the messages for the API call
            messages = []
            
            # Add system message
            messages.append({
                "role": "system", 
                "content": "You are a helpful, friendly voice assistant. Provide concise, informative responses."
            })
            
            # Add conversation history (limited to last 10 messages to avoid token limits)
            for msg in conversation_history[-10:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Add the current user message if not already in history
            if not conversation_history or conversation_history[-1]["role"] != "user" or conversation_history[-1]["content"] != user_message:
                messages.append({
                    "role": "user",
                    "content": user_message
                })
            
            # Make the API call
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
            
            payload = {
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            response = requests.post(
                f"{self.api_base}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}",
                headers=headers,
                data=json.dumps(payload),
                timeout=10
            )
            
            # Parse the response
            if response.status_code == 200:
                response_data = response.json()
                generated_text = response_data["choices"][0]["message"]["content"].strip()
                logger.info(f"Azure OpenAI response generated: {len(generated_text)} chars")
                return generated_text
            else:
                logger.error(f"Azure OpenAI API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating Azure OpenAI response: {str(e)}")
            return None


class AnthropicIntegration:
    """
    Integration with Anthropic's Claude API for enhanced text generation.
    """
    
    def __init__(self):
        """Initialize the Anthropic integration"""
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.model = os.getenv('ANTHROPIC_MODEL', 'claude-2')
        self.max_tokens = int(os.getenv('ANTHROPIC_MAX_TOKENS', 150))
        self.temperature = float(os.getenv('ANTHROPIC_TEMPERATURE', 0.7))
        
        # Check if API key is available
        self.is_available = self.api_key is not None and self.api_key != ""
        
        if self.is_available:
            logger.info(f"Anthropic integration initialized with model: {self.model}")
        else:
            logger.warning("Anthropic API key not found. Integration will be disabled.")
    
    def generate_response(self, user_message: str, conversation_history: List[Dict[str, str]]) -> Optional[str]:
        """
        Generate a response using Anthropic's API.
        
        Args:
            user_message: The user's message
            conversation_history: List of previous messages in the conversation
            
        Returns:
            Generated response text or None if API call fails
        """
        if not self.is_available:
            logger.warning("Anthropic integration is not available. Skipping API call.")
            return None
        
        try:
            # Format conversation history for Anthropic's API
            prompt = "\n\nHuman: You are a helpful, friendly voice assistant. Provide concise, informative responses.\n\nAssistant: I'll provide helpful, concise responses as a friendly voice assistant.\n\n"
            
            # Add conversation history (limited to last 10 messages to avoid token limits)
            for msg in conversation_history[-10:]:
                role = "Human" if msg["role"] == "user" else "Assistant"
                prompt += f"{role}: {msg['content']}\n\n"
            
            # Add the current user message if not already in history
            if not conversation_history or conversation_history[-1]["role"] != "user" or conversation_history[-1]["content"] != user_message:
                prompt += f"Human: {user_message}\n\n"
            
            prompt += "Assistant:"
            
            # Make the API call
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens_to_sample": self.max_tokens,
                "temperature": self.temperature
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/complete",
                headers=headers,
                data=json.dumps(payload),
                timeout=10
            )
            
            # Parse the response
            if response.status_code == 200:
                response_data = response.json()
                generated_text = response_data["completion"].strip()
                logger.info(f"Anthropic response generated: {len(generated_text)} chars")
                return generated_text
            else:
                logger.error(f"Anthropic API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating Anthropic response: {str(e)}")
            return None
    
    def is_enabled(self) -> bool:
        """Check if the Anthropic integration is enabled"""
        return self.is_available


# Factory function to get the appropriate LLM integration
def get_llm_integration() -> Optional[Any]:
    """
    Get the appropriate LLM integration based on available API keys.
    Prioritizes: OpenAI > Azure OpenAI > Anthropic > None
    
    Returns:
        An instance of the appropriate LLM integration class or None if no integration is available
    """
    # Try OpenAI first
    openai_integration = OpenAIIntegration()
    if openai_integration.is_enabled():
        return openai_integration
    
    # Try Azure OpenAI next
    azure_integration = AzureOpenAIIntegration()
    if azure_integration.is_enabled():
        return azure_integration
    
    # Try Anthropic next
    anthropic_integration = AnthropicIntegration()
    if anthropic_integration.is_enabled():
        return anthropic_integration
    
    # No integration available
    logger.warning("No LLM integration available. Using default responses.")
    return None