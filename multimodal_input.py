"""
Multi-modal Input Module for CSM Voice Chat Assistant

This module provides support for processing multi-modal inputs such as images,
integrating with OpenAI Vision API, Azure Computer Vision, and local models.
"""

import os
import json
import logging
import time
import base64
import io
from typing import Dict, Any, List, Optional, Tuple, Union
import requests
from PIL import Image
import numpy as np
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class MultiModalProcessor:
    """Base class for multi-modal input processors"""
    
    def __init__(self):
        """Initialize the multi-modal processor"""
        pass
    
    def process_image(self, image_data: bytes) -> Optional[Dict[str, Any]]:
        """
        Process an image
        
        Args:
            image_data: Raw image data in bytes
            
        Returns:
            Dictionary with processing results or None if processing fails
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def enhance_prompt(self, prompt: str, image_data: bytes) -> str:
        """
        Enhance a prompt with image description
        
        Args:
            prompt: Original prompt
            image_data: Raw image data in bytes
            
        Returns:
            Enhanced prompt
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def is_enabled(self) -> bool:
        """Check if the processor is enabled"""
        raise NotImplementedError("Subclasses must implement this method")


class OpenAIVisionProcessor(MultiModalProcessor):
    """OpenAI Vision API processor for multi-modal inputs"""
    
    def __init__(self):
        """Initialize the OpenAI Vision processor"""
        super().__init__()
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
        self.model = os.getenv('OPENAI_VISION_MODEL', 'gpt-4-vision-preview')
        self.max_tokens = int(os.getenv('OPENAI_VISION_MAX_TOKENS', 300))
        
        # Check if API key is available
        self.is_available = self.api_key is not None and self.api_key != ""
        
        if self.is_available:
            logger.info(f"OpenAI Vision processor initialized with model: {self.model}")
        else:
            logger.warning("OpenAI API key not found. OpenAI Vision processor will be disabled.")
    
    def is_enabled(self) -> bool:
        """Check if the processor is enabled"""
        return self.is_available
    
    def process_image(self, image_data: bytes) -> Optional[Dict[str, Any]]:
        """
        Process an image using OpenAI's Vision API
        
        Args:
            image_data: Raw image data in bytes
            
        Returns:
            Dictionary with processing results or None if processing fails
        """
        if not self.is_enabled():
            logger.warning("OpenAI Vision processor is not enabled. Skipping image processing.")
            return None
        
        try:
            # Encode image as base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare the API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in detail. Include information about objects, people, scenes, colors, text, and any other relevant details."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": self.max_tokens
            }
            
            # Make the API call
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # Parse the response
            if response.status_code == 200:
                response_data = response.json()
                
                # Extract the description
                description = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                if description:
                    logger.info(f"Image processed successfully: {description[:50]}...")
                    
                    return {
                        "description": description,
                        "model": self.model,
                        "provider": "openai"
                    }
                else:
                    logger.warning("Empty description returned from OpenAI Vision API")
                    return None
            else:
                logger.error(f"OpenAI Vision API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error processing image with OpenAI Vision API: {str(e)}")
            return None
    
    def enhance_prompt(self, prompt: str, image_data: bytes) -> str:
        """
        Enhance a prompt with image description
        
        Args:
            prompt: Original prompt
            image_data: Raw image data in bytes
            
        Returns:
            Enhanced prompt
        """
        if not self.is_enabled():
            return prompt
        
        try:
            # Process the image
            result = self.process_image(image_data)
            
            if result and "description" in result:
                # Enhance the prompt with the image description
                enhanced_prompt = f"{prompt}\n\nImage description: {result['description']}"
                logger.info(f"Enhanced prompt with image description: {enhanced_prompt[:100]}...")
                return enhanced_prompt
            else:
                return prompt
        except Exception as e:
            logger.error(f"Error enhancing prompt with image description: {str(e)}")
            return prompt


class AzureVisionProcessor(MultiModalProcessor):
    """Azure Computer Vision processor for multi-modal inputs"""
    
    def __init__(self):
        """Initialize the Azure Computer Vision processor"""
        super().__init__()
        self.api_key = os.getenv('AZURE_VISION_KEY')
        self.endpoint = os.getenv('AZURE_VISION_ENDPOINT')
        
        # Check if API key and endpoint are available
        self.is_available = (self.api_key is not None and self.api_key != "" and 
                            self.endpoint is not None and self.endpoint != "")
        
        if self.is_available:
            logger.info(f"Azure Computer Vision processor initialized with endpoint: {self.endpoint}")
        else:
            logger.warning("Azure Vision API key or endpoint not found. Azure Vision processor will be disabled.")
    
    def is_enabled(self) -> bool:
        """Check if the processor is enabled"""
        return self.is_available
    
    def process_image(self, image_data: bytes) -> Optional[Dict[str, Any]]:
        """
        Process an image using Azure Computer Vision
        
        Args:
            image_data: Raw image data in bytes
            
        Returns:
            Dictionary with processing results or None if processing fails
        """
        if not self.is_enabled():
            logger.warning("Azure Vision processor is not enabled. Skipping image processing.")
            return None
        
        try:
            # Prepare the API request
            analyze_url = f"{self.endpoint}/vision/v3.2/analyze"
            headers = {
                "Content-Type": "application/octet-stream",
                "Ocp-Apim-Subscription-Key": self.api_key
            }
            
            params = {
                "visualFeatures": "Categories,Description,Objects,Tags",
                "language": "en",
                "model-version": "latest"
            }
            
            # Make the API call
            response = requests.post(
                analyze_url,
                headers=headers,
                params=params,
                data=image_data,
                timeout=30
            )
            
            # Parse the response
            if response.status_code == 200:
                response_data = response.json()
                
                # Extract the description
                captions = response_data.get("description", {}).get("captions", [])
                description = captions[0].get("text") if captions else ""
                
                # Extract tags
                tags = response_data.get("tags", [])
                tag_names = [tag.get("name") for tag in tags if tag.get("confidence", 0) > 0.5]
                
                # Extract objects
                objects = response_data.get("objects", [])
                object_names = [obj.get("object") for obj in objects if obj.get("confidence", 0) > 0.5]
                
                if description:
                    logger.info(f"Image processed successfully: {description}")
                    
                    # Combine all information
                    full_description = f"{description}. "
                    
                    if tag_names:
                        full_description += f"Tags: {', '.join(tag_names)}. "
                    
                    if object_names:
                        full_description += f"Objects: {', '.join(object_names)}."
                    
                    return {
                        "description": full_description,
                        "raw_description": description,
                        "tags": tag_names,
                        "objects": object_names,
                        "provider": "azure"
                    }
                else:
                    logger.warning("Empty description returned from Azure Computer Vision")
                    return None
            else:
                logger.error(f"Azure Computer Vision error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error processing image with Azure Computer Vision: {str(e)}")
            return None
    
    def enhance_prompt(self, prompt: str, image_data: bytes) -> str:
        """
        Enhance a prompt with image description
        
        Args:
            prompt: Original prompt
            image_data: Raw image data in bytes
            
        Returns:
            Enhanced prompt
        """
        if not self.is_enabled():
            return prompt
        
        try:
            # Process the image
            result = self.process_image(image_data)
            
            if result and "description" in result:
                # Enhance the prompt with the image description
                enhanced_prompt = f"{prompt}\n\nImage description: {result['description']}"
                logger.info(f"Enhanced prompt with image description: {enhanced_prompt[:100]}...")
                return enhanced_prompt
            else:
                return prompt
        except Exception as e:
            logger.error(f"Error enhancing prompt with image description: {str(e)}")
            return prompt


class LocalVisionProcessor(MultiModalProcessor):
    """Local vision model processor for multi-modal inputs"""
    
    def __init__(self):
        """Initialize the Local Vision processor"""
        super().__init__()
        self.model = None
        self.processor = None
        
        # Check if we should use the local model
        use_local = os.getenv('USE_LOCAL_VISION', 'False').lower() in ('true', '1', 't')
        self.is_available = use_local
        
        if self.is_available:
            logger.info("Local Vision processor will be initialized on first use")
        else:
            logger.info("Local Vision processor is disabled. Set USE_LOCAL_VISION=True to enable.")
    
    def is_enabled(self) -> bool:
        """Check if the processor is enabled"""
        return self.is_available
    
    def _load_model(self) -> bool:
        """
        Load the vision model
        
        Returns:
            True if model was loaded successfully, False otherwise
        """
        try:
            # Import here to avoid dependency issues if not using local vision
            from transformers import AutoProcessor, AutoModelForCausalLM
            
            # Load model and processor
            self.processor = AutoProcessor.from_pretrained("microsoft/git-base")
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
            
            logger.info("Local Vision model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading local Vision model: {str(e)}")
            self.is_available = False
            return False
    
    def process_image(self, image_data: bytes) -> Optional[Dict[str, Any]]:
        """
        Process an image using local vision model
        
        Args:
            image_data: Raw image data in bytes
            
        Returns:
            Dictionary with processing results or None if processing fails
        """
        if not self.is_enabled():
            logger.warning("Local Vision processor is not enabled. Skipping image processing.")
            return None
        
        try:
            # Load model if not already loaded
            if self.model is None or self.processor is None:
                if not self._load_model():
                    return None
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Prepare inputs
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Generate caption
            generated_ids = self.model.generate(
                pixel_values=inputs["pixel_values"],
                max_length=50,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode caption
            generated_caption = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
            
            if generated_caption:
                logger.info(f"Image processed successfully: {generated_caption}")
                
                return {
                    "description": generated_caption,
                    "provider": "local"
                }
            else:
                logger.warning("Empty description returned from local Vision model")
                return None
        except Exception as e:
            logger.error(f"Error processing image with local Vision model: {str(e)}")
            return None
    
    def enhance_prompt(self, prompt: str, image_data: bytes) -> str:
        """
        Enhance a prompt with image description
        
        Args:
            prompt: Original prompt
            image_data: Raw image data in bytes
            
        Returns:
            Enhanced prompt
        """
        if not self.is_enabled():
            return prompt
        
        try:
            # Process the image
            result = self.process_image(image_data)
            
            if result and "description" in result:
                # Enhance the prompt with the image description
                enhanced_prompt = f"{prompt}\n\nImage description: {result['description']}"
                logger.info(f"Enhanced prompt with image description: {enhanced_prompt[:100]}...")
                return enhanced_prompt
            else:
                return prompt
        except Exception as e:
            logger.error(f"Error enhancing prompt with image description: {str(e)}")
            return prompt


def get_multimodal_processor() -> MultiModalProcessor:
    """
    Get the appropriate multi-modal processor based on available options.
    Prioritizes: OpenAI Vision > Azure Vision > Local Vision
    
    Returns:
        An instance of the appropriate multi-modal processor
    """
    # Try OpenAI Vision first
    openai_processor = OpenAIVisionProcessor()
    if openai_processor.is_enabled():
        return openai_processor
    
    # Try Azure Vision next
    azure_processor = AzureVisionProcessor()
    if azure_processor.is_enabled():
        return azure_processor
    
    # Try Local Vision last
    local_processor = LocalVisionProcessor()
    if local_processor.is_enabled():
        return local_processor
    
    # No processor available, return a disabled processor
    logger.warning("No multi-modal processor available.")
    return MultiModalProcessor()


def process_image(image_base64: str) -> Optional[Dict[str, Any]]:
    """
    Process an image from the web interface
    
    Args:
        image_base64: Base64-encoded image data
        
    Returns:
        Dictionary with processing results or None if processing fails
    """
    try:
        # Decode base64 image data
        image_data = base64.b64decode(image_base64)
        
        # Get the appropriate multi-modal processor
        processor = get_multimodal_processor()
        
        # Process the image
        return processor.process_image(image_data)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None


def enhance_prompt_with_image(prompt: str, image_base64: str) -> str:
    """
    Enhance a prompt with image description
    
    Args:
        prompt: Original prompt
        image_base64: Base64-encoded image data
        
    Returns:
        Enhanced prompt
    """
    try:
        # Decode base64 image data
        image_data = base64.b64decode(image_base64)
        
        # Get the appropriate multi-modal processor
        processor = get_multimodal_processor()
        
        # Enhance the prompt
        return processor.enhance_prompt(prompt, image_data)
        
    except Exception as e:
        logger.error(f"Error enhancing prompt with image: {str(e)}")
        return prompt