"""
Unit and Integration Tests for Moshi Integration Module

This module contains comprehensive tests for the Moshi integration with CSM,
which integrates with Kyutai's Mimi audio codec and Moshi dialogue framework.

Test coverage includes:
1. Unit tests for individual components
2. Integration tests for the full pipeline
3. Performance tests for latency measurement
4. Error handling and edge cases
"""

import unittest
import torch
import time
import threading
import os
import sys
import logging
from unittest.mock import MagicMock, patch
from typing import List, Tuple, Optional

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the module to test
from moshi_integration import (
    MoshiConfig, 
    MimiCodec, 
    MoshiModel, 
    FullDuplexDialogueSystem,
    DummyMimiEncoder,
    DummyMimiDecoder,
    DummyDepthTransformer,
    DummyTemporalTransformer
)


class TestMoshiConfig(unittest.TestCase):
    """Tests for the MoshiConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = MoshiConfig()
        self.assertEqual(config.sample_rate, 24000)
        self.assertEqual(config.frame_size_ms, 80)
        self.assertEqual(config.stream_buffer_size, 10)
        self.assertEqual(config.max_monologue_tokens, 64)
        self.assertEqual(config.sampling_rate, 12.5)
        self.assertEqual(config.bandwidth_kbps, 1.1)
        self.assertEqual(config.temporal_model_size, "7B")
        self.assertEqual(config.latency_target_ms, 200)
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = MoshiConfig(
            sample_rate=16000,
            frame_size_ms=40,
            stream_buffer_size=5,
            max_monologue_tokens=32,
            device="cpu",
            sampling_rate=25.0,
            bandwidth_kbps=2.0,
            temporal_model_size="1B",
            latency_target_ms=100
        )
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.frame_size_ms, 40)
        self.assertEqual(config.stream_buffer_size, 5)
        self.assertEqual(config.max_monologue_tokens, 32)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.sampling_rate, 25.0)
        self.assertEqual(config.bandwidth_kbps, 2.0)
        self.assertEqual(config.temporal_model_size, "1B")
        self.assertEqual(config.latency_target_ms, 100)


class TestMimiCodec(unittest.TestCase):
    """Tests for the MimiCodec class"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = MoshiConfig(device="cpu")  # Force CPU for tests
        self.codec = MimiCodec(self.config)
    
    def test_initialization(self):
        """Test codec initialization"""
        self.assertIsNotNone(self.codec)
        self.assertEqual(self.codec.config.sample_rate, 24000)
        self.assertFalse(self.codec.is_initialized)
        self.assertIsNone(self.codec.encoder)
        self.assertIsNone(self.codec.decoder)
    
    def test_load_models(self):
        """Test model loading"""
        success = self.codec.load_models()
        self.assertTrue(success)
        self.assertTrue(self.codec.is_initialized)
        self.assertIsNotNone(self.codec.encoder)
        self.assertIsNotNone(self.codec.decoder)
    
    def test_encode_decode(self):
        """Test encoding and decoding"""
        # Generate dummy audio (1 second at 24kHz)
        dummy_audio = torch.randn(1, 24000)
        
        # Test encode
        encoded = self.codec.encode(dummy_audio)
        self.assertIsNotNone(encoded)
        self.assertEqual(len(encoded.shape), 3)  # [batch, frames, features]
        
        # Test decode
        decoded = self.codec.decode(encoded)
        self.assertIsNotNone(decoded)
        self.assertEqual(len(decoded.shape), 2)  # [batch, samples]


class TestMoshiModel(unittest.TestCase):
    """Tests for the MoshiModel class"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = MoshiConfig(device="cpu")  # Force CPU for tests
        self.mimi = MimiCodec(self.config)
        self.mimi.load_models()
        self.moshi = MoshiModel(self.config, self.mimi)
    
    def test_initialization(self):
        """Test model initialization"""
        self.assertIsNotNone(self.moshi)
        self.assertEqual(self.moshi.config.sample_rate, 24000)
        self.assertFalse(self.moshi.is_initialized)
        self.assertIsNone(self.moshi.depth_transformer)
        self.assertIsNone(self.moshi.temporal_transformer)
        self.assertEqual(len(self.moshi.system_stream), 0)
        self.assertEqual(len(self.moshi.user_stream), 0)
        self.assertEqual(len(self.moshi.monologue_tokens), 0)
    
    def test_load_models(self):
        """Test model loading"""
        success = self.moshi.load_models()
        self.assertTrue(success)
        self.assertTrue(self.moshi.is_initialized)
        self.assertIsNotNone(self.moshi.depth_transformer)
        self.assertIsNotNone(self.moshi.temporal_transformer)
    
    def test_process_user_audio(self):
        """Test processing user audio"""
        # Load models first
        self.moshi.load_models()
        
        # Generate dummy audio (1 second at 24kHz)
        dummy_audio = torch.randn(1, 24000)
        
        # Process audio
        self.moshi.process_user_audio(dummy_audio)
        
        # Verify user stream was updated
        self.assertGreater(len(self.moshi.user_stream), 0)
    
    def test_generate_response(self):
        """Test response generation"""
        # Load models first
        self.moshi.load_models()
        
        # Generate dummy audio (1 second at 24kHz)
        dummy_audio = torch.randn(1, 24000)
        
        # Process audio
        self.moshi.process_user_audio(dummy_audio)
        
        # Generate response
        audio_response, text_monologue = self.moshi.generate_response()
        
        # Verify response
        self.assertIsNotNone(audio_response)
        self.assertGreater(audio_response.numel(), 0)
        self.assertIsNotNone(text_monologue)
        self.assertGreater(len(text_monologue), 0)
        
        # Verify system stream was updated
        self.assertGreater(len(self.moshi.system_stream), 0)
    
    def test_reset_state(self):
        """Test state reset"""
        # Load models first
        self.moshi.load_models()
        
        # Generate dummy audio (1 second at 24kHz)
        dummy_audio = torch.randn(1, 24000)
        
        # Process audio and generate response
        self.moshi.process_user_audio(dummy_audio)
        self.moshi.generate_response()
        
        # Verify streams have data
        self.assertGreater(len(self.moshi.user_stream), 0)
        self.assertGreater(len(self.moshi.system_stream), 0)
        
        # Reset state
        self.moshi.reset_state()
        
        # Verify streams are empty
        self.assertEqual(len(self.moshi.user_stream), 0)
        self.assertEqual(len(self.moshi.system_stream), 0)
        self.assertEqual(len(self.moshi.monologue_tokens), 0)


class TestFullDuplexDialogueSystem(unittest.TestCase):
    """Tests for the FullDuplexDialogueSystem class"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = MoshiConfig(device="cpu")  # Force CPU for tests
        self.system = FullDuplexDialogueSystem(self.config)
    
    def tearDown(self):
        """Clean up after tests"""
        if self.system.is_running:
            self.system.stop()
    
    def test_initialization(self):
        """Test system initialization"""
        self.assertIsNotNone(self.system)
        self.assertEqual(self.system.config.sample_rate, 24000)
        self.assertFalse(self.system.is_running)
        self.assertIsNone(self.system.processing_thread)
    
    def test_initialize_components(self):
        """Test initializing all components"""
        success = self.system.initialize()
        self.assertTrue(success)
        self.assertTrue(self.system.mimi.is_initialized)
        self.assertTrue(self.system.moshi.is_initialized)
    
    def test_start_stop(self):
        """Test starting and stopping the system"""
        # Initialize first
        self.system.initialize()
        
        # Start
        self.system.start()
        self.assertTrue(self.system.is_running)
        self.assertIsNotNone(self.system.processing_thread)
        self.assertTrue(self.system.processing_thread.is_alive())
        
        # Stop
        self.system.stop()
        self.assertFalse(self.system.is_running)
        time.sleep(0.2)  # Give thread time to terminate
        self.assertFalse(self.system.processing_thread.is_alive())
    
    def test_feed_audio_get_response(self):
        """Test feeding audio and getting response"""
        # Initialize and start
        self.system.initialize()
        self.system.start()
        
        # Generate dummy audio (1 second at 24kHz)
        dummy_audio = torch.randn(1, 24000)
        
        # Feed audio
        self.system.feed_audio(dummy_audio)
        
        # Wait for processing
        time.sleep(0.5)
        
        # Get response
        response = self.system.get_response()
        
        # Verify response
        self.assertIsNotNone(response)
        audio, text = response
        self.assertIsNotNone(audio)
        self.assertGreater(audio.numel(), 0)
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)
    
    def test_reset(self):
        """Test resetting the system"""
        # Initialize and start
        self.system.initialize()
        self.system.start()
        
        # Generate dummy audio (1 second at 24kHz)
        dummy_audio = torch.randn(1, 24000)
        
        # Feed audio
        self.system.feed_audio(dummy_audio)
        
        # Wait for processing
        time.sleep(0.5)
        
        # Reset
        self.system.reset()
        
        # Verify state
        self.assertEqual(len(self.system.moshi.user_stream), 0)
        self.assertEqual(len(self.system.moshi.system_stream), 0)
        
        # Try to get response (should be empty after reset)
        response = self.system.get_response()
        self.assertIsNone(response)


class TestPerformance(unittest.TestCase):
    """Performance tests for the full system"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = MoshiConfig(device="cpu")  # Force CPU for tests
        self.system = FullDuplexDialogueSystem(self.config)
        self.system.initialize()
        self.system.start()
    
    def tearDown(self):
        """Clean up after tests"""
        if self.system.is_running:
            self.system.stop()
    
    def test_latency(self):
        """Test end-to-end latency"""
        # Generate dummy audio (1 second at 24kHz)
        dummy_audio = torch.randn(1, 24000)
        
        # Measure time for processing
        start_time = time.time()
        
        # Feed audio
        self.system.feed_audio(dummy_audio)
        
        # Wait for and get response
        response = None
        max_wait = 2.0  # Maximum wait time in seconds
        elapsed = 0.0
        poll_interval = 0.01
        
        while response is None and elapsed < max_wait:
            response = self.system.get_response()
            if response is None:
                time.sleep(poll_interval)
                elapsed += poll_interval
        
        end_time = time.time()
        
        # Calculate latency
        latency_ms = (end_time - start_time) * 1000
        
        # Verify response was received
        self.assertIsNotNone(response, "No response received within timeout")
        
        # Log latency
        logger.info(f"End-to-end latency: {latency_ms:.2f} ms")
        
        # Assert latency is within reasonable bounds
        # Note: CPU tests will be slower, so we're being lenient here
        self.assertLess(latency_ms, 1000)  # Should be under 1 second
    
    def test_streaming_performance(self):
        """Test performance with continuous streaming"""
        # Number of audio chunks to test
        num_chunks = 10
        chunk_size = 2400  # 100ms at 24kHz
        
        # Generate dummy audio chunks
        audio_chunks = [torch.randn(1, chunk_size) for _ in range(num_chunks)]
        
        # Track responses
        responses_received = 0
        
        # Start feeding audio and collecting responses
        start_time = time.time()
        
        for chunk in audio_chunks:
            # Feed audio chunk
            self.system.feed_audio(chunk)
            
            # Small delay to simulate real-time streaming
            time.sleep(0.02)
            
            # Try to get response
            response = self.system.get_response()
            if response is not None:
                responses_received += 1
        
        # Wait for any remaining responses
        max_wait = 1.0
        elapsed = 0.0
        poll_interval = 0.01
        
        while elapsed < max_wait:
            response = self.system.get_response()
            if response is not None:
                responses_received += 1
            time.sleep(poll_interval)
            elapsed += poll_interval
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Log results
        logger.info(f"Streaming test: {responses_received} responses in {total_time:.2f} seconds")
        logger.info(f"Average response time: {total_time/max(1, responses_received)*1000:.2f} ms")
        
        # Assert we got at least some responses
        self.assertGreater(responses_received, 0)


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling and edge cases"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = MoshiConfig(device="cpu")  # Force CPU for tests
    
    def test_empty_audio(self):
        """Test handling empty audio"""
        system = FullDuplexDialogueSystem(self.config)
        system.initialize()
        system.start()
        
        # Feed empty audio
        empty_audio = torch.zeros(1, 0)
        
        try:
            # This should not crash
            system.feed_audio(empty_audio)
            time.sleep(0.1)
            # No exception means test passed
            passed = True
        except Exception as e:
            logger.error(f"Exception with empty audio: {str(e)}")
            passed = False
        finally:
            system.stop()
        
        self.assertTrue(passed)
    
    def test_malformed_audio(self):
        """Test handling malformed audio"""
        system = FullDuplexDialogueSystem(self.config)
        system.initialize()
        system.start()
        
        # Feed malformed audio (wrong dimensions)
        bad_audio = torch.randn(3, 4, 5)  # Should be [batch, samples]
        
        try:
            # This should not crash the system
            system.feed_audio(bad_audio)
            time.sleep(0.1)
            
            # Try normal audio after to verify system still works
            good_audio = torch.randn(1, 24000)
            system.feed_audio(good_audio)
            time.sleep(0.5)
            
            # Get response
            response = system.get_response()
            still_working = response is not None
        except Exception as e:
            logger.error(f"Exception with malformed audio: {str(e)}")
            still_working = False
        finally:
            system.stop()
        
        # Verify system is still operational
        self.assertTrue(still_working)
    
    @patch.object(MimiCodec, 'encode')
    def test_encoder_failure(self, mock_encode):
        """Test handling encoder failure"""
        # Make encode method raise an exception
        mock_encode.side_effect = RuntimeError("Simulated encoder failure")
        
        system = FullDuplexDialogueSystem(self.config)
        system.initialize()
        system.start()
        
        # Feed audio
        audio = torch.randn(1, 24000)
        
        try:
            # This should not crash the system
            system.feed_audio(audio)
            time.sleep(0.5)
            passed = True
        except Exception as e:
            logger.error(f"Exception during encoder failure: {str(e)}")
            passed = False
        finally:
            system.stop()
        
        self.assertTrue(passed)
    
    @patch.object(MimiCodec, 'decode')
    def test_decoder_failure(self, mock_decode):
        """Test handling decoder failure"""
        # Let encode work normally but make decode raise an exception
        mock_decode.side_effect = RuntimeError("Simulated decoder failure")
        
        system = FullDuplexDialogueSystem(self.config)
        system.initialize()
        system.start()
        
        # Feed audio
        audio = torch.randn(1, 24000)
        
        try:
            # This should not crash the system
            system.feed_audio(audio)
            time.sleep(0.5)
            passed = True
        except Exception as e:
            logger.error(f"Exception during decoder failure: {str(e)}")
            passed = False
        finally:
            system.stop()
        
        self.assertTrue(passed)


class TestIntegrationWithCSM(unittest.TestCase):
    """Tests for integration with CSM"""
    
    @unittest.skip("CSM integration test requires CSM module, mock implementation needed")
    def test_csm_integration(self):
        """Test integration with CSM (mock implementation)"""
        # This would be a comprehensive test of the integration with CSM
        # For now, we skip it since we would need a mock implementation of CSM
        pass


if __name__ == '__main__':
    unittest.main()
