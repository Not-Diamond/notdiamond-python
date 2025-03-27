"""Tests for utility functions in _utils.py"""

import pytest

from notdiamond._utils import token_counter


class TestTokenCounter:
    """Test class for token_counter function"""

    def test_token_counter_with_text(self):
        """Test token_counter with text input"""
        text = "This is a simple test sentence."
        count = token_counter(model="gpt-4", text=text)
        assert isinstance(count, int)
        assert count > 0
        assert count <= len(text)  # Tokens should be fewer than characters

    def test_token_counter_with_messages(self):
        """Test token_counter with messages input"""
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
        ]
        count = token_counter(model="gpt-3.5-turbo", messages=messages)
        assert isinstance(count, int)
        assert count > 0
        
        # Verify that computing tokens from text and from messages gives consistent results
        combined_text = "Hello, how are you?I'm doing well, thank you for asking!"
        text_count = token_counter(model="gpt-3.5-turbo", text=combined_text)
        assert count >= text_count  # Messages might add a few tokens for role markers

    def test_token_counter_different_models(self):
        """Test token_counter with different models"""
        text = "This is a test of token counting across different models."
        
        gpt4_count = token_counter(model="gpt-4", text=text)
        gpt35_count = token_counter(model="gpt-3.5-turbo", text=text)
        
        # Token counts should be similar for same text across different models
        assert abs(gpt4_count - gpt35_count) <= 2

    def test_token_counter_fallback(self):
        """Test token_counter fallback mechanism with unknown model"""
        text = "Testing the fallback mechanism with an unknown model."
        
        # Test with a model name that won't have a specific tokenizer
        count = token_counter(model="unknown-model", text=text)
        assert isinstance(count, int)
        assert count > 0
        
        # Compare with a known model's count
        known_model_count = token_counter(model="gpt-4", text=text)
        # The counts should be reasonably similar
        assert abs(count - known_model_count) <= 5

    def test_token_counter_error_handling(self):
        """Test token_counter error handling"""
        # Test with neither text nor messages
        with pytest.raises(ValueError):
            token_counter(model="gpt-4") 