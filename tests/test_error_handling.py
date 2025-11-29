"""
Tests for error handling and token rotation in yupp2Api
"""
import os
import time
import json
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
import requests

# Set test environment variables before importing app
os.environ["CLIENT_API_KEYS"] = "test-api-key"
os.environ["YUPP_TOKENS"] = "test-token-1,test-token-2,test-token-3"
os.environ["MAX_ERROR_COUNT"] = "2"
os.environ["ERROR_COOLDOWN"] = "5"
os.environ["DEBUG_MODE"] = "false"

# Ensure a local model file exists to avoid network calls
TEST_MODEL_FILE = os.path.join(os.path.dirname(__file__), "model.json")
os.makedirs(os.path.dirname(TEST_MODEL_FILE), exist_ok=True)
with open(TEST_MODEL_FILE, "w", encoding="utf-8") as model_file:
    json.dump(
        [{"label": "test-model", "name": "test-yupp-model", "publisher": "test"}],
        model_file,
    )
os.environ["MODEL_FILE"] = TEST_MODEL_FILE

from yyapi import (
    app,
    ErrorType,
    classify_error,
    mask_token,
    should_retry_error,
    should_mark_invalid,
    should_increment_error_count,
    get_best_yupp_account,
    mark_account_invalid,
    set_account_cooldown,
    increment_account_error,
    YUPP_ACCOUNTS,
    load_yupp_accounts,
)


class TestErrorClassification:
    """Test error classification logic"""

    def test_classify_auth_error(self):
        """Test that 401/403 errors are classified as AUTH_ERROR"""
        mock_response = Mock()
        mock_response.status_code = 401
        error = requests.exceptions.HTTPError(response=mock_response)
        assert classify_error(error) == ErrorType.AUTH_ERROR

        mock_response.status_code = 403
        error = requests.exceptions.HTTPError(response=mock_response)
        assert classify_error(error) == ErrorType.AUTH_ERROR

    def test_classify_rate_limit(self):
        """Test that 429 errors are classified as RATE_LIMIT"""
        mock_response = Mock()
        mock_response.status_code = 429
        error = requests.exceptions.HTTPError(response=mock_response)
        assert classify_error(error) == ErrorType.RATE_LIMIT

    def test_classify_server_error(self):
        """Test that 5xx errors are classified as SERVER_ERROR"""
        for status in [500, 502, 503, 504]:
            mock_response = Mock()
            mock_response.status_code = status
            error = requests.exceptions.HTTPError(response=mock_response)
            assert classify_error(error) == ErrorType.SERVER_ERROR

    def test_classify_client_error(self):
        """Test that 4xx errors (not 401/403/429) are classified as CLIENT_ERROR"""
        mock_response = Mock()
        mock_response.status_code = 400
        error = requests.exceptions.HTTPError(response=mock_response)
        assert classify_error(error) == ErrorType.CLIENT_ERROR

    def test_classify_network_error(self):
        """Test that connection/timeout errors are classified as NETWORK_ERROR"""
        error = requests.exceptions.ConnectionError()
        assert classify_error(error) == ErrorType.NETWORK_ERROR

        error = requests.exceptions.Timeout()
        assert classify_error(error) == ErrorType.NETWORK_ERROR

    def test_classify_parse_error(self):
        """Test that JSON decode errors are classified as PARSE_ERROR"""
        error = json.JSONDecodeError("msg", "doc", 0)
        assert classify_error(error) == ErrorType.PARSE_ERROR

    def test_classify_unknown_error(self):
        """Test that unknown errors are classified as UNKNOWN_ERROR"""
        error = ValueError("Something went wrong")
        assert classify_error(error) == ErrorType.UNKNOWN_ERROR


class TestTokenMasking:
    """Test token masking for secure logging"""

    def test_mask_token_with_default(self):
        """Test masking with default visible chars"""
        token = "sk-1234567890abcdef"
        masked = mask_token(token)
        assert masked == "...cdef"
        assert "1234567890" not in masked

    def test_mask_token_custom_visible(self):
        """Test masking with custom visible chars"""
        token = "sk-1234567890abcdef"
        masked = mask_token(token, visible_chars=8)
        assert masked == "...90abcdef"

    def test_mask_short_token(self):
        """Test masking a token shorter than visible chars"""
        token = "abc"
        masked = mask_token(token, visible_chars=4)
        assert masked == "***"

    def test_mask_empty_token(self):
        """Test masking empty/None token"""
        assert mask_token("") == "***"
        assert mask_token(None) == "***"


class TestErrorHandlingLogic:
    """Test error handling decision logic"""

    def test_should_retry_error(self):
        """Test which errors should trigger retries"""
        assert should_retry_error(ErrorType.AUTH_ERROR) is True
        assert should_retry_error(ErrorType.RATE_LIMIT) is True
        assert should_retry_error(ErrorType.SERVER_ERROR) is True
        assert should_retry_error(ErrorType.NETWORK_ERROR) is True
        assert should_retry_error(ErrorType.CLIENT_ERROR) is False

    def test_should_mark_invalid(self):
        """Test which errors should mark account as invalid"""
        assert should_mark_invalid(ErrorType.AUTH_ERROR) is True
        assert should_mark_invalid(ErrorType.RATE_LIMIT) is False
        assert should_mark_invalid(ErrorType.SERVER_ERROR) is False
        assert should_mark_invalid(ErrorType.NETWORK_ERROR) is False

    def test_should_increment_error_count(self):
        """Test which errors should increment error count"""
        assert should_increment_error_count(ErrorType.AUTH_ERROR) is False
        assert should_increment_error_count(ErrorType.RATE_LIMIT) is True
        assert should_increment_error_count(ErrorType.SERVER_ERROR) is True
        assert should_increment_error_count(ErrorType.NETWORK_ERROR) is True
        assert should_increment_error_count(ErrorType.CLIENT_ERROR) is False


class TestAccountRotation:
    """Test account rotation and cooldown logic"""

    def setup_method(self):
        """Reset accounts before each test"""
        load_yupp_accounts()

    def test_get_best_account_returns_valid_account(self):
        """Test that get_best_yupp_account returns a valid account"""
        account = get_best_yupp_account()
        assert account is not None
        assert account["is_valid"] is True
        assert account["token"] in ["test-token-1", "test-token-2", "test-token-3"]

    def test_account_rotation_order(self):
        """Test that accounts are rotated in order"""
        first = get_best_yupp_account()
        second = get_best_yupp_account()
        # Should rotate to next account
        assert first["token"] != second["token"]

    def test_account_marked_invalid(self):
        """Test that accounts can be marked invalid"""
        account = get_best_yupp_account()
        original_token = account["token"]
        mark_account_invalid(account, "test reason")
        
        assert account["is_valid"] is False
        
        # Next account should be different
        next_account = get_best_yupp_account()
        assert next_account["token"] != original_token

    def test_account_cooldown(self):
        """Test that accounts enter cooldown after max errors"""
        account = get_best_yupp_account()
        original_token = account["token"]
        
        # Increment error count to max
        max_errors = int(os.environ["MAX_ERROR_COUNT"])
        for _ in range(max_errors):
            increment_account_error(account)
        
        # Account should now be in cooldown
        set_account_cooldown(account)
        
        # Next call should skip this account
        next_account = get_best_yupp_account()
        assert next_account["token"] != original_token

    def test_all_accounts_in_cooldown(self):
        """Test behavior when all accounts are in cooldown"""
        # Put all accounts in cooldown
        for account in YUPP_ACCOUNTS:
            max_errors = int(os.environ["MAX_ERROR_COUNT"])
            for _ in range(max_errors):
                increment_account_error(account)
            set_account_cooldown(account)
        
        # Should return None
        account = get_best_yupp_account()
        assert account is None

    def test_cooldown_expiry(self):
        """Test that accounts become available after cooldown expires"""
        account = YUPP_ACCOUNTS[0]
        original_error_count = account["error_count"]
        
        # Set cooldown with very short duration
        account["cooldown_until"] = time.time() - 1  # Already expired
        account["error_count"] = 2
        
        # Should be available again
        selected = get_best_yupp_account()
        assert selected is not None


class TestAPIEndpoints:
    """Test API endpoints with various error scenarios"""

    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
        self.headers = {"Authorization": "Bearer test-api-key"}
        load_yupp_accounts()

    def teardown_method(self):
        """Cleanup test client"""
        if hasattr(self, "client"):
            self.client.close()

    def test_authentication_required(self):
        """Test that endpoints require authentication"""
        response = self.client.get("/v1/models")
        assert response.status_code == 401

    def test_invalid_api_key(self):
        """Test that invalid API keys are rejected"""
        headers = {"Authorization": "Bearer invalid-key"}
        response = self.client.get("/v1/models", headers=headers)
        assert response.status_code == 403

    def test_models_endpoint_with_auth(self):
        """Test that models endpoint works with authentication"""
        with patch("yyapi.YUPP_MODELS", [{"label": "test-model", "publisher": "test"}]):
            response = self.client.get("/v1/models", headers=self.headers)
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert isinstance(data["data"], list)

    @patch("yyapi.create_requests_session")
    def test_chat_completions_auth_error(self, mock_session):
        """Test handling of auth errors from Yupp"""
        # Setup mock
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        
        mock_session_instance = Mock()
        mock_session_instance.post.return_value = mock_response
        mock_session.return_value = mock_session_instance

        # Setup models
        with patch("yyapi.YUPP_MODELS", [{"label": "test-model", "name": "test-yupp-model"}]):
            payload = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            }
            
            # Should fail after trying all accounts
            response = self.client.post(
                "/v1/chat/completions",
                headers=self.headers,
                json=payload
            )
            
            assert response.status_code == 503
            data = response.json()
            assert "detail" in data

    @patch("yyapi.create_requests_session")
    def test_chat_completions_timeout(self, mock_session):
        """Test handling of timeout errors"""
        mock_session_instance = Mock()
        mock_session_instance.post.side_effect = requests.exceptions.Timeout()
        mock_session.return_value = mock_session_instance

        with patch("yyapi.YUPP_MODELS", [{"label": "test-model", "name": "test-yupp-model"}]):
            payload = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            }
            
            response = self.client.post(
                "/v1/chat/completions",
                headers=self.headers,
                json=payload
            )
            
            assert response.status_code == 503

    def test_chat_completions_missing_model(self):
        """Test error when model is not found"""
        payload = {
            "model": "non-existent-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        }
        
        response = self.client.post(
            "/v1/chat/completions",
            headers=self.headers,
            json=payload
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_chat_completions_missing_messages(self):
        """Test error when messages are missing"""
        with patch("yyapi.YUPP_MODELS", [{"label": "test-model", "name": "test-yupp-model"}]):
            payload = {
                "model": "test-model",
                "messages": [],
                "stream": False
            }
            
            response = self.client.post(
                "/v1/chat/completions",
                headers=self.headers,
                json=payload
            )
            
            assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
