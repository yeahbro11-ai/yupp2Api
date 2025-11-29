# Error Handling and Token Rotation Improvements

## Overview

This document describes the error handling and token rotation improvements implemented in yupp2Api to enhance stability, reliability, and observability.

## Key Improvements

### 1. Centralized Error Classification

All errors are now classified into specific types for appropriate handling:

- **AUTH_ERROR** (401, 403): Invalid or expired tokens
- **RATE_LIMIT** (429): Too many requests
- **SERVER_ERROR** (500, 502, 503, 504): Upstream server issues
- **NETWORK_ERROR**: Connection/timeout issues
- **CLIENT_ERROR** (4xx): Invalid client requests
- **STREAM_ERROR**: Errors during streaming responses
- **PARSE_ERROR**: JSON parsing failures
- **UNKNOWN_ERROR**: Unexpected errors

### 2. Enhanced Token Rotation

#### Token Selection Algorithm
- Tracks error counts and cooldown periods for each token
- Automatically rotates to healthy tokens
- Skips tokens that are:
  - Marked invalid (auth failures)
  - In cooldown period (exceeded max errors)
  - Already used recently

#### Cooldown Management
- Tokens enter cooldown after reaching `MAX_ERROR_COUNT` (default: 3)
- Cooldown duration configured via `ERROR_COOLDOWN` (default: 300 seconds)
- Error counts reset after successful cooldown period
- Permanent invalidation for auth errors

#### Clear Error Messages
When all tokens are unavailable, the API returns specific 503 errors:
- "All upstream tokens are marked invalid" - Check token configuration
- "All upstream tokens temporarily unavailable (N in cooldown)" - Retry later
- Includes list of failure reasons per token (with masked tokens)

### 3. Secure Logging

#### Token Masking
All token logging uses `mask_token()` to prevent secret leakage:
```python
# Shows only last 4 characters
mask_token("sk-1234567890abcdef")  # Returns "...cdef"
```

#### Structured Error Logging
All errors are logged with:
- Error type classification
- Masked token identifier
- Endpoint being called
- High-level error reason
- Exception details (truncated)

Example log:
```
[ERROR] Type: network_error | Token: ...cdef | Endpoint: chat.completions | Request to Yupp.ai timed out | Exception: ReadTimeout(...)
```

### 4. Improved Streaming Error Handling

#### Mid-Stream Error Recovery
- Network disconnects are detected and properly classified
- Malformed chunks are logged but don't crash the stream
- Errors are returned in OpenAI-compatible SSE format:
  ```json
  data: {"error": {"message": "...", "type": "error_type"}}
  ```

#### Reward Claiming Safety
- Reward claiming always runs with timeout protection
- Failures are logged but don't crash the response
- Includes token and reward ID in logs (both masked)

### 5. Request Timeouts

All Yupp API requests now have configurable timeouts:
- Default: 45 seconds (configurable via `YUPP_REQUEST_TIMEOUT`)
- Prevents hanging requests
- Timeout errors trigger token rotation

### 6. Retry Logic

#### Smart Retry Strategy
- Retries with different tokens for retryable errors
- Stops immediately for client errors (4xx)
- Records failure reason per token attempt
- Returns aggregate failure info when all attempts fail

#### Error-Specific Handling
| Error Type | Mark Invalid? | Increment Error Count? | Retry? |
|------------|---------------|------------------------|--------|
| AUTH_ERROR | Yes | No | Yes (next token) |
| RATE_LIMIT | No | Yes | Yes (next token) |
| SERVER_ERROR | No | Yes | Yes (next token) |
| NETWORK_ERROR | No | Yes | Yes (next token) |
| CLIENT_ERROR | No | No | No |

## Configuration

### Environment Variables

```bash
# Token management
YUPP_TOKENS=token1,token2,token3
MAX_ERROR_COUNT=3           # Errors before cooldown
ERROR_COOLDOWN=300          # Cooldown duration in seconds

# Request settings
YUPP_REQUEST_TIMEOUT=45     # Request timeout in seconds

# Debugging
DEBUG_MODE=true             # Enable detailed debug logs
```

## Testing

Comprehensive test suite covering:

1. **Error Classification Tests**
   - HTTP status code mapping
   - Network error detection
   - Parse error handling

2. **Token Masking Tests**
   - Various token lengths
   - Edge cases (empty, short tokens)

3. **Token Rotation Tests**
   - Account selection order
   - Cooldown behavior
   - Recovery after cooldown
   - All-accounts-unavailable scenarios

4. **API Endpoint Tests**
   - Auth error handling
   - Timeout handling
   - Model validation
   - Message validation

Run tests:
```bash
pytest tests/test_error_handling.py -v
```

## Monitoring and Debugging

### Log Patterns to Monitor

**Account Events:**
```
[ACCOUNT] Token ...abcd: Marked invalid: HTTP 401
[ACCOUNT] Token ...abcd: Reached max error count (3), entering cooldown
[ACCOUNT] Token ...abcd: Cooldown elapsed, account re-enabled
```

**Errors:**
```
[ERROR] Type: rate_limit | Token: ...abcd | Endpoint: chat.completions | Upstream HTTP error 429
[ERROR] Type: network_error | Token: ...abcd | Endpoint: reward.claim | Reward claim timeout
```

**Rewards:**
```
[REWARD] Claimed successfully for token ...abcd. New balance: 1250
```

### Common Issues and Solutions

#### Issue: All tokens in cooldown
**Symptom:** 503 errors with "All upstream tokens temporarily unavailable"
**Solution:** 
- Wait for cooldown period to expire
- Check if tokens are hitting rate limits
- Consider adding more tokens or increasing `MAX_ERROR_COUNT`

#### Issue: All tokens marked invalid
**Symptom:** 503 errors with "All upstream tokens are marked invalid"
**Solution:**
- Verify tokens are still valid on Yupp.ai
- Restart service to reload token state
- Update `YUPP_TOKENS` with valid tokens

#### Issue: Frequent timeouts
**Symptom:** Many `network_error` logs with timeout messages
**Solution:**
- Increase `YUPP_REQUEST_TIMEOUT`
- Check network connectivity to Yupp.ai
- Verify proxy settings if using proxies

## API Error Responses

All errors now return consistent structured responses:

```json
{
  "detail": {
    "message": "Human-readable error message",
    "type": "error_type",
    "failures": ["...abc: reason", "...def: reason"]
  }
}
```

Benefits:
- Clients can programmatically handle different error types
- Failure reasons help debug multi-token issues
- No token secrets exposed in responses

## Migration Guide

### Breaking Changes
None - all changes are backward compatible.

### New Features to Leverage
1. **Monitor cooldown_until field** - Track when tokens will be available again
2. **Check error.type in responses** - Handle different error scenarios
3. **Use DEBUG_MODE** - Enable for troubleshooting
4. **Configure timeouts** - Adjust `YUPP_REQUEST_TIMEOUT` for your use case

## Future Enhancements

Potential improvements for future iterations:
1. Metrics/statistics tracking (error rates, token usage)
2. Health check endpoint exposing token status
3. Configurable retry strategies per error type
4. Circuit breaker pattern for faster failover
5. Token refresh/rotation from external source
