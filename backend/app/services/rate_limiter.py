"""Rate limiting service for external API calls and resource protection."""

import asyncio
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import logging
from functools import wraps

from app.services.redis_service import redis_service
from app.core.config import settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter using Redis for distributed rate limiting."""
    
    def __init__(self):
        # Default rate limits for different services
        self.default_limits = {
            "openai": {"requests": 50, "window": 3600},      # 50 requests/hour
            "google_ai": {"requests": 60, "window": 3600},   # 60 requests/hour
            "anthropic": {"requests": 40, "window": 3600},   # 40 requests/hour
            "web_search": {"requests": 100, "window": 3600}, # 100 searches/hour
            "embeddings": {"requests": 200, "window": 3600}, # 200 embeddings/hour
            "general": {"requests": 1000, "window": 3600}    # General fallback
        }
    
    async def check_rate_limit(
        self,
        service: str,
        identifier: str,
        custom_limit: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Check rate limit for a service and identifier.
        
        Args:
            service: Service name (openai, google_ai, anthropic, etc.)
            identifier: Unique identifier (user_id, api_key_hash, etc.)
            custom_limit: Optional custom limit override
            
        Returns:
            Dictionary with rate limit status
        """
        # Get rate limit configuration
        if custom_limit:
            limit_config = custom_limit
        else:
            limit_config = self.default_limits.get(service, self.default_limits["general"])
        
        # Create rate limit key
        rate_limit_key = f"{service}:{identifier}"
        
        # Check rate limit using Redis
        result = await redis_service.rate_limit_check(
            identifier=rate_limit_key,
            limit=limit_config["requests"],
            window_seconds=limit_config["window"]
        )
        
        # Add service context to result
        result["service"] = service
        result["window_seconds"] = limit_config["window"]
        
        logger.debug(f"Rate limit check for {service}:{identifier}: {result}")
        
        return result
    
    async def enforce_rate_limit(
        self,
        service: str,
        identifier: str,
        custom_limit: Optional[Dict[str, int]] = None
    ) -> bool:
        """
        Enforce rate limit - raises exception if limit exceeded.
        
        Args:
            service: Service name
            identifier: Unique identifier
            custom_limit: Optional custom limit override
            
        Returns:
            True if request is allowed
            
        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        result = await self.check_rate_limit(service, identifier, custom_limit)
        
        if not result["allowed"]:
            reset_time = result.get("reset_in_seconds", 3600)
            raise RateLimitExceeded(
                f"Rate limit exceeded for {service}. "
                f"Limit: {result['limit']} requests. "
                f"Reset in {reset_time} seconds."
            )
        
        return True
    
    def rate_limited(
        self,
        service: str,
        identifier_func: Optional[Callable] = None,
        custom_limit: Optional[Dict[str, int]] = None
    ):
        """
        Decorator for rate limiting function calls.
        
        Args:
            service: Service name for rate limiting
            identifier_func: Function to extract identifier from args/kwargs
            custom_limit: Optional custom rate limit
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Determine identifier
                if identifier_func:
                    identifier = identifier_func(*args, **kwargs)
                else:
                    # Default: use function name as identifier
                    identifier = func.__name__
                
                # Check rate limit
                await self.enforce_rate_limit(service, identifier, custom_limit)
                
                # Execute function
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass


# Global rate limiter instance
rate_limiter = RateLimiter()


# Decorator shortcuts for common services
def openai_rate_limited(identifier_func: Optional[Callable] = None):
    """Rate limit decorator for OpenAI API calls."""
    return rate_limiter.rate_limited("openai", identifier_func)


def google_ai_rate_limited(identifier_func: Optional[Callable] = None):
    """Rate limit decorator for Google AI API calls."""
    return rate_limiter.rate_limited("google_ai", identifier_func)


def anthropic_rate_limited(identifier_func: Optional[Callable] = None):
    """Rate limit decorator for Anthropic API calls."""
    return rate_limiter.rate_limited("anthropic", identifier_func)


def web_search_rate_limited(identifier_func: Optional[Callable] = None):
    """Rate limit decorator for web search operations."""
    return rate_limiter.rate_limited("web_search", identifier_func)


# Enhanced LLM wrapper with caching and rate limiting
class CachedLLMWrapper:
    """Wrapper for LLM calls with automatic caching and rate limiting."""
    
    def __init__(self, llm, service_name: str):
        self.llm = llm
        self.service_name = service_name
    
    async def ainvoke_cached(
        self,
        messages,
        identifier: str = "default",
        cache_ttl: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Invoke LLM with automatic caching and rate limiting.
        
        Args:
            messages: Messages to send to LLM
            identifier: Identifier for rate limiting
            cache_ttl: Cache TTL override
            **kwargs: Additional arguments for LLM
            
        Returns:
            LLM response (cached or fresh)
        """
        # Create prompt hash for caching
        prompt_content = str(messages)
        prompt_hash = redis_service.create_prompt_hash(prompt_content, **kwargs)
        
        # Check cache first
        cached_response = await redis_service.get_cached_llm_response(
            prompt_hash, 
            self.service_name
        )
        
        if cached_response:
            logger.debug(f"LLM cache hit for {self.service_name}:{prompt_hash[:8]}")
            return cached_response["response"]
        
        # Check rate limit
        await rate_limiter.enforce_rate_limit(self.service_name, identifier)
        
        # Make LLM call
        logger.debug(f"LLM cache miss for {self.service_name}:{prompt_hash[:8]}, making API call")
        response = await self.llm.ainvoke(messages, **kwargs)
        
        # Cache response
        await redis_service.cache_llm_response(
            prompt_hash,
            self.service_name,
            response,
            cache_ttl
        )
        
        return response


# Utility function to wrap existing LLM instances
def wrap_llm_with_caching(llm, service_name: str) -> CachedLLMWrapper:
    """Wrap an existing LLM instance with caching and rate limiting."""
    return CachedLLMWrapper(llm, service_name)


# Context manager for rate-limited operations
class RateLimitedOperation:
    """Context manager for rate-limited operations with automatic retry."""
    
    def __init__(
        self,
        service: str,
        identifier: str,
        max_retries: int = 3,
        retry_delay: float = 60.0
    ):
        self.service = service
        self.identifier = identifier
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retries = 0
    
    async def __aenter__(self):
        """Enter context with rate limit checking."""
        while self.retries <= self.max_retries:
            try:
                await rate_limiter.enforce_rate_limit(self.service, self.identifier)
                return self
            except RateLimitExceeded as e:
                if self.retries >= self.max_retries:
                    logger.error(f"Rate limit exceeded for {self.service}:{self.identifier} after {self.retries} retries")
                    raise
                
                logger.warning(f"Rate limit exceeded for {self.service}:{self.identifier}, retrying in {self.retry_delay}s")
                await asyncio.sleep(self.retry_delay)
                self.retries += 1
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        pass