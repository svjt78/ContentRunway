"""Redis service for caching, rate limiting, and session management."""

import redis.asyncio as redis
import json
import pickle
import hashlib
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager

from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisService:
    """Service for Redis operations with connection pooling and namespacing."""
    
    def __init__(self):
        self.redis_url = settings.REDIS_URL
        self.pool = None
        self.client = None
        
        # Key namespaces
        self.namespaces = {
            "pipeline": "pipeline:",
            "cache": "cache:",
            "rate_limit": "rate_limit:",
            "session": "session:",
            "llm_response": "llm:",
            "research": "research:",
            "quality": "quality:"
        }
        
        # Default TTL values (in seconds)
        self.default_ttls = {
            "pipeline": 3600 * 24,      # 24 hours
            "cache": 3600,              # 1 hour
            "rate_limit": 3600,         # 1 hour
            "session": 3600 * 2,        # 2 hours
            "llm_response": 3600 * 6,   # 6 hours
            "research": 3600 * 24,      # 24 hours
            "quality": 3600 * 12        # 12 hours
        }
    
    async def connect(self):
        """Initialize Redis connection pool."""
        try:
            self.pool = redis.ConnectionPool.from_url(
                self.redis_url,
                decode_responses=False,
                max_connections=20,
                retry_on_timeout=True
            )
            self.client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.client.ping()
            logger.info("Connected to Redis successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Close Redis connections."""
        if self.client:
            await self.client.close()
        if self.pool:
            await self.pool.disconnect()
        logger.info("Disconnected from Redis")
    
    @asynccontextmanager
    async def get_client(self):
        """Context manager for Redis client with automatic cleanup."""
        if not self.client:
            await self.connect()
        try:
            yield self.client
        except Exception as e:
            logger.error(f"Redis operation failed: {e}")
            raise
    
    def _make_key(self, namespace: str, key: str) -> str:
        """Create namespaced key."""
        prefix = self.namespaces.get(namespace, f"{namespace}:")
        return f"{prefix}{key}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for Redis storage."""
        if isinstance(value, (str, int, float, bool)):
            return json.dumps(value).encode('utf-8')
        else:
            return pickle.dumps(value)
    
    def _deserialize_value(self, value: bytes) -> Any:
        """Deserialize value from Redis."""
        try:
            # Try JSON first (for simple types)
            return json.loads(value.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(value)
    
    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set a key-value pair with optional TTL."""
        try:
            async with self.get_client() as client:
                redis_key = self._make_key(namespace, key)
                serialized_value = self._serialize_value(value)
                
                if ttl is None:
                    ttl = self.default_ttls.get(namespace, 3600)
                
                await client.setex(redis_key, ttl, serialized_value)
                return True
                
        except Exception as e:
            logger.error(f"Failed to set {namespace}:{key}: {e}")
            return False
    
    async def get(self, namespace: str, key: str) -> Optional[Any]:
        """Get a value by key."""
        try:
            async with self.get_client() as client:
                redis_key = self._make_key(namespace, key)
                value = await client.get(redis_key)
                
                if value is None:
                    return None
                
                return self._deserialize_value(value)
                
        except Exception as e:
            logger.error(f"Failed to get {namespace}:{key}: {e}")
            return None
    
    async def delete(self, namespace: str, key: str) -> bool:
        """Delete a key."""
        try:
            async with self.get_client() as client:
                redis_key = self._make_key(namespace, key)
                result = await client.delete(redis_key)
                return result > 0
                
        except Exception as e:
            logger.error(f"Failed to delete {namespace}:{key}: {e}")
            return False
    
    async def exists(self, namespace: str, key: str) -> bool:
        """Check if a key exists."""
        try:
            async with self.get_client() as client:
                redis_key = self._make_key(namespace, key)
                return await client.exists(redis_key) > 0
                
        except Exception as e:
            logger.error(f"Failed to check existence of {namespace}:{key}: {e}")
            return False
    
    async def set_hash(
        self,
        namespace: str,
        key: str,
        field_values: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Set multiple fields in a hash."""
        try:
            async with self.get_client() as client:
                redis_key = self._make_key(namespace, key)
                
                # Serialize all values
                serialized_values = {
                    field: self._serialize_value(value)
                    for field, value in field_values.items()
                }
                
                await client.hset(redis_key, mapping=serialized_values)
                
                if ttl is None:
                    ttl = self.default_ttls.get(namespace, 3600)
                
                await client.expire(redis_key, ttl)
                return True
                
        except Exception as e:
            logger.error(f"Failed to set hash {namespace}:{key}: {e}")
            return False
    
    async def get_hash(self, namespace: str, key: str) -> Optional[Dict[str, Any]]:
        """Get all fields from a hash."""
        try:
            async with self.get_client() as client:
                redis_key = self._make_key(namespace, key)
                hash_data = await client.hgetall(redis_key)
                
                if not hash_data:
                    return None
                
                # Deserialize all values
                result = {}
                for field, value in hash_data.items():
                    field_name = field.decode('utf-8') if isinstance(field, bytes) else field
                    result[field_name] = self._deserialize_value(value)
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to get hash {namespace}:{key}: {e}")
            return None
    
    async def get_hash_field(self, namespace: str, key: str, field: str) -> Optional[Any]:
        """Get a specific field from a hash."""
        try:
            async with self.get_client() as client:
                redis_key = self._make_key(namespace, key)
                value = await client.hget(redis_key, field)
                
                if value is None:
                    return None
                
                return self._deserialize_value(value)
                
        except Exception as e:
            logger.error(f"Failed to get hash field {namespace}:{key}:{field}: {e}")
            return None
    
    async def increment(
        self,
        namespace: str,
        key: str,
        amount: int = 1,
        ttl: Optional[int] = None
    ) -> int:
        """Increment a counter and return new value."""
        try:
            async with self.get_client() as client:
                redis_key = self._make_key(namespace, key)
                new_value = await client.incrby(redis_key, amount)
                
                if ttl is None:
                    ttl = self.default_ttls.get(namespace, 3600)
                
                await client.expire(redis_key, ttl)
                return new_value
                
        except Exception as e:
            logger.error(f"Failed to increment {namespace}:{key}: {e}")
            return 0
    
    async def rate_limit_check(
        self,
        identifier: str,
        limit: int,
        window_seconds: int = 3600
    ) -> Dict[str, Any]:
        """Check and update rate limit for an identifier."""
        try:
            async with self.get_client() as client:
                key = self._make_key("rate_limit", f"{identifier}:{window_seconds}")
                
                # Get current count
                current_count = await client.get(key)
                current_count = int(current_count) if current_count else 0
                
                # Check if limit exceeded
                if current_count >= limit:
                    ttl = await client.ttl(key)
                    return {
                        "allowed": False,
                        "current_count": current_count,
                        "limit": limit,
                        "reset_in_seconds": ttl if ttl > 0 else window_seconds
                    }
                
                # Increment and set expiry if new key
                await client.incr(key)
                if current_count == 0:
                    await client.expire(key, window_seconds)
                
                return {
                    "allowed": True,
                    "current_count": current_count + 1,
                    "limit": limit,
                    "remaining": limit - current_count - 1
                }
                
        except Exception as e:
            logger.error(f"Rate limit check failed for {identifier}: {e}")
            # Fail open - allow request if Redis fails
            return {
                "allowed": True,
                "current_count": 0,
                "limit": limit,
                "remaining": limit,
                "error": str(e)
            }
    
    async def cache_llm_response(
        self,
        prompt_hash: str,
        model: str,
        response: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache LLM response with prompt hash as key."""
        cache_key = f"{model}:{prompt_hash}"
        
        cache_data = {
            "response": response,
            "model": model,
            "cached_at": datetime.now().isoformat(),
            "prompt_hash": prompt_hash
        }
        
        return await self.set("llm_response", cache_key, cache_data, ttl)
    
    async def get_cached_llm_response(
        self,
        prompt_hash: str,
        model: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached LLM response."""
        cache_key = f"{model}:{prompt_hash}"
        return await self.get("llm_response", cache_key)
    
    def create_prompt_hash(self, prompt: str, **kwargs) -> str:
        """Create deterministic hash for prompt caching."""
        # Include prompt and any relevant parameters
        cache_input = json.dumps({
            "prompt": prompt,
            **kwargs
        }, sort_keys=True)
        
        return hashlib.sha256(cache_input.encode('utf-8')).hexdigest()[:16]
    
    async def cache_research_results(
        self,
        query: str,
        domain: str,
        results: List[Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache research results for a domain query."""
        # Create cache key from query and domain
        query_hash = hashlib.sha256(f"{query}:{domain}".encode('utf-8')).hexdigest()[:16]
        cache_key = f"{domain}:{query_hash}"
        
        cache_data = {
            "query": query,
            "domain": domain,
            "results": results,
            "cached_at": datetime.now().isoformat(),
            "result_count": len(results)
        }
        
        if ttl is None:
            ttl = self.default_ttls["research"]
        
        return await self.set("research", cache_key, cache_data, ttl)
    
    async def get_cached_research_results(
        self,
        query: str,
        domain: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached research results."""
        query_hash = hashlib.sha256(f"{query}:{domain}".encode('utf-8')).hexdigest()[:16]
        cache_key = f"{domain}:{query_hash}"
        
        cached_data = await self.get("research", cache_key)
        if cached_data:
            return cached_data.get("results", [])
        return None
    
    async def store_pipeline_state(
        self,
        run_id: str,
        state: Dict[str, Any]
    ) -> bool:
        """Store pipeline state for resume capability."""
        # Store both full state and checkpoint markers
        full_state_key = f"full_state:{run_id}"
        checkpoint_key = f"checkpoint:{run_id}"
        
        # Store full state
        state_stored = await self.set("pipeline", full_state_key, state)
        
        # Store lightweight checkpoint for quick status checks
        checkpoint_data = {
            "run_id": run_id,
            "status": state.get("status"),
            "current_step": state.get("current_step"),
            "progress_percentage": state.get("progress_percentage"),
            "updated_at": datetime.now().isoformat()
        }
        checkpoint_stored = await self.set("pipeline", checkpoint_key, checkpoint_data)
        
        return state_stored and checkpoint_stored
    
    async def get_pipeline_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get full pipeline state."""
        full_state_key = f"full_state:{run_id}"
        return await self.get("pipeline", full_state_key)
    
    async def get_pipeline_checkpoint(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get lightweight pipeline checkpoint."""
        checkpoint_key = f"checkpoint:{run_id}"
        return await self.get("pipeline", checkpoint_key)
    
    async def cache_quality_gate_result(
        self,
        content_hash: str,
        gate_type: str,
        result: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache quality gate results for similar content."""
        cache_key = f"{gate_type}:{content_hash}"
        
        cache_data = {
            "gate_type": gate_type,
            "result": result,
            "content_hash": content_hash,
            "cached_at": datetime.now().isoformat()
        }
        
        return await self.set("quality", cache_key, cache_data, ttl)
    
    async def get_cached_quality_result(
        self,
        content_hash: str,
        gate_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached quality gate result."""
        cache_key = f"{gate_type}:{content_hash}"
        cached_data = await self.get("quality", cache_key)
        
        if cached_data:
            return cached_data.get("result")
        return None
    
    def create_content_hash(self, content: str, additional_context: str = "") -> str:
        """Create hash for content caching."""
        hash_input = f"{content}{additional_context}"
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()[:16]
    
    async def add_to_set(
        self,
        namespace: str,
        key: str,
        value: str,
        ttl: Optional[int] = None
    ) -> bool:
        """Add value to a Redis set."""
        try:
            async with self.get_client() as client:
                redis_key = self._make_key(namespace, key)
                await client.sadd(redis_key, value)
                
                if ttl is None:
                    ttl = self.default_ttls.get(namespace, 3600)
                
                await client.expire(redis_key, ttl)
                return True
                
        except Exception as e:
            logger.error(f"Failed to add to set {namespace}:{key}: {e}")
            return False
    
    async def is_in_set(self, namespace: str, key: str, value: str) -> bool:
        """Check if value is in Redis set."""
        try:
            async with self.get_client() as client:
                redis_key = self._make_key(namespace, key)
                return await client.sismember(redis_key, value)
                
        except Exception as e:
            logger.error(f"Failed to check set membership {namespace}:{key}: {e}")
            return False
    
    async def get_set_members(self, namespace: str, key: str) -> List[str]:
        """Get all members of a Redis set."""
        try:
            async with self.get_client() as client:
                redis_key = self._make_key(namespace, key)
                members = await client.smembers(redis_key)
                return [member.decode('utf-8') if isinstance(member, bytes) else member for member in members]
                
        except Exception as e:
            logger.error(f"Failed to get set members {namespace}:{key}: {e}")
            return []
    
    async def push_to_list(
        self,
        namespace: str,
        key: str,
        value: Any,
        max_length: Optional[int] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """Push value to Redis list with optional length limit."""
        try:
            async with self.get_client() as client:
                redis_key = self._make_key(namespace, key)
                serialized_value = self._serialize_value(value)
                
                await client.lpush(redis_key, serialized_value)
                
                # Trim list if max_length specified
                if max_length:
                    await client.ltrim(redis_key, 0, max_length - 1)
                
                if ttl is None:
                    ttl = self.default_ttls.get(namespace, 3600)
                
                await client.expire(redis_key, ttl)
                return True
                
        except Exception as e:
            logger.error(f"Failed to push to list {namespace}:{key}: {e}")
            return False
    
    async def get_list_range(
        self,
        namespace: str,
        key: str,
        start: int = 0,
        end: int = -1
    ) -> List[Any]:
        """Get range of values from Redis list."""
        try:
            async with self.get_client() as client:
                redis_key = self._make_key(namespace, key)
                values = await client.lrange(redis_key, start, end)
                
                return [self._deserialize_value(value) for value in values]
                
        except Exception as e:
            logger.error(f"Failed to get list range {namespace}:{key}: {e}")
            return []
    
    async def flush_namespace(self, namespace: str) -> bool:
        """Delete all keys in a namespace."""
        try:
            async with self.get_client() as client:
                pattern = self._make_key(namespace, "*")
                keys = await client.keys(pattern)
                
                if keys:
                    await client.delete(*keys)
                    logger.info(f"Flushed {len(keys)} keys from namespace {namespace}")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to flush namespace {namespace}: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis usage statistics."""
        try:
            async with self.get_client() as client:
                info = await client.info()
                
                stats = {
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory_human", "0B"),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                    "uptime_in_seconds": info.get("uptime_in_seconds", 0)
                }
                
                # Calculate hit ratio
                hits = stats["keyspace_hits"]
                misses = stats["keyspace_misses"]
                if hits + misses > 0:
                    stats["cache_hit_ratio"] = hits / (hits + misses)
                else:
                    stats["cache_hit_ratio"] = 0.0
                
                # Get namespace key counts
                namespace_counts = {}
                for namespace in self.namespaces.keys():
                    pattern = self._make_key(namespace, "*")
                    keys = await client.keys(pattern)
                    namespace_counts[namespace] = len(keys)
                
                stats["namespace_key_counts"] = namespace_counts
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {}


# Global Redis service instance
redis_service = RedisService()


# Context manager for automatic connection management
@asynccontextmanager
async def get_redis_service():
    """Context manager for Redis service with automatic connection management."""
    if not redis_service.client:
        await redis_service.connect()
    try:
        yield redis_service
    finally:
        # Connection stays open for reuse
        pass


# Utility functions for common caching patterns
async def cache_with_redis(
    cache_key: str,
    cache_namespace: str,
    fetch_function,
    ttl: Optional[int] = None,
    **fetch_kwargs
) -> Any:
    """
    Generic caching wrapper that checks Redis first, then fetches if needed.
    
    Args:
        cache_key: Key to use for caching
        cache_namespace: Redis namespace to use
        fetch_function: Function to call if cache miss
        ttl: Cache TTL in seconds
        **fetch_kwargs: Arguments to pass to fetch_function
    
    Returns:
        Cached or freshly fetched data
    """
    try:
        # Try cache first
        cached_result = await redis_service.get(cache_namespace, cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for {cache_namespace}:{cache_key}")
            return cached_result
        
        # Cache miss - fetch fresh data
        logger.debug(f"Cache miss for {cache_namespace}:{cache_key}, fetching fresh data")
        result = await fetch_function(**fetch_kwargs)
        
        # Store in cache
        await redis_service.set(cache_namespace, cache_key, result, ttl)
        
        return result
        
    except Exception as e:
        logger.error(f"Cache operation failed for {cache_namespace}:{cache_key}: {e}")
        # Fallback to direct fetch
        return await fetch_function(**fetch_kwargs)