"""Enhanced LLM Response Caching for Cost Optimization."""

import hashlib
import json
import asyncio
import logging
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class CacheCategory(Enum):
    """Categories for different types of cached content."""
    QUALITY_GATES = "quality_gates"
    CLASSIFICATION = "classification"
    SEO_ANALYSIS = "seo_analysis"
    FACT_CHECKING = "fact_checking"
    CONTENT_ANALYSIS = "content_analysis"
    RESEARCH_RESULTS = "research_results"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: datetime
    category: CacheCategory
    agent_name: str
    model_used: str
    content_hash: str
    hit_count: int = 0
    cost_saved: float = 0.0


class LLMCache:
    """Enhanced caching layer for LLM responses with cost tracking."""
    
    def __init__(self, redis_service=None):
        self.redis_service = redis_service
        self.in_memory_cache: Dict[str, CacheEntry] = {}
        self.max_memory_cache_size = 1000
        
        # TTL configurations for different cache categories
        self.ttl_hours = {
            CacheCategory.QUALITY_GATES: 24,      # Quality scores for similar content
            CacheCategory.CLASSIFICATION: 48,     # Classification results
            CacheCategory.SEO_ANALYSIS: 72,       # SEO analysis for similar topics
            CacheCategory.FACT_CHECKING: 168,     # Fact-check results (7 days)
            CacheCategory.CONTENT_ANALYSIS: 48,   # Content analysis results
            CacheCategory.RESEARCH_RESULTS: 96    # Research results (4 days)
        }
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "cost_savings": 0.0,
            "cache_size": 0
        }
    
    def _generate_cache_key(
        self, 
        agent_name: str, 
        model: str, 
        prompt: str, 
        content: str = "", 
        extra_params: Dict = None
    ) -> str:
        """Generate unique cache key for LLM request."""
        
        # Create content hash for similarity matching
        content_hash = self._create_content_hash(content)
        
        # Include key parameters in cache key
        key_data = {
            "agent": agent_name,
            "model": model,
            "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()[:16],
            "content_hash": content_hash,
            "params": extra_params or {}
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return f"llm_cache:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _create_content_hash(self, content: str) -> str:
        """Create content hash for similarity matching."""
        if not content:
            return ""
            
        # Use first 500 characters for similarity matching
        content_sample = content[:500].lower().strip()
        return hashlib.md5(content_sample.encode()).hexdigest()[:12]
    
    def _get_cache_category(self, agent_name: str) -> CacheCategory:
        """Determine cache category based on agent name."""
        if "Gate" in agent_name:
            return CacheCategory.QUALITY_GATES
        elif "Classifier" in agent_name or "Generator" in agent_name:
            return CacheCategory.CLASSIFICATION
        elif "SEO" in agent_name:
            return CacheCategory.SEO_ANALYSIS
        elif "FactCheck" in agent_name:
            return CacheCategory.FACT_CHECKING
        elif "Research" in agent_name:
            return CacheCategory.RESEARCH_RESULTS
        else:
            return CacheCategory.CONTENT_ANALYSIS
    
    async def get_cached_response(
        self,
        agent_name: str,
        model: str,
        prompt: str,
        content: str = "",
        extra_params: Dict = None
    ) -> Optional[Any]:
        """Get cached LLM response if available."""
        
        cache_key = self._generate_cache_key(agent_name, model, prompt, content, extra_params)
        
        # Try in-memory cache first
        if cache_key in self.in_memory_cache:
            entry = self.in_memory_cache[cache_key]
            
            # Check if expired
            if datetime.now() > entry.expires_at:
                del self.in_memory_cache[cache_key]
                return None
            
            # Update hit statistics
            entry.hit_count += 1
            self.stats["hits"] += 1
            
            logger.info(f"Cache hit for {agent_name} (in-memory)")
            return entry.value
        
        # Try Redis cache if available
        if self.redis_service:
            try:
                cached_data = await self.redis_service.get(cache_key)
                if cached_data:
                    entry_dict = json.loads(cached_data)
                    entry = CacheEntry(**entry_dict)
                    
                    # Check if expired
                    if datetime.now() > entry.expires_at:
                        await self.redis_service.delete(cache_key)
                        return None
                    
                    # Add to in-memory cache for faster access
                    self._add_to_memory_cache(cache_key, entry)
                    
                    # Update statistics
                    entry.hit_count += 1
                    self.stats["hits"] += 1
                    
                    logger.info(f"Cache hit for {agent_name} (Redis)")
                    return entry.value
                    
            except Exception as e:
                logger.warning(f"Redis cache retrieval failed: {e}")
        
        # Cache miss
        self.stats["misses"] += 1
        logger.debug(f"Cache miss for {agent_name}")
        return None
    
    async def cache_response(
        self,
        agent_name: str,
        model: str,
        prompt: str,
        response: Any,
        content: str = "",
        extra_params: Dict = None,
        cost_saved: float = 0.0
    ) -> None:
        """Cache LLM response with metadata."""
        
        cache_key = self._generate_cache_key(agent_name, model, prompt, content, extra_params)
        category = self._get_cache_category(agent_name)
        
        # Calculate expiration
        ttl_hours = self.ttl_hours.get(category, 24)
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            value=response,
            created_at=datetime.now(),
            expires_at=expires_at,
            category=category,
            agent_name=agent_name,
            model_used=model,
            content_hash=self._create_content_hash(content),
            cost_saved=cost_saved
        )
        
        # Add to in-memory cache
        self._add_to_memory_cache(cache_key, entry)
        
        # Add to Redis cache if available
        if self.redis_service:
            try:
                entry_dict = asdict(entry)
                # Convert datetime objects to ISO strings for JSON serialization
                entry_dict["created_at"] = entry.created_at.isoformat()
                entry_dict["expires_at"] = entry.expires_at.isoformat()
                entry_dict["category"] = entry.category.value
                
                await self.redis_service.setex(
                    cache_key, 
                    int(ttl_hours * 3600),  # TTL in seconds
                    json.dumps(entry_dict, default=str)
                )
                
            except Exception as e:
                logger.warning(f"Redis cache storage failed: {e}")
        
        # Update statistics
        self.stats["cost_savings"] += cost_saved
        self.stats["cache_size"] = len(self.in_memory_cache)
        
        logger.debug(f"Cached response for {agent_name} (TTL: {ttl_hours}h)")
    
    def _add_to_memory_cache(self, cache_key: str, entry: CacheEntry) -> None:
        """Add entry to in-memory cache with size management."""
        
        # If cache is full, remove oldest entries
        if len(self.in_memory_cache) >= self.max_memory_cache_size:
            # Remove 10% of oldest entries
            oldest_keys = sorted(
                self.in_memory_cache.keys(),
                key=lambda k: self.in_memory_cache[k].created_at
            )[:self.max_memory_cache_size // 10]
            
            for old_key in oldest_keys:
                del self.in_memory_cache[old_key]
        
        self.in_memory_cache[cache_key] = entry
    
    async def find_similar_cached_responses(
        self,
        agent_name: str,
        content: str,
        similarity_threshold: float = 0.8
    ) -> List[CacheEntry]:
        """Find cached responses for similar content."""
        
        if not content:
            return []
        
        content_hash = self._create_content_hash(content)
        similar_entries = []
        
        # Search in-memory cache
        for entry in self.in_memory_cache.values():
            if (entry.agent_name == agent_name and 
                entry.content_hash == content_hash and
                datetime.now() <= entry.expires_at):
                similar_entries.append(entry)
        
        return similar_entries
    
    async def invalidate_cache(
        self, 
        agent_name: Optional[str] = None,
        category: Optional[CacheCategory] = None,
        older_than_hours: Optional[int] = None
    ) -> int:
        """Invalidate cache entries based on criteria."""
        
        invalidated_count = 0
        cutoff_time = None
        
        if older_than_hours:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        # Invalidate in-memory cache
        keys_to_remove = []
        for key, entry in self.in_memory_cache.items():
            should_remove = False
            
            if agent_name and entry.agent_name != agent_name:
                continue
            if category and entry.category != category:
                continue
            if cutoff_time and entry.created_at > cutoff_time:
                continue
                
            should_remove = True
            
            if should_remove:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.in_memory_cache[key]
            invalidated_count += 1
        
        logger.info(f"Invalidated {invalidated_count} cache entries")
        return invalidated_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "cache_hits": self.stats["hits"],
            "cache_misses": self.stats["misses"],
            "hit_rate_percentage": round(hit_rate, 2),
            "total_cost_savings": round(self.stats["cost_savings"], 4),
            "cache_size": self.stats["cache_size"],
            "categories": {cat.value: self.ttl_hours[cat] for cat in CacheCategory}
        }
    
    async def cleanup_expired_entries(self) -> int:
        """Clean up expired cache entries."""
        
        now = datetime.now()
        expired_keys = []
        
        # Find expired entries in memory cache
        for key, entry in self.in_memory_cache.items():
            if now > entry.expires_at:
                expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            del self.in_memory_cache[key]
        
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        return len(expired_keys)


# Global cache instance
llm_cache = LLMCache()


async def get_cached_llm_response(
    agent_name: str,
    model: str,
    prompt: str,
    content: str = "",
    extra_params: Dict = None
) -> Optional[Any]:
    """Convenience function to get cached LLM response."""
    return await llm_cache.get_cached_response(agent_name, model, prompt, content, extra_params)


async def cache_llm_response(
    agent_name: str,
    model: str,
    prompt: str,
    response: Any,
    content: str = "",
    extra_params: Dict = None,
    cost_saved: float = 0.0
) -> None:
    """Convenience function to cache LLM response."""
    await llm_cache.cache_response(agent_name, model, prompt, response, content, extra_params, cost_saved)