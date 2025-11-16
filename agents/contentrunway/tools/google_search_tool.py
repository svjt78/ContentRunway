"""Google Custom Search API tool as SearXNG alternative."""

import requests
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class GoogleSearchTool:
    """Google Custom Search API tool for trending content discovery."""
    
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        self.search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
        if not self.api_key or not self.search_engine_id:
            logger.warning("Google Search API credentials not found. Tool will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
    
    async def search_trending_content(
        self,
        domain: str,
        platforms: List[str] = ["twitter", "linkedin"],
        max_results_per_platform: int = 10,
        time_filter: str = "week"
    ) -> List[Dict[str, Any]]:
        """Search for trending content using Google Custom Search."""
        
        if not self.enabled:
            logger.warning("Google Search API not configured")
            return []
        
        all_sources = []
        
        for platform in platforms:
            try:
                platform_sources = await self._search_platform(
                    platform, 
                    domain, 
                    max_results_per_platform,
                    time_filter
                )
                all_sources.extend(platform_sources)
            except Exception as e:
                logger.warning(f"Failed to search {platform} for {domain}: {e}")
        
        return all_sources
    
    async def _search_platform(
        self,
        platform: str,
        domain: str,
        max_results: int,
        time_filter: str
    ) -> List[Dict[str, Any]]:
        """Search a specific platform using Google Custom Search."""
        
        # Create platform-specific search queries
        search_queries = self._generate_search_queries(platform, domain, time_filter)
        platform_sources = []
        
        for query in search_queries[:3]:  # Limit to first 3 queries
            try:
                params = {
                    'key': self.api_key,
                    'cx': self.search_engine_id,
                    'q': query,
                    'num': min(max_results // len(search_queries), 10),  # Google API limit is 10
                    'sort': 'date'  # Sort by recency
                }
                
                response = requests.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                items = data.get('items', [])
                
                for item in items:
                    enhanced_result = self._enhance_search_result(
                        item, platform, domain, query
                    )
                    if enhanced_result:
                        platform_sources.append(enhanced_result)
                        
            except Exception as e:
                logger.warning(f"Google search failed for query '{query}': {e}")
        
        return platform_sources[:max_results]
    
    def _generate_search_queries(
        self,
        platform: str,
        domain: str,
        time_filter: str
    ) -> List[str]:
        """Generate search queries for platform and domain."""
        
        domain_keywords = {
            'it_insurance': ['cyber insurance', 'insurtech', 'digital transformation insurance'],
            'ai': ['artificial intelligence', 'machine learning', 'AI breakthrough'],
            'agentic_ai': ['multi-agent systems', 'agentic AI', 'AI agents'],
            'ai_software_engineering': ['AI coding', 'code generation', 'AI developer tools']
        }
        
        platform_sites = {
            'twitter': 'site:twitter.com OR site:x.com',
            'linkedin': 'site:linkedin.com'
        }
        
        keywords = domain_keywords.get(domain, ['AI', 'technology'])
        site_filter = platform_sites.get(platform, '')
        
        queries = []
        for keyword in keywords[:2]:  # Use top 2 keywords
            query = f'{site_filter} "{keyword}" trending popular'
            if time_filter == 'day':
                query += ' after:2024-01-01'  # Adjust date as needed
            queries.append(query)
        
        return queries
    
    def _enhance_search_result(
        self,
        item: Dict[str, Any],
        platform: str,
        domain: str,
        query: str
    ) -> Optional[Dict[str, Any]]:
        """Enhance Google search result with metadata."""
        
        url = item.get('link', '')
        if not url:
            return None
        
        # Filter for correct platform
        if not self._is_platform_url(url, platform):
            return None
        
        enhanced_result = {
            'url': url,
            'title': item.get('title', ''),
            'summary': item.get('snippet', '')[:500],
            'platform': platform,
            'domain': domain,
            'source_type': 'social_post',
            'engagement_score': 0.7,  # Default score
            'recency_score': 0.8,     # Google sorts by date
            'relevance_score': 0.8,   # Google relevance
            'credibility_score': 0.6,  # Default for social
            'search_query': query,
            'extracted_at': datetime.now(),
            'content_type': 'trending_idea'
        }
        
        return enhanced_result
    
    def _is_platform_url(self, url: str, platform: str) -> bool:
        """Check if URL matches the expected platform."""
        url_lower = url.lower()
        
        platform_domains = {
            'twitter': ['twitter.com', 'x.com'],
            'linkedin': ['linkedin.com']
        }
        
        domains = platform_domains.get(platform, [])
        return any(domain in url_lower for domain in domains)