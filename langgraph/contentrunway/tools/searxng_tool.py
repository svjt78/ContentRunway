"""SearXNG integration tool for trending content discovery from social platforms."""

import requests
import json
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode, quote_plus
import logging
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


class SearXNGTool:
    """Tool for conducting web search using SearXNG to find trending content from social platforms."""
    
    def __init__(self, searxng_base_url: str = "http://localhost/search"):
        self.base_url = searxng_base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ContentRunway/1.0 (Trend Research Bot; +https://contentrunway.ai)',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        # Creative search templates for different platforms and domains
        self.search_templates = {
            "twitter": {
                "it_insurance": [
                    "site:twitter.com \"cyber insurance\" trending discussions 2024",
                    "site:twitter.com \"insurtech\" hot topics latest trends",
                    "site:twitter.com \"digital transformation insurance\" popular posts",
                    "site:twitter.com \"IT risk insurance\" viral discussions",
                ],
                "ai": [
                    "site:twitter.com \"artificial intelligence\" trending discussions latest",
                    "site:twitter.com \"machine learning\" viral posts popular",
                    "site:twitter.com \"AI breakthrough\" hot topics trending",
                    "site:twitter.com \"LLM\" latest discussions popular posts",
                ],
                "agentic_ai": [
                    "site:twitter.com \"multi-agent systems\" trending discussions",
                    "site:twitter.com \"agentic AI\" popular posts latest",
                    "site:twitter.com \"AI agents\" viral discussions trending",
                    "site:twitter.com \"agent orchestration\" hot topics",
                ],
                "ai_software_engineering": [
                    "site:twitter.com \"AI coding\" trending discussions latest",
                    "site:twitter.com \"code generation\" popular posts viral",
                    "site:twitter.com \"AI developer tools\" hot topics",
                    "site:twitter.com \"copilot\" trending discussions",
                ]
            },
            "linkedin": {
                "it_insurance": [
                    "site:linkedin.com \"cyber insurance trends\" popular posts industry",
                    "site:linkedin.com \"insurtech innovation\" trending discussions",
                    "site:linkedin.com \"insurance digital transformation\" thought leaders",
                    "site:linkedin.com \"IT security insurance\" industry insights",
                ],
                "ai": [
                    "site:linkedin.com \"AI industry trends\" thought leaders popular",
                    "site:linkedin.com \"machine learning business\" trending posts",
                    "site:linkedin.com \"artificial intelligence ROI\" industry discussions",
                    "site:linkedin.com \"AI transformation\" popular insights",
                ],
                "agentic_ai": [
                    "site:linkedin.com \"autonomous agents\" industry discussions",
                    "site:linkedin.com \"multi-agent AI\" thought leaders trending",
                    "site:linkedin.com \"agent-based systems\" popular posts",
                    "site:linkedin.com \"AI orchestration\" industry insights",
                ],
                "ai_software_engineering": [
                    "site:linkedin.com \"AI development tools\" trending discussions",
                    "site:linkedin.com \"automated coding\" popular insights",
                    "site:linkedin.com \"AI-assisted development\" industry trends",
                    "site:linkedin.com \"developer productivity AI\" thought leaders",
                ]
            }
        }
    
    async def search_trending_content(
        self,
        domain: str,
        platforms: List[str] = ["twitter", "linkedin"],
        max_results_per_platform: int = 10,
        time_filter: str = "week"
    ) -> List[Dict[str, Any]]:
        """
        Search for trending content ideas from social platforms.
        
        Args:
            domain: Domain to search for (it_insurance, ai, agentic_ai, ai_software_engineering)
            platforms: List of platforms to search (twitter, linkedin)
            max_results_per_platform: Maximum results per platform
            time_filter: Time filter (day, week, month)
            
        Returns:
            List of trending content sources with metadata
        """
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
        
        # Sort by engagement and recency
        all_sources.sort(key=lambda x: (x.get('engagement_score', 0), x.get('recency_score', 0)), reverse=True)
        
        return all_sources
    
    async def _search_platform(
        self,
        platform: str,
        domain: str,
        max_results: int,
        time_filter: str
    ) -> List[Dict[str, Any]]:
        """Search a specific platform for trending content."""
        
        if platform not in self.search_templates:
            logger.warning(f"No search templates for platform: {platform}")
            return []
        
        if domain not in self.search_templates[platform]:
            logger.warning(f"No search templates for domain {domain} on {platform}")
            return []
        
        search_queries = self.search_templates[platform][domain]
        platform_sources = []
        
        # Use multiple search queries to get diverse results
        for i, query_template in enumerate(search_queries[:3]):  # Use first 3 templates
            try:
                # Add time filter to query
                time_filtered_query = self._add_time_filter(query_template, time_filter)
                
                results = await self._execute_searxng_search(
                    time_filtered_query,
                    max_results // len(search_queries)
                )
                
                # Process and enhance results
                for result in results:
                    enhanced_result = await self._enhance_social_result(
                        result, 
                        platform, 
                        domain, 
                        query_template
                    )
                    if enhanced_result:
                        platform_sources.append(enhanced_result)
                        
            except Exception as e:
                logger.warning(f"Failed search query '{query_template}': {e}")
        
        return platform_sources[:max_results]
    
    def _add_time_filter(self, query: str, time_filter: str) -> str:
        """Add time filtering to search query."""
        time_map = {
            "day": "after:1d",
            "week": "after:7d", 
            "month": "after:30d"
        }
        
        time_param = time_map.get(time_filter, "after:7d")
        return f"{query} {time_param}"
    
    async def _execute_searxng_search(
        self,
        query: str,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Execute search query against SearXNG."""
        
        # SearXNG search parameters
        params = {
            'q': query,
            'format': 'json',
            'engines': 'google,bing,duckduckgo',
            'categories': 'general',
            'safesearch': 0,
            'pageno': 1
        }
        
        try:
            search_url = f"{self.base_url}/"
            response = self.session.get(search_url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            results = data.get('results', [])
            
            # Filter and process results
            processed_results = []
            for result in results[:max_results]:
                processed_result = {
                    'url': result.get('url', ''),
                    'title': result.get('title', ''),
                    'content': result.get('content', ''),
                    'engine': result.get('engine', ''),
                    'score': result.get('score', 0),
                    'category': result.get('category', ''),
                    'pretty_url': result.get('pretty_url', ''),
                    'template': result.get('template', ''),
                }
                processed_results.append(processed_result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"SearXNG search failed for query '{query}': {e}")
            return []
    
    async def _enhance_social_result(
        self,
        result: Dict[str, Any],
        platform: str,
        domain: str,
        original_query: str
    ) -> Optional[Dict[str, Any]]:
        """Enhance search result with social platform metadata."""
        
        url = result.get('url', '')
        if not url:
            return None
        
        # Filter out non-social platform URLs
        if not self._is_social_platform_url(url, platform):
            return None
        
        # Extract engagement indicators from title/content
        engagement_score = self._calculate_engagement_score(result)
        recency_score = self._calculate_recency_score(result)
        
        # Calculate relevance to domain
        relevance_score = self._calculate_domain_relevance(result, domain, original_query)
        
        enhanced_result = {
            'url': url,
            'title': result.get('title', ''),
            'summary': result.get('content', '')[:500],
            'platform': platform,
            'domain': domain,
            'source_type': 'social_post',
            'engagement_score': engagement_score,
            'recency_score': recency_score,
            'relevance_score': relevance_score,
            'credibility_score': self._calculate_social_credibility(url, platform),
            'search_query': original_query,
            'extracted_at': datetime.now(),
            'content_type': 'trending_idea'
        }
        
        return enhanced_result
    
    def _is_social_platform_url(self, url: str, platform: str) -> bool:
        """Check if URL is from the expected social platform."""
        url_lower = url.lower()
        
        platform_domains = {
            'twitter': ['twitter.com', 'x.com'],
            'linkedin': ['linkedin.com']
        }
        
        domains = platform_domains.get(platform, [])
        return any(domain in url_lower for domain in domains)
    
    def _calculate_engagement_score(self, result: Dict[str, Any]) -> float:
        """Calculate engagement score based on content indicators."""
        score = 0.5  # Base score
        
        title = result.get('title', '').lower()
        content = result.get('content', '').lower()
        combined_text = f"{title} {content}"
        
        # Engagement indicators
        engagement_keywords = [
            'viral', 'trending', 'popular', 'hot', 'breaking', 'latest',
            'likes', 'shares', 'retweets', 'comments', 'replies',
            'discussion', 'thread', 'debate', 'insights'
        ]
        
        # Count engagement indicators
        indicator_count = sum(1 for keyword in engagement_keywords if keyword in combined_text)
        score += min(0.3, indicator_count * 0.05)
        
        # Check for numbers (could indicate metrics)
        import re
        numbers = re.findall(r'\d+[km]?\s*(?:likes|shares|retweets|comments)', combined_text)
        if numbers:
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_recency_score(self, result: Dict[str, Any]) -> float:
        """Calculate recency score (higher for more recent content)."""
        # Since SearXNG doesn't always provide timestamps,
        # we estimate based on search result ranking and content indicators
        
        score = 0.7  # Default score
        
        title = result.get('title', '').lower()
        content = result.get('content', '').lower()
        combined_text = f"{title} {content}"
        
        # Recency indicators
        recent_keywords = [
            'today', 'yesterday', 'this week', 'recent', 'latest', 'new',
            '2024', 'just', 'now', 'breaking', 'update', 'current'
        ]
        
        recent_count = sum(1 for keyword in recent_keywords if keyword in combined_text)
        score += min(0.2, recent_count * 0.05)
        
        # Higher score for higher search ranking (SearXNG score)
        searxng_score = result.get('score', 0)
        if searxng_score > 80:
            score += 0.1
        elif searxng_score > 60:
            score += 0.05
        
        return min(1.0, score)
    
    def _calculate_domain_relevance(
        self,
        result: Dict[str, Any],
        domain: str,
        query: str
    ) -> float:
        """Calculate how relevant the result is to the target domain."""
        
        title = result.get('title', '').lower()
        content = result.get('content', '').lower()
        combined_text = f"{title} {content}"
        
        # Domain-specific keywords
        domain_keywords = {
            'it_insurance': [
                'cyber insurance', 'insurtech', 'digital transformation', 
                'insurance technology', 'risk management', 'cybersecurity insurance'
            ],
            'ai': [
                'artificial intelligence', 'machine learning', 'neural networks',
                'deep learning', 'AI models', 'automation', 'AI applications'
            ],
            'agentic_ai': [
                'multi-agent', 'agent systems', 'autonomous agents', 'langgraph',
                'agent orchestration', 'react patterns', 'AI agents'
            ],
            'ai_software_engineering': [
                'AI coding', 'code generation', 'developer tools', 'programming AI',
                'automated development', 'AI-assisted coding', 'copilot'
            ]
        }
        
        domain_terms = domain_keywords.get(domain, [])
        if not domain_terms:
            return 0.5
        
        # Calculate relevance based on keyword presence
        matches = sum(1 for term in domain_terms if term in combined_text)
        relevance_score = min(1.0, matches / len(domain_terms) + 0.3)
        
        # Boost score if query terms are present
        query_terms = query.lower().split()
        query_matches = sum(1 for term in query_terms if term in combined_text)
        if query_matches > 0:
            relevance_score += min(0.2, query_matches * 0.05)
        
        return min(1.0, relevance_score)
    
    def _calculate_social_credibility(self, url: str, platform: str) -> float:
        """Calculate credibility score for social media content."""
        
        # Base credibility for social platforms (lower than traditional sources)
        base_scores = {
            'twitter': 0.6,  # Medium credibility due to real-time nature
            'linkedin': 0.7,  # Higher credibility for professional content
        }
        
        score = base_scores.get(platform, 0.5)
        
        # Adjust based on URL patterns (verified accounts, company pages, etc.)
        url_lower = url.lower()
        
        # LinkedIn company pages and verified profiles
        if platform == 'linkedin':
            if '/company/' in url_lower or '/in/' in url_lower:
                score += 0.1
        
        # Check for potential spam indicators
        spam_indicators = ['bit.ly', 'tinyurl', 'goo.gl', 'short.link']
        if any(indicator in url_lower for indicator in spam_indicators):
            score -= 0.2
        
        return min(1.0, max(0.1, score))
    
    async def generate_creative_search_queries(
        self,
        base_topic: str,
        domain: str,
        platforms: List[str] = ["twitter", "linkedin"]
    ) -> List[str]:
        """
        Generate creative search queries for finding trending content ideas.
        
        Args:
            base_topic: Base research topic
            domain: Target domain
            platforms: Platforms to search
            
        Returns:
            List of creative search query strings
        """
        
        queries = []
        
        # Time-based variations
        time_variations = [
            "latest trends in",
            "what's trending in",
            "hot topics in",
            "popular discussions about",
            "viral posts about",
            "breaking news in"
        ]
        
        # Engagement variations  
        engagement_variations = [
            "most liked posts about",
            "trending discussions on",
            "popular threads about",
            "viral content on",
            "hot takes on",
            "industry buzz about"
        ]
        
        for platform in platforms:
            if platform in self.search_templates and domain in self.search_templates[platform]:
                # Use predefined templates
                queries.extend(self.search_templates[platform][domain])
            
            # Generate dynamic queries
            for variation in time_variations[:2]:  # Use first 2 variations
                query = f"site:{platform}.com \"{variation} {base_topic}\" {domain}"
                queries.append(query)
            
            for variation in engagement_variations[:2]:  # Use first 2 variations  
                query = f"site:{platform}.com \"{variation} {base_topic}\" trending"
                queries.append(query)
        
        # Shuffle to randomize search order
        random.shuffle(queries)
        
        return queries[:8]  # Return top 8 creative queries
    
    async def extract_trending_topics(
        self,
        search_results: List[Dict[str, Any]],
        min_engagement_score: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Extract trending topic ideas from search results.
        
        Args:
            search_results: Raw search results from SearXNG
            min_engagement_score: Minimum engagement score threshold
            
        Returns:
            List of trending topic ideas with metadata
        """
        
        trending_topics = []
        
        # Filter by engagement score
        high_engagement_results = [
            result for result in search_results 
            if result.get('engagement_score', 0) >= min_engagement_score
        ]
        
        # Group similar topics
        topic_clusters = self._cluster_similar_topics(high_engagement_results)
        
        # Extract key insights from each cluster
        for cluster in topic_clusters:
            topic_idea = await self._extract_topic_from_cluster(cluster)
            if topic_idea:
                trending_topics.append(topic_idea)
        
        return trending_topics
    
    def _cluster_similar_topics(
        self,
        results: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Group similar search results into topic clusters."""
        
        clusters = []
        used_indices = set()
        
        for i, result in enumerate(results):
            if i in used_indices:
                continue
                
            cluster = [result]
            used_indices.add(i)
            
            # Find similar results
            for j, other_result in enumerate(results[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                if self._are_topics_similar(result, other_result):
                    cluster.append(other_result)
                    used_indices.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _are_topics_similar(
        self,
        result1: Dict[str, Any],
        result2: Dict[str, Any]
    ) -> bool:
        """Check if two search results represent similar topics."""
        
        title1 = result1.get('title', '').lower()
        title2 = result2.get('title', '').lower()
        
        # Simple keyword overlap check
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words1 -= common_words
        words2 -= common_words
        
        if not words1 or not words2:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity > 0.3  # 30% similarity threshold
    
    async def _extract_topic_from_cluster(
        self,
        cluster: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract a trending topic idea from a cluster of similar results."""
        
        if not cluster:
            return None
        
        # Use the highest engagement result as primary
        primary_result = max(cluster, key=lambda x: x.get('engagement_score', 0))
        
        # Aggregate information from cluster
        all_titles = [result.get('title', '') for result in cluster]
        all_content = [result.get('summary', '') for result in cluster]
        
        # Calculate cluster metrics
        avg_engagement = sum(r.get('engagement_score', 0) for r in cluster) / len(cluster)
        avg_recency = sum(r.get('recency_score', 0) for r in cluster) / len(cluster)
        avg_relevance = sum(r.get('relevance_score', 0) for r in cluster) / len(cluster)
        
        # Extract common themes and keywords
        common_keywords = self._extract_common_keywords(all_titles + all_content)
        
        topic_idea = {
            'title': primary_result.get('title', ''),
            'description': primary_result.get('summary', ''),
            'platform': primary_result.get('platform', ''),
            'domain': primary_result.get('domain', ''),
            'source_urls': [r.get('url', '') for r in cluster],
            'trending_keywords': common_keywords,
            'engagement_score': avg_engagement,
            'recency_score': avg_recency,
            'relevance_score': avg_relevance,
            'cluster_size': len(cluster),
            'credibility_score': primary_result.get('credibility_score', 0.6),
            'content_type': 'trending_topic',
            'extracted_at': datetime.now()
        }
        
        return topic_idea
    
    def _extract_common_keywords(self, texts: List[str], min_frequency: int = 2) -> List[str]:
        """Extract commonly occurring keywords from a list of texts."""
        
        from collections import Counter
        import re
        
        # Combine all texts
        combined_text = ' '.join(texts).lower()
        
        # Extract meaningful words (3+ characters, alphanumeric)
        words = re.findall(r'\b[a-z]{3,}\b', combined_text)
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'men', 'use', 'way', 'will', 'with'
        }
        
        meaningful_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count frequency and return top keywords
        word_counts = Counter(meaningful_words)
        common_keywords = [word for word, count in word_counts.most_common(10) if count >= min_frequency]
        
        return common_keywords[:5]  # Return top 5 keywords