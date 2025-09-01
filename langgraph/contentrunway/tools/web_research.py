"""Web research tools for gathering sources and information."""

import requests
from bs4 import BeautifulSoup
import feedparser
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import time
import logging
from datetime import datetime
import os
import json

from .searxng_tool import SearXNGTool

logger = logging.getLogger(__name__)


class WebResearchTool:
    """Tool for conducting web research and gathering sources."""
    
    def __init__(self, searxng_base_url: str = "http://localhost/search"):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ContentRunway/1.0 (Research Bot; +https://contentrunway.ai)'
        })
        
        # Initialize SearXNG tool for trending content discovery
        self.searxng_tool = SearXNGTool(searxng_base_url)
        
        # Domain-specific sources
        self.domain_sources = {
            "it_insurance": [
                "https://www.insurancejournal.com/rss/",
                "https://www.dig-in.com/feeds/all",
                "https://www.propertycasualty360.com/feed/",
                "https://feeds.feedburner.com/insurancetech",
            ],
            "ai": [
                "https://feeds.feedburner.com/oreilly/radar",
                "https://blog.openai.com/rss/",
                "https://deepmind.com/blog/rss.xml",
                "https://ai.googleblog.com/feeds/posts/default",
                "https://research.google/pubs/",
            ],
            "agentic_ai": [
                "https://blog.langchain.dev/rss/",
                "https://www.anthropic.com/news/rss",
                "https://arxiv.org/rss/cs.AI",
                "https://arxiv.org/rss/cs.MA",  # Multi-agent systems
            ],
            "ai_software_engineering": [
                "https://github.blog/engineering.atom",
                "https://research.microsoft.com/en-us/feed/",
                "https://engineering.fb.com/feed/",
                "https://blog.google/technology/developers/rss/",
            ]
        }
    
    async def search_domain_sources(
        self,
        query: str,
        domain: str,
        max_sources: int = 10,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Search domain-specific sources for relevant content."""
        # Check cache first if enabled
        if use_cache:
            from app.services.redis_service import redis_service
            cached_results = await redis_service.get_cached_research_results(query, domain)
            if cached_results:
                logger.info(f"Using cached research results for {domain}:{query}")
                return cached_results[:max_sources]
        
        sources = []
        
        # Get RSS feeds for domain
        rss_feeds = self.domain_sources.get(domain, [])
        
        for feed_url in rss_feeds[:5]:  # Limit to prevent overwhelming
            try:
                feed_sources = await self._search_rss_feed(feed_url, query, max_sources // len(rss_feeds))
                sources.extend(feed_sources)
            except Exception as e:
                logger.warning(f"Failed to search RSS feed {feed_url}: {e}")
        
        # Also search arXiv for academic sources if AI-related
        if domain in ["ai", "agentic_ai", "ai_software_engineering"]:
            try:
                arxiv_sources = await self._search_arxiv(query, max_sources // 2)
                sources.extend(arxiv_sources)
            except Exception as e:
                logger.warning(f"Failed to search arXiv: {e}")
        
        final_sources = sources[:max_sources]
        
        # Cache results if enabled and we have results
        if use_cache and final_sources:
            from app.services.redis_service import redis_service
            await redis_service.cache_research_results(query, domain, final_sources)
        
        return final_sources
    
    async def _search_rss_feed(
        self,
        feed_url: str,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search an RSS feed for relevant articles."""
        try:
            response = self.session.get(feed_url, timeout=10)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            sources = []
            
            query_terms = query.lower().split()
            
            for entry in feed.entries[:20]:  # Check first 20 entries
                # Calculate relevance score based on title and description
                title = entry.get('title', '').lower()
                summary = entry.get('summary', '').lower()
                content = f"{title} {summary}"
                
                relevance_score = sum(1 for term in query_terms if term in content) / len(query_terms)
                
                if relevance_score > 0.3:  # Threshold for relevance
                    sources.append({
                        'url': entry.get('link', ''),
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', '')[:500],
                        'publication_date': self._parse_date(entry.get('published')),
                        'author': entry.get('author', ''),
                        'source_type': 'article',
                        'domain': self._extract_domain_from_url(entry.get('link', '')),
                        'relevance_score': relevance_score,
                        'credibility_score': 0.8  # Default for known sources
                    })
                
                if len(sources) >= max_results:
                    break
            
            return sources
            
        except Exception as e:
            logger.error(f"Failed to search RSS feed {feed_url}: {e}")
            return []
    
    async def _search_arxiv(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search arXiv for academic papers."""
        try:
            # arXiv API query
            base_url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f"all:{query}",
                'start': 0,
                'max_results': max_results,
                'sortBy': 'lastUpdatedDate',
                'sortOrder': 'descending'
            }
            
            response = self.session.get(base_url, params=params, timeout=15)
            response.raise_for_status()
            
            # Parse arXiv response
            sources = []
            feed = feedparser.parse(response.content)
            
            for entry in feed.entries:
                sources.append({
                    'url': entry.get('link', ''),
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', '')[:500],
                    'publication_date': self._parse_date(entry.get('published')),
                    'author': ', '.join([author.get('name', '') for author in entry.get('authors', [])]),
                    'source_type': 'paper',
                    'domain': 'arxiv',
                    'relevance_score': 0.9,  # arXiv results are generally highly relevant
                    'credibility_score': 0.95  # High credibility for academic papers
                })
            
            return sources
            
        except Exception as e:
            logger.error(f"Failed to search arXiv: {e}")
            return []
    
    async def fetch_article_content(self, url: str) -> Dict[str, Any]:
        """Fetch and extract main content from an article URL."""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Try to find main content
            content = ""
            
            # Common content selectors
            content_selectors = [
                'article',
                '.entry-content',
                '.post-content', 
                '.article-body',
                '.content',
                'main'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(strip=True)
                    break
            
            # Fallback to body if no specific content found
            if not content:
                content = soup.get_text(strip=True)
            
            # Extract metadata
            title = ""
            title_elem = soup.select_one('title')
            if title_elem:
                title = title_elem.get_text(strip=True)
            
            # Extract key points (first few paragraphs)
            paragraphs = soup.find_all('p')
            key_points = []
            for p in paragraphs[:5]:
                text = p.get_text(strip=True)
                if len(text) > 50:  # Substantial paragraphs only
                    key_points.append(text[:200])
            
            return {
                'title': title,
                'content': content[:5000],  # Limit content length
                'key_points': key_points,
                'word_count': len(content.split()),
                'extracted_at': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch article content from {url}: {e}")
            return {
                'title': '',
                'content': '',
                'key_points': [],
                'word_count': 0,
                'error': str(e)
            }
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str:
            return None
        
        try:
            # Try common date formats
            import dateutil.parser
            return dateutil.parser.parse(date_str)
        except:
            return None
    
    def _extract_domain_from_url(self, url: str) -> str:
        """Extract domain name from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return 'unknown'
    
    async def validate_source_credibility(
        self,
        source: Dict[str, Any]
    ) -> float:
        """Validate the credibility of a source."""
        score = 0.5  # Base score
        
        domain = source.get('domain', '').lower()
        
        # Known high-credibility domains
        high_credibility_domains = {
            'arxiv.org': 0.95,
            'nature.com': 0.95,
            'science.org': 0.95,
            'ieee.org': 0.90,
            'acm.org': 0.90,
            'openai.com': 0.85,
            'deepmind.com': 0.85,
            'research.google.com': 0.85,
            'microsoft.com': 0.80,
            'insurancejournal.com': 0.85,
            'propertycasualty360.com': 0.80,
        }
        
        # Check for known credible domains
        for credible_domain, credibility in high_credibility_domains.items():
            if credible_domain in domain:
                score = credibility
                break
        
        # Adjust based on source type
        source_type = source.get('source_type', '')
        if source_type == 'paper':
            score += 0.1
        elif source_type == 'article' and 'blog' in domain:
            score -= 0.1
        
        # Check for author information
        if source.get('author') and len(source['author']) > 5:
            score += 0.05
        
        # Check publication date (recent = better)
        pub_date = source.get('publication_date')
        if pub_date and isinstance(pub_date, datetime):
            days_old = (datetime.now() - pub_date).days
            if days_old < 30:
                score += 0.1
            elif days_old < 180:
                score += 0.05
        
        return min(1.0, max(0.1, score))  # Clamp between 0.1 and 1.0
    
    async def search_trending_topics(
        self,
        base_topic: str,
        domain: str,
        platforms: List[str] = ["twitter", "linkedin"],
        max_results: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Search for trending content ideas from social platforms.
        
        Args:
            base_topic: Base research topic
            domain: Target domain
            platforms: Social platforms to search
            max_results: Maximum number of trending results to return
            
        Returns:
            List of trending content sources and topic ideas
        """
        logger.info(f"Searching trending topics for '{base_topic}' in domain '{domain}'")
        
        try:
            # Search for trending content using SearXNG
            trending_sources = await self.searxng_tool.search_trending_content(
                domain=domain,
                platforms=platforms,
                max_results_per_platform=max_results // len(platforms)
            )
            
            # Extract trending topic ideas
            trending_topics = await self.searxng_tool.extract_trending_topics(
                trending_sources,
                min_engagement_score=0.6
            )
            
            # Combine trending sources and topics
            all_trending_content = []
            
            # Add trending sources (social posts/discussions)
            for source in trending_sources:
                all_trending_content.append({
                    'url': source.get('url', ''),
                    'title': source.get('title', ''),
                    'summary': source.get('summary', ''),
                    'platform': source.get('platform', ''),
                    'domain': source.get('domain', ''),
                    'source_type': 'trending_post',
                    'engagement_score': source.get('engagement_score', 0),
                    'recency_score': source.get('recency_score', 0),
                    'relevance_score': source.get('relevance_score', 0),
                    'credibility_score': source.get('credibility_score', 0.6),
                    'publication_date': source.get('extracted_at'),
                    'author': '',  # Social platforms don't always provide clear authorship
                    'trending_context': source.get('search_query', '')
                })
            
            # Add trending topic ideas
            for topic in trending_topics:
                all_trending_content.append({
                    'url': '',  # Topic ideas don't have single URLs
                    'title': topic.get('title', ''),
                    'summary': topic.get('description', ''),
                    'platform': 'multiple',
                    'domain': topic.get('domain', ''),
                    'source_type': 'trending_topic_idea',
                    'engagement_score': topic.get('engagement_score', 0),
                    'recency_score': topic.get('recency_score', 0),
                    'relevance_score': topic.get('relevance_score', 0),
                    'credibility_score': topic.get('credibility_score', 0.6),
                    'publication_date': topic.get('extracted_at'),
                    'author': '',
                    'trending_keywords': topic.get('trending_keywords', []),
                    'cluster_size': topic.get('cluster_size', 1),
                    'source_urls': topic.get('source_urls', [])
                })
            
            logger.info(f"Found {len(all_trending_content)} trending content items")
            return all_trending_content[:max_results]
            
        except Exception as e:
            logger.error(f"Trending topics search failed: {e}")
            return []
    
    async def enhanced_domain_search(
        self,
        query: str,
        domain: str,
        include_trending: bool = True,
        max_sources: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Enhanced domain search that combines traditional sources with trending content.
        
        Args:
            query: Search query
            domain: Target domain
            include_trending: Whether to include trending social content
            max_sources: Maximum total sources to return
            
        Returns:
            Combined list of traditional and trending sources
        """
        
        # Get traditional sources (RSS + arXiv)
        traditional_sources = await self.search_domain_sources(
            query, 
            domain, 
            max_sources=max_sources // 2 if include_trending else max_sources
        )
        
        all_sources = traditional_sources
        
        # Add trending content if requested
        if include_trending:
            trending_sources = await self.search_trending_topics(
                base_topic=query,
                domain=domain,
                max_results=max_sources // 2
            )
            all_sources.extend(trending_sources)
        
        # Sort by combined score (relevance + credibility + engagement)
        for source in all_sources:
            # Calculate combined score
            relevance = source.get('relevance_score', 0.5)
            credibility = source.get('credibility_score', 0.5)
            engagement = source.get('engagement_score', 0.5)
            recency = source.get('recency_score', 0.5)
            
            # Weight traditional sources vs trending content
            if source.get('source_type') in ['trending_post', 'trending_topic_idea']:
                # Trending content: emphasize engagement and recency
                source['combined_score'] = (
                    relevance * 0.3 + 
                    credibility * 0.2 + 
                    engagement * 0.3 + 
                    recency * 0.2
                )
            else:
                # Traditional sources: emphasize credibility and relevance  
                source['combined_score'] = (
                    relevance * 0.4 + 
                    credibility * 0.4 + 
                    engagement * 0.1 + 
                    recency * 0.1
                )
        
        # Sort by combined score and return top results
        all_sources.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        logger.info(f"Enhanced search returned {len(all_sources)} sources ({len(traditional_sources)} traditional, {len(trending_sources) if include_trending else 0} trending)")
        
        return all_sources[:max_sources]
    
    
    
