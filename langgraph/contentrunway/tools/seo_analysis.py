"""SEO analysis tools for keyword research, SERP analysis, and optimization."""

import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Optional
import logging
import time
import re
from urllib.parse import urljoin, urlparse, quote
from collections import Counter
import json

logger = logging.getLogger(__name__)


class SEOAnalysisTool:
    """Tool for SEO analysis including keyword research, SERP analysis, and optimization recommendations."""
    
    def __init__(self):
        """Initialize SEO analysis tool."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Keyword difficulty estimation data
        self.competitive_domains = {
            'high_authority': [
                'wikipedia.org', 'youtube.com',
                'stackoverflow.com', 'github.com', 'reddit.com', 'forbes.com',
                'techcrunch.com', 'harvard.edu', 'mit.edu'
            ],
            'medium_authority': [
                'dev.to', 'hackernoon.com', 'towards-data-science.com',
                'freecodecamp.org', 'digitalocean.com', 'aws.amazon.com'
            ]
        }
        
        # Search intent patterns
        self.intent_patterns = {
            'informational': [
                'what is', 'how to', 'why', 'when', 'where', 'guide', 'tutorial',
                'explain', 'definition', 'meaning', 'learn', 'understand'
            ],
            'commercial': [
                'best', 'top', 'review', 'compare', 'vs', 'alternative',
                'pricing', 'cost', 'buy', 'purchase', 'service', 'tool'
            ],
            'navigational': [
                'login', 'download', 'official', 'website', 'homepage',
                'contact', 'support', 'documentation', 'api'
            ],
            'transactional': [
                'buy', 'purchase', 'order', 'signup', 'subscribe',
                'trial', 'demo', 'quote', 'price', 'discount'
            ]
        }
    
    def analyze_keyword_difficulty(self, keyword: str, domain_context: str = "") -> Dict[str, Any]:
        """
        Analyze keyword difficulty and competition.
        
        Args:
            keyword: Target keyword to analyze
            domain_context: Domain context (ai, insurance, etc.)
            
        Returns:
            Dictionary with keyword difficulty metrics
        """
        logger.info(f"Analyzing keyword difficulty for: {keyword}")
        
        try:
            # Step 1: Perform SERP analysis
            serp_results = self._perform_serp_analysis(keyword)
            
            # Step 2: Analyze competitor strength
            competitor_analysis = self._analyze_competitor_strength(serp_results)
            
            # Step 3: Estimate keyword metrics
            keyword_metrics = self._estimate_keyword_metrics(keyword, domain_context)
            
            # Step 4: Calculate difficulty score
            difficulty_score = self._calculate_difficulty_score(
                competitor_analysis,
                keyword_metrics,
                len(serp_results.get('organic_results', []))
            )
            
            # Step 5: Generate keyword recommendations
            recommendations = self._generate_keyword_recommendations(
                keyword,
                difficulty_score,
                serp_results,
                domain_context
            )
            
            return {
                'keyword': keyword,
                'difficulty_score': difficulty_score,
                'difficulty_level': self._categorize_difficulty(difficulty_score),
                'search_volume_estimate': keyword_metrics.get('search_volume_estimate'),
                'competition_level': competitor_analysis.get('competition_level'),
                'serp_features': serp_results.get('features', []),
                'top_competitors': competitor_analysis.get('top_domains', []),
                'recommendations': recommendations,
                'search_intent': keyword_metrics.get('search_intent'),
                'related_keywords': keyword_metrics.get('related_keywords', []),
                'timestamp': serp_results.get('timestamp')
            }
            
        except Exception as e:
            logger.error(f"Keyword difficulty analysis failed: {e}")
            return {
                'keyword': keyword,
                'difficulty_score': 0.5,  # Medium difficulty fallback
                'difficulty_level': 'medium',
                'error': str(e)
            }
    
    def _perform_serp_analysis(self, keyword: str) -> Dict[str, Any]:
        """Perform search engine results page analysis."""
        
        try:
            # Encode search query
            encoded_keyword = quote(keyword)
            search_url = f"https://www.google.com/search?q={encoded_keyword}&num=20"
            
            # Add delay to respect rate limits
            time.sleep(1)
            
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract organic results
            organic_results = []
            result_selectors = [
                'div.g',  # Standard result container
                'div[data-ved]',  # Alternative result container
                '.tF2Cxc'  # Another common result container
            ]
            
            for selector in result_selectors:
                results = soup.select(selector)
                if results:
                    break
            
            for i, result in enumerate(results[:10]):  # Top 10 results
                try:
                    # Extract title
                    title_elem = result.select_one('h3, .LC20lb, .DKV0Md')
                    title = title_elem.get_text(strip=True) if title_elem else ""
                    
                    # Extract URL
                    link_elem = result.select_one('a[href]')
                    url = link_elem.get('href', '') if link_elem else ""
                    
                    # Clean URL (remove Google redirect)
                    if url.startswith('/url?q='):
                        url = url.split('&')[0].replace('/url?q=', '')
                    
                    # Extract snippet
                    snippet_elem = result.select_one('.VwiC3b, .s3v9rd, .IsZvec')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    if title and url:
                        organic_results.append({
                            'position': i + 1,
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'domain': self._extract_domain(url)
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to parse result {i}: {e}")
                    continue
            
            # Detect SERP features
            serp_features = self._detect_serp_features(soup)
            
            return {
                'organic_results': organic_results,
                'features': serp_features,
                'total_results_found': len(organic_results),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"SERP analysis failed for '{keyword}': {e}")
            return {
                'organic_results': [],
                'features': [],
                'total_results_found': 0,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _detect_serp_features(self, soup: BeautifulSoup) -> List[str]:
        """Detect SERP features that might affect ranking difficulty."""
        
        features = []
        
        # Featured snippet
        if soup.select('.xpdopen, .kp-blk, .IZ6rdc'):
            features.append('featured_snippet')
        
        # Knowledge panel
        if soup.select('.kp-wholepage, .knowledge-panel'):
            features.append('knowledge_panel')
        
        # People Also Ask
        if soup.select('.related-question-pair, .cbphWd'):
            features.append('people_also_ask')
        
        # Images
        if soup.select('.tn-tnc, .images_table'):
            features.append('image_pack')
        
        # News results
        if soup.select('.SoAPf, .TTmAKf'):
            features.append('news_results')
        
        # Video results
        if soup.select('.P94G9b, .OmQG3b'):
            features.append('video_results')
        
        # Local pack
        if soup.select('.rllt__details, .VkpGBb'):
            features.append('local_pack')
        
        # Shopping results
        if soup.select('.pla-unit, .PLla9e'):
            features.append('shopping_results')
        
        return features
    
    def _analyze_competitor_strength(self, serp_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the strength of competitors in search results."""
        
        organic_results = serp_results.get('organic_results', [])
        
        if not organic_results:
            return {
                'competition_level': 'unknown',
                'top_domains': [],
                'authority_score': 0.5
            }
        
        # Analyze domains in top 10 results
        domains = [result['domain'] for result in organic_results if result.get('domain')]
        domain_counts = Counter(domains)
        
        # Calculate authority scores
        high_authority_count = 0
        medium_authority_count = 0
        
        for domain in domains:
            if any(auth_domain in domain for auth_domain in self.competitive_domains['high_authority']):
                high_authority_count += 1
            elif any(auth_domain in domain for auth_domain in self.competitive_domains['medium_authority']):
                medium_authority_count += 1
        
        # Calculate competition level
        total_results = len(organic_results)
        high_authority_ratio = high_authority_count / total_results if total_results > 0 else 0
        
        if high_authority_ratio >= 0.7:
            competition_level = 'very_high'
            authority_score = 0.9
        elif high_authority_ratio >= 0.5:
            competition_level = 'high'
            authority_score = 0.8
        elif high_authority_ratio >= 0.3:
            competition_level = 'medium'
            authority_score = 0.6
        else:
            competition_level = 'low'
            authority_score = 0.4
        
        return {
            'competition_level': competition_level,
            'authority_score': authority_score,
            'high_authority_domains': high_authority_count,
            'medium_authority_domains': medium_authority_count,
            'top_domains': [domain for domain, count in domain_counts.most_common(5)],
            'domain_diversity': len(set(domains)),
            'serp_features_count': len(serp_results.get('features', []))
        }
    
    def _estimate_keyword_metrics(self, keyword: str, domain_context: str) -> Dict[str, Any]:
        """Estimate keyword metrics including search volume and intent."""
        
        # Basic keyword analysis
        keyword_lower = keyword.lower()
        word_count = len(keyword.split())
        
        # Estimate search volume based on keyword characteristics
        search_volume_estimate = self._estimate_search_volume(keyword, domain_context)
        
        # Determine search intent
        search_intent = self._determine_search_intent(keyword)
        
        # Generate related keywords
        related_keywords = self._generate_related_keywords(keyword, domain_context)
        
        # Commercial intent scoring
        commercial_indicators = ['buy', 'price', 'cost', 'review', 'best', 'top', 'compare']
        commercial_score = sum(1 for indicator in commercial_indicators if indicator in keyword_lower)
        commercial_intent = commercial_score > 0
        
        return {
            'search_volume_estimate': search_volume_estimate,
            'search_intent': search_intent,
            'commercial_intent': commercial_intent,
            'keyword_length': word_count,
            'related_keywords': related_keywords,
            'long_tail': word_count >= 3,
            'branded': any(brand in keyword_lower for brand in ['openai', 'google', 'microsoft', 'ibm', 'aws'])
        }
    
    def _estimate_search_volume(self, keyword: str, domain_context: str) -> str:
        """Estimate search volume category based on keyword characteristics."""
        
        word_count = len(keyword.split())
        keyword_lower = keyword.lower()
        
        # Domain-specific volume adjustments
        niche_multiplier = 1.0
        if domain_context in ['ai', 'agentic_ai']:
            niche_multiplier = 0.7  # AI topics are more niche
        elif domain_context == 'it_insurance':
            niche_multiplier = 0.5  # Very niche industry
        
        # Base estimation logic
        if word_count == 1:
            # Single word keywords tend to be high volume
            base_volume = 'high'
        elif word_count == 2:
            # Two-word keywords are medium volume
            base_volume = 'medium'
        else:
            # Long-tail keywords are typically low volume
            base_volume = 'low'
        
        # Adjust for common terms
        common_tech_terms = [
            'ai', 'artificial intelligence', 'machine learning', 'software',
            'technology', 'digital', 'automation', 'cloud', 'data'
        ]
        
        if any(term in keyword_lower for term in common_tech_terms):
            # Common tech terms boost volume
            if base_volume == 'low':
                base_volume = 'medium'
            elif base_volume == 'medium':
                base_volume = 'high'
        
        # Apply niche multiplier
        if niche_multiplier < 0.7 and base_volume == 'high':
            base_volume = 'medium'
        elif niche_multiplier < 0.7 and base_volume == 'medium':
            base_volume = 'low'
        
        return base_volume
    
    def _determine_search_intent(self, keyword: str) -> str:
        """Determine the primary search intent for a keyword."""
        
        keyword_lower = keyword.lower()
        
        # Check each intent type
        for intent, patterns in self.intent_patterns.items():
            if any(pattern in keyword_lower for pattern in patterns):
                return intent
        
        # Default to informational for most content topics
        return 'informational'
    
    def _generate_related_keywords(self, keyword: str, domain_context: str) -> List[str]:
        """Generate related keywords based on the main keyword and domain."""
        
        related = []
        keyword_words = keyword.lower().split()
        
        # Domain-specific keyword extensions
        domain_extensions = {
            'ai': [
                'artificial intelligence', 'machine learning', 'deep learning',
                'neural networks', 'nlp', 'computer vision', 'ai applications',
                'ai trends', 'ai tools', 'ai development'
            ],
            'agentic_ai': [
                'ai agents', 'multi-agent systems', 'autonomous ai',
                'agent orchestration', 'langgraph', 'langchain',
                'react patterns', 'ai workflows', 'agent frameworks'
            ],
            'it_insurance': [
                'insurtech', 'digital transformation', 'cyber insurance',
                'insurance technology', 'fintech', 'regulatory compliance',
                'risk management', 'digital claims', 'insurance automation'
            ],
            'ai_software_engineering': [
                'ai coding', 'code generation', 'ai development tools',
                'software engineering ai', 'developer ai', 'coding assistant',
                'ai programming', 'automated coding', 'ai software development'
            ]
        }
        
        extensions = domain_extensions.get(domain_context, [])
        
        # Create related keywords
        for extension in extensions[:8]:  # Limit to 8 related keywords
            # Combine with main keyword words
            if keyword_words[0] not in extension:
                related.append(f"{keyword} {extension}")
            related.append(extension)
        
        # Add question-based variations
        question_prefixes = ['what is', 'how to', 'why', 'best practices for']
        for prefix in question_prefixes[:2]:  # Limit to 2 questions
            related.append(f"{prefix} {keyword}")
        
        # Add modifier variations
        modifiers = ['guide', 'tutorial', 'best practices', 'trends', 'tools', 'examples']
        for modifier in modifiers[:3]:  # Limit to 3 modifiers
            related.append(f"{keyword} {modifier}")
        
        return list(set(related))[:15]  # Return unique keywords, max 15
    
    def _calculate_difficulty_score(
        self,
        competitor_analysis: Dict[str, Any],
        keyword_metrics: Dict[str, Any],
        serp_count: int
    ) -> float:
        """Calculate overall keyword difficulty score (0.0 = easy, 1.0 = very hard)."""
        
        # Base difficulty from competitor authority
        authority_score = competitor_analysis.get('authority_score', 0.5)
        
        # Adjust for keyword characteristics
        keyword_length = keyword_metrics.get('keyword_length', 2)
        commercial_intent = keyword_metrics.get('commercial_intent', False)
        
        # Long-tail keywords are generally easier
        length_multiplier = max(0.5, 1.0 - (keyword_length - 2) * 0.15)
        
        # Commercial intent increases difficulty
        commercial_multiplier = 1.2 if commercial_intent else 1.0
        
        # SERP features increase difficulty
        features_count = competitor_analysis.get('serp_features_count', 0)
        features_multiplier = 1.0 + (features_count * 0.05)
        
        # Calculate final difficulty
        difficulty = (
            authority_score * 
            length_multiplier * 
            commercial_multiplier * 
            features_multiplier
        )
        
        return min(1.0, max(0.0, difficulty))
    
    def _categorize_difficulty(self, score: float) -> str:
        """Categorize difficulty score into readable levels."""
        if score >= 0.8:
            return 'very_hard'
        elif score >= 0.6:
            return 'hard'
        elif score >= 0.4:
            return 'medium'
        elif score >= 0.2:
            return 'easy'
        else:
            return 'very_easy'
    
    def _extract_domain(self, url: str) -> str:
        """Extract clean domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return 'unknown'
    
    def _generate_keyword_recommendations(
        self,
        keyword: str,
        difficulty_score: float,
        serp_results: Dict[str, Any],
        domain_context: str
    ) -> List[str]:
        """Generate actionable SEO recommendations."""
        
        recommendations = []
        
        # Difficulty-based recommendations
        if difficulty_score >= 0.8:
            recommendations.append("Consider targeting long-tail variations of this keyword")
            recommendations.append("Focus on creating exceptional, comprehensive content to compete")
            recommendations.append("Build domain authority before targeting this competitive keyword")
        elif difficulty_score >= 0.6:
            recommendations.append("Target this keyword with high-quality, in-depth content")
            recommendations.append("Consider building topic clusters around related keywords first")
        elif difficulty_score >= 0.4:
            recommendations.append("Good opportunity - create detailed, well-researched content")
            recommendations.append("Include related keywords and semantic variations")
        else:
            recommendations.append("Excellent keyword opportunity - relatively easy to rank")
            recommendations.append("Create comprehensive content to dominate this keyword space")
        
        # SERP feature recommendations
        features = serp_results.get('features', [])
        if 'featured_snippet' in features:
            recommendations.append("Optimize for featured snippets with clear, concise answers")
        if 'people_also_ask' in features:
            recommendations.append("Address related questions that appear in 'People Also Ask'")
        if 'image_pack' in features:
            recommendations.append("Include relevant, optimized images in your content")
        
        # Content type recommendations based on competitors
        organic_results = serp_results.get('organic_results', [])
        if organic_results:
            top_domains = [result['domain'] for result in organic_results[:5]]
            
            if any('youtube.com' in domain for domain in top_domains):
                recommendations.append("Consider creating video content to compete")
            if any('blog' in domain for domain in top_domains):
                recommendations.append("Long-form blog content performs well for this keyword")
            if any('stackoverflow.com' in domain or 'github.com' in domain for domain in top_domains):
                recommendations.append("Include technical examples and code snippets")
        
        return recommendations[:6]  # Limit to 6 recommendations
    
    def analyze_content_seo_optimization(
        self,
        content: str,
        title: str,
        target_keyword: str,
        secondary_keywords: List[str] = None
    ) -> Dict[str, Any]:
        """Analyze how well content is optimized for target keywords."""
        
        if secondary_keywords is None:
            secondary_keywords = []
        
        logger.info(f"Analyzing SEO optimization for keyword: {target_keyword}")
        
        try:
            # Keyword density analysis
            keyword_density = self._analyze_keyword_density(content, target_keyword, secondary_keywords)
            
            # Title optimization analysis
            title_optimization = self._analyze_title_optimization(title, target_keyword)
            
            # Header optimization analysis
            header_optimization = self._analyze_header_optimization(content, target_keyword, secondary_keywords)
            
            # Content structure analysis for SEO
            content_structure = self._analyze_seo_content_structure(content)
            
            # Internal linking opportunities
            linking_opportunities = self._identify_internal_linking_opportunities(content, secondary_keywords)
            
            # Calculate overall SEO score
            seo_score = self._calculate_seo_optimization_score(
                keyword_density,
                title_optimization,
                header_optimization,
                content_structure
            )
            
            return {
                'seo_optimization_score': seo_score,
                'keyword_density': keyword_density,
                'title_optimization': title_optimization,
                'header_optimization': header_optimization,
                'content_structure': content_structure,
                'internal_linking': linking_opportunities,
                'seo_recommendations': self._generate_seo_optimization_recommendations(
                    keyword_density, title_optimization, header_optimization, content_structure
                )
            }
            
        except Exception as e:
            logger.error(f"SEO optimization analysis failed: {e}")
            return {
                'seo_optimization_score': 0.5,
                'error': str(e)
            }
    
    def _analyze_keyword_density(
        self,
        content: str,
        target_keyword: str,
        secondary_keywords: List[str]
    ) -> Dict[str, Any]:
        """Analyze keyword density and distribution."""
        
        content_lower = content.lower()
        words = content_lower.split()
        total_words = len(words)
        
        # Target keyword analysis
        target_occurrences = content_lower.count(target_keyword.lower())
        target_density = (target_occurrences / total_words * 100) if total_words > 0 else 0
        
        # Secondary keywords analysis
        secondary_analysis = {}
        for keyword in secondary_keywords:
            occurrences = content_lower.count(keyword.lower())
            density = (occurrences / total_words * 100) if total_words > 0 else 0
            secondary_analysis[keyword] = {
                'occurrences': occurrences,
                'density': round(density, 2)
            }
        
        # Keyword distribution analysis (check if keywords appear throughout content)
        content_thirds = [
            content_lower[:len(content_lower)//3],
            content_lower[len(content_lower)//3:2*len(content_lower)//3],
            content_lower[2*len(content_lower)//3:]
        ]
        
        distribution_score = 0
        for third in content_thirds:
            if target_keyword.lower() in third:
                distribution_score += 1
        distribution_score = distribution_score / 3  # Normalize to 0-1
        
        # Optimal density assessment (1-3% is generally good)
        density_score = 1.0
        if target_density < 0.5:
            density_score = 0.6  # Too low
        elif target_density > 4.0:
            density_score = 0.4  # Too high (keyword stuffing)
        elif 1.0 <= target_density <= 3.0:
            density_score = 1.0  # Optimal range
        else:
            density_score = 0.8  # Acceptable range
        
        return {
            'target_keyword': target_keyword,
            'target_occurrences': target_occurrences,
            'target_density': round(target_density, 2),
            'density_score': density_score,
            'distribution_score': round(distribution_score, 2),
            'secondary_keywords': secondary_analysis,
            'total_keywords_found': target_occurrences + sum(data['occurrences'] for data in secondary_analysis.values()),
            'keyword_distribution': 'even' if distribution_score >= 0.6 else 'uneven'
        }
    
    def _analyze_title_optimization(self, title: str, target_keyword: str) -> Dict[str, Any]:
        """Analyze title SEO optimization."""
        
        title_lower = title.lower()
        target_lower = target_keyword.lower()
        
        # Check if target keyword is in title
        keyword_in_title = target_lower in title_lower
        
        # Check keyword position (earlier is better)
        keyword_position = 'not_found'
        if keyword_in_title:
            position = title_lower.find(target_lower)
            if position == 0:
                keyword_position = 'beginning'
            elif position < len(title) / 3:
                keyword_position = 'early'
            elif position < 2 * len(title) / 3:
                keyword_position = 'middle'
            else:
                keyword_position = 'end'
        
        # Title length analysis
        title_length = len(title)
        length_score = 1.0
        if title_length > 60:
            length_score = 0.6  # Too long for SEO
        elif title_length < 30:
            length_score = 0.7  # Too short
        elif 40 <= title_length <= 55:
            length_score = 1.0  # Optimal range
        else:
            length_score = 0.8  # Acceptable
        
        # Emotional/power words
        power_words = [
            'ultimate', 'complete', 'comprehensive', 'essential', 'proven',
            'effective', 'powerful', 'advanced', 'expert', 'professional',
            'innovative', 'revolutionary', 'breakthrough', 'cutting-edge'
        ]
        
        power_words_count = sum(1 for word in power_words if word in title_lower)
        
        # Calculate title optimization score
        title_score = 0.0
        if keyword_in_title:
            title_score += 0.4
            if keyword_position in ['beginning', 'early']:
                title_score += 0.2
        
        title_score += length_score * 0.3
        title_score += min(0.1, power_words_count * 0.05)  # Bonus for power words
        
        return {
            'title_optimization_score': round(min(1.0, title_score), 3),
            'keyword_in_title': keyword_in_title,
            'keyword_position': keyword_position,
            'title_length': title_length,
            'length_score': length_score,
            'power_words_count': power_words_count,
            'title_recommendations': self._generate_title_recommendations(
                keyword_in_title, keyword_position, title_length, target_keyword
            )
        }
    
    def _generate_title_recommendations(
        self,
        keyword_in_title: bool,
        keyword_position: str,
        title_length: int,
        target_keyword: str
    ) -> List[str]:
        """Generate title optimization recommendations."""
        
        recommendations = []
        
        if not keyword_in_title:
            recommendations.append(f"Include the target keyword '{target_keyword}' in the title")
        elif keyword_position in ['middle', 'end']:
            recommendations.append("Move target keyword closer to the beginning of the title")
        
        if title_length > 60:
            recommendations.append("Shorten title to under 60 characters for better search display")
        elif title_length < 30:
            recommendations.append("Expand title to be more descriptive (aim for 40-55 characters)")
        
        if not any(word in target_keyword.lower() for word in ['guide', 'how', 'what', 'best']):
            recommendations.append("Consider adding action words like 'guide', 'how to', or 'best practices'")
        
        return recommendations
    
    def _analyze_header_optimization(
        self,
        content: str,
        target_keyword: str,
        secondary_keywords: List[str]
    ) -> Dict[str, Any]:
        """Analyze header structure and keyword optimization."""
        
        # Extract headers
        headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        
        if not headers:
            return {
                'header_optimization_score': 0.3,
                'header_count': 0,
                'keywords_in_headers': 0,
                'header_recommendations': ['Add section headers to improve content structure']
            }
        
        # Analyze keyword usage in headers
        target_in_headers = 0
        secondary_in_headers = 0
        
        for level, header_text in headers:
            header_lower = header_text.lower()
            if target_keyword.lower() in header_lower:
                target_in_headers += 1
            
            for secondary in secondary_keywords:
                if secondary.lower() in header_lower:
                    secondary_in_headers += 1
        
        # Header hierarchy analysis
        header_levels = [len(h[0]) for h in headers]
        has_h1 = 1 in header_levels
        proper_hierarchy = all(
            level <= level + 1 for level in sorted(set(header_levels))
        )
        
        # Calculate optimization score
        header_score = 0.2  # Base score
        
        if target_in_headers > 0:
            header_score += 0.3
        if secondary_in_headers > 0:
            header_score += 0.2
        if has_h1:
            header_score += 0.15
        if proper_hierarchy:
            header_score += 0.15
        
        return {
            'header_optimization_score': round(min(1.0, header_score), 3),
            'header_count': len(headers),
            'target_keyword_in_headers': target_in_headers,
            'secondary_keywords_in_headers': secondary_in_headers,
            'has_h1': has_h1,
            'proper_hierarchy': proper_hierarchy,
            'header_recommendations': self._generate_header_recommendations(
                target_in_headers, has_h1, len(headers)
            )
        }
    
    def _generate_header_recommendations(
        self,
        target_in_headers: int,
        has_h1: bool,
        header_count: int
    ) -> List[str]:
        """Generate header optimization recommendations."""
        
        recommendations = []
        
        if target_in_headers == 0:
            recommendations.append("Include target keyword in at least one header")
        elif target_in_headers > 3:
            recommendations.append("Avoid keyword stuffing in headers - use variations instead")
        
        if not has_h1:
            recommendations.append("Add an H1 header with the target keyword")
        
        if header_count < 2:
            recommendations.append("Add more section headers to improve content structure")
        elif header_count > 10:
            recommendations.append("Consider consolidating headers - too many may fragment content")
        
        return recommendations
    
    def _analyze_seo_content_structure(self, content: str) -> Dict[str, Any]:
        """Analyze content structure from SEO perspective."""
        
        # Introduction and conclusion analysis
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        has_introduction = len(paragraphs) > 0 and len(paragraphs[0].split()) > 50
        has_conclusion = len(paragraphs) > 1 and len(paragraphs[-1].split()) > 30
        
        # Content depth analysis
        content_sections = len(re.findall(r'^#{2,6}\s+', content, re.MULTILINE))
        
        # List usage (good for featured snippets)
        has_lists = bool(re.search(r'^\s*[-*+]\s+|\s*\d+\.\s+', content, re.MULTILINE))
        
        # FAQ-style content
        faq_indicators = len(re.findall(r'\b(what|how|why|when|where|who)\b.*\?', content, re.IGNORECASE))
        
        # Calculate structure score
        structure_score = 0.4  # Base score
        
        if has_introduction:
            structure_score += 0.15
        if has_conclusion:
            structure_score += 0.15
        if content_sections >= 3:
            structure_score += 0.15
        if has_lists:
            structure_score += 0.1
        if faq_indicators > 0:
            structure_score += 0.05
        
        return {
            'content_structure_score': round(min(1.0, structure_score), 3),
            'has_introduction': has_introduction,
            'has_conclusion': has_conclusion,
            'content_sections': content_sections,
            'has_lists': has_lists,
            'faq_indicators': faq_indicators,
            'structure_recommendations': self._generate_structure_recommendations(
                has_introduction, has_conclusion, content_sections, has_lists
            )
        }
    
    def _generate_structure_recommendations(
        self,
        has_intro: bool,
        has_conclusion: bool,
        sections: int,
        has_lists: bool
    ) -> List[str]:
        """Generate content structure recommendations."""
        
        recommendations = []
        
        if not has_intro:
            recommendations.append("Add a clear introduction that includes target keywords")
        if not has_conclusion:
            recommendations.append("Add a conclusion that summarizes key points and includes a call-to-action")
        if sections < 3:
            recommendations.append("Add more content sections for better structure and keyword distribution")
        if not has_lists:
            recommendations.append("Include bullet points or numbered lists for featured snippet optimization")
        
        return recommendations
    
    def _identify_internal_linking_opportunities(
        self,
        content: str,
        secondary_keywords: List[str]
    ) -> Dict[str, Any]:
        """Identify opportunities for internal linking."""
        
        # Find phrases that could link to other content
        linkable_phrases = []
        
        # Look for secondary keywords that could be internal links
        content_lower = content.lower()
        for keyword in secondary_keywords:
            if keyword.lower() in content_lower:
                # Find context around the keyword
                keyword_positions = []
                start = 0
                while True:
                    pos = content_lower.find(keyword.lower(), start)
                    if pos == -1:
                        break
                    keyword_positions.append(pos)
                    start = pos + 1
                
                for pos in keyword_positions[:3]:  # Limit to 3 occurrences per keyword
                    # Extract sentence containing the keyword
                    sentence_start = max(0, content.rfind('.', 0, pos) + 1)
                    sentence_end = content.find('.', pos)
                    if sentence_end == -1:
                        sentence_end = len(content)
                    
                    context = content[sentence_start:sentence_end].strip()
                    if len(context) > 20:
                        linkable_phrases.append({
                            'keyword': keyword,
                            'context': context[:100] + "..." if len(context) > 100 else context,
                            'suggested_anchor': keyword
                        })
        
        # Look for common linkable terms
        common_linkable_terms = [
            'best practices', 'case study', 'implementation guide',
            'getting started', 'documentation', 'tutorial', 'examples'
        ]
        
        for term in common_linkable_terms:
            if term in content_lower:
                linkable_phrases.append({
                    'keyword': term,
                    'context': f"Link to related {term} content",
                    'suggested_anchor': term
                })
        
        return {
            'linking_opportunities_count': len(linkable_phrases),
            'suggested_links': linkable_phrases[:8],  # Limit to 8 suggestions
            'internal_linking_score': min(1.0, len(linkable_phrases) / 5),  # Up to 5 links is optimal
            'recommendations': [
                f"Consider adding internal links for: {phrase['keyword']}"
                for phrase in linkable_phrases[:3]
            ]
        }
    
    def _calculate_seo_optimization_score(
        self,
        keyword_density: Dict[str, Any],
        title_optimization: Dict[str, Any],
        header_optimization: Dict[str, Any],
        content_structure: Dict[str, Any]
    ) -> float:
        """Calculate overall SEO optimization score."""
        
        # Weighted scoring
        weights = {
            'keyword_density': 0.3,
            'title_optimization': 0.3,
            'header_optimization': 0.2,
            'content_structure': 0.2
        }
        
        scores = {
            'keyword_density': keyword_density.get('density_score', 0.5),
            'title_optimization': title_optimization.get('title_optimization_score', 0.5),
            'header_optimization': header_optimization.get('header_optimization_score', 0.5),
            'content_structure': content_structure.get('content_structure_score', 0.5)
        }
        
        overall_score = sum(scores[key] * weights[key] for key in weights.keys())
        
        return round(min(1.0, max(0.0, overall_score)), 3)
    
    def _generate_seo_optimization_recommendations(
        self,
        keyword_density: Dict[str, Any],
        title_optimization: Dict[str, Any],
        header_optimization: Dict[str, Any],
        content_structure: Dict[str, Any]
    ) -> List[str]:
        """Generate comprehensive SEO optimization recommendations."""
        
        recommendations = []
        
        # Combine all specific recommendations
        recommendations.extend(title_optimization.get('title_recommendations', []))
        recommendations.extend(header_optimization.get('header_recommendations', []))
        recommendations.extend(content_structure.get('structure_recommendations', []))
        
        # Keyword density recommendations
        density_score = keyword_density.get('density_score', 1.0)
        if density_score < 0.7:
            target_density = keyword_density.get('target_density', 0)
            if target_density < 0.5:
                recommendations.append("Increase target keyword usage naturally throughout content")
            elif target_density > 4.0:
                recommendations.append("Reduce keyword density to avoid keyword stuffing penalties")
        
        # Distribution recommendations
        distribution_score = keyword_density.get('distribution_score', 1.0)
        if distribution_score < 0.6:
            recommendations.append("Distribute target keyword more evenly throughout the content")
        
        # Remove duplicates and limit
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:8]