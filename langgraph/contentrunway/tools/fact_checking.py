"""Fact-checking tools for verifying claims against sources and external verification."""

import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import asyncio
import aiohttp
from urllib.parse import quote
import hashlib

logger = logging.getLogger(__name__)


class FactCheckTool:
    """Tool for fact-checking content claims against sources and external verification."""
    
    def __init__(self):
        """Initialize fact-checking tool."""
        
        # External fact-checking sources
        self.fact_check_sources = {
            'snopes': {
                'base_url': 'https://www.snopes.com',
                'search_url': 'https://www.snopes.com/search/?q={}',
                'credibility_score': 0.9
            },
            'factcheck_org': {
                'base_url': 'https://www.factcheck.org',
                'search_url': 'https://www.factcheck.org/search/?q={}',
                'credibility_score': 0.95
            },
            'politifact': {
                'base_url': 'https://www.politifact.com',
                'search_url': 'https://www.politifact.com/search/?q={}',
                'credibility_score': 0.85
            }
        }
        
        # Government and institutional sources for verification
        self.authoritative_sources = {
            'academic': [
                'arxiv.org', 'nature.com', 'science.org', 'ieee.org', 'acm.org',
                'springer.com', 'elsevier.com', 'wiley.com', 'cambridge.org', 'oxford.ac.uk'
            ],
            'government': [
                'gov', 'edu', 'nasa.gov', 'nih.gov', 'cdc.gov', 'fda.gov',
                'sec.gov', 'ftc.gov', 'census.gov'
            ],
            'tech_official': [
                'openai.com', 'google.ai', 'microsoft.com', 'ibm.com',
                'research.google.com', 'research.microsoft.com', 'blog.openai.com'
            ],
            'financial_regulatory': [
                'sec.gov', 'finra.org', 'federalreserve.gov', 'treasury.gov',
                'cftc.gov', 'fdic.gov', 'occ.gov'
            ]
        }
        
        # Common claim patterns to identify factual statements
        self.claim_patterns = [
            r'according to [^,]+,\s*(.+?)(?:\.|$)',  # "According to X, claim"
            r'research shows (?:that\s+)?(.+?)(?:\.|$)',  # "Research shows claim"
            r'studies indicate (?:that\s+)?(.+?)(?:\.|$)',  # "Studies indicate claim"
            r'data suggests (?:that\s+)?(.+?)(?:\.|$)',  # "Data suggests claim"
            r'(\d+(?:\.\d+)?%?) of (.+?)(?:\.|$)',  # "X% of something"
            r'(\d+(?:,\d{3})*) (.+?)(?:\.|$)',  # "Number something"
            r'in (\d{4}), (.+?)(?:\.|$)',  # "In year, claim"
            r'as of (\d{4}|\w+ \d{4}), (.+?)(?:\.|$)',  # "As of date, claim"
            r'(.+?) (?:increased|decreased|rose|fell) by (\d+(?:\.\d+)?%?)(?:\.|$)'  # Change claims
        ]
        
        # Initialize HTTP session
        self.session = None
        self._init_session()
    
    def _init_session(self):
        """Initialize HTTP session for web requests."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ContentRunway/1.0 (Fact-checking Bot; +https://contentrunway.ai)'
        })
    
    async def fact_check_content(
        self,
        content: str,
        sources: List[Dict[str, Any]] = None,
        check_external: bool = True,
        domain_context: str = "general"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive fact-checking of content.
        
        Args:
            content: Content to fact-check
            sources: Source documents for verification
            check_external: Whether to check against external fact-check sources
            domain_context: Domain context for specialized fact-checking
            
        Returns:
            Dictionary with fact-checking results
        """
        logger.info("Starting comprehensive fact-checking analysis")
        
        try:
            # Step 1: Extract factual claims from content
            claims_extraction = self._extract_factual_claims(content)
            
            # Step 2: Verify claims against provided sources
            source_verification = await self._verify_against_sources(
                claims_extraction['claims'],
                sources or []
            )
            
            # Step 3: External fact-checking (if enabled)
            external_verification = {}
            if check_external and claims_extraction['claims']:
                external_verification = await self._verify_against_external_sources(
                    claims_extraction['claims'][:5],  # Limit to top 5 claims for external checking
                    domain_context
                )
            
            # Step 4: Check for citation coverage
            citation_coverage = self._analyze_citation_coverage(content, sources or [])
            
            # Step 5: Validate statistical claims
            statistical_validation = self._validate_statistical_claims(content)
            
            # Step 6: Check temporal accuracy (dates, versions, etc.)
            temporal_accuracy = self._check_temporal_accuracy(content)
            
            # Step 7: Calculate overall fact-check score
            fact_check_score = self._calculate_fact_check_score(
                source_verification,
                external_verification,
                citation_coverage,
                statistical_validation,
                temporal_accuracy
            )
            
            return {
                'fact_check_score': fact_check_score,
                'claims_extracted': claims_extraction,
                'source_verification': source_verification,
                'external_verification': external_verification,
                'citation_coverage': citation_coverage,
                'statistical_validation': statistical_validation,
                'temporal_accuracy': temporal_accuracy,
                'recommendations': self._generate_fact_check_recommendations(
                    fact_check_score, source_verification, external_verification,
                    citation_coverage, statistical_validation
                ),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fact-checking failed: {e}")
            return {
                'fact_check_score': 0.5,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _extract_factual_claims(self, content: str) -> Dict[str, Any]:
        """Extract factual claims from content that need verification."""
        
        claims = []
        
        # Extract claims using patterns
        for pattern in self.claim_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                claim_text = match.group().strip()
                
                # Extract the actual claim part (varies by pattern)
                if 'according to' in match.group().lower():
                    claim = match.group(1).strip()
                    source_mention = match.group().split(',')[0].replace('according to', '').strip()
                elif any(phrase in match.group().lower() for phrase in ['research shows', 'studies indicate', 'data suggests']):
                    claim = match.group(1).strip()
                    source_mention = "research/studies"
                else:
                    claim = claim_text
                    source_mention = None
                
                if len(claim) > 10:  # Filter out very short claims
                    claims.append({
                        'claim_id': f"claim_{len(claims)}",
                        'claim_text': claim,
                        'full_context': claim_text,
                        'source_mentioned': source_mention,
                        'position_in_content': match.start(),
                        'pattern_matched': pattern,
                        'claim_type': self._categorize_claim_type(claim)
                    })
        
        # Extract numerical claims separately
        numerical_claims = self._extract_numerical_claims(content)
        claims.extend(numerical_claims)
        
        # Extract temporal claims (dates, years, etc.)
        temporal_claims = self._extract_temporal_claims(content)
        claims.extend(temporal_claims)
        
        return {
            'claims': claims,
            'total_claims_found': len(claims),
            'claim_types': list(set(claim['claim_type'] for claim in claims)),
            'claims_with_sources': len([c for c in claims if c['source_mentioned']]),
            'numerical_claims': len([c for c in claims if c['claim_type'] == 'numerical']),
            'temporal_claims': len([c for c in claims if c['claim_type'] == 'temporal'])
        }
    
    def _categorize_claim_type(self, claim: str) -> str:
        """Categorize the type of factual claim."""
        
        claim_lower = claim.lower()
        
        # Numerical claims
        if re.search(r'\d+(?:\.\d+)?%?', claim):
            return 'numerical'
        
        # Temporal claims
        if re.search(r'\b(?:19|20)\d{2}\b|january|february|march|april|may|june|july|august|september|october|november|december', claim_lower):
            return 'temporal'
        
        # Technical claims
        if any(term in claim_lower for term in ['algorithm', 'model', 'system', 'technology', 'software', 'hardware']):
            return 'technical'
        
        # Regulatory/legal claims
        if any(term in claim_lower for term in ['law', 'regulation', 'compliance', 'legal', 'requirement']):
            return 'regulatory'
        
        # Business/financial claims
        if any(term in claim_lower for term in ['revenue', 'profit', 'market', 'investment', 'cost', 'price']):
            return 'financial'
        
        # Default
        return 'general'
    
    def _extract_numerical_claims(self, content: str) -> List[Dict[str, Any]]:
        """Extract numerical claims that need verification."""
        
        numerical_patterns = [
            r'(\d+(?:\.\d+)?%?) (?:of|increase|decrease|growth|decline) (.+?)(?:\.|$)',
            r'(\d+(?:,\d{3})*) (?:people|users|customers|companies|businesses) (.+?)(?:\.|$)',
            r'(?:costs?|prices?|revenue|profit) (?:of|at|reached) \$?(\d+(?:,\d{3})*(?:\.\d+)?[kmb]?) (.+?)(?:\.|$)',
            r'(\d+(?:\.\d+)?) (?:times|fold) (?:increase|improvement|growth) (.+?)(?:\.|$)'
        ]
        
        claims = []
        
        for pattern in numerical_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                claim_text = match.group().strip()
                
                claims.append({
                    'claim_id': f"numerical_claim_{len(claims)}",
                    'claim_text': claim_text,
                    'full_context': claim_text,
                    'source_mentioned': None,
                    'position_in_content': match.start(),
                    'pattern_matched': pattern,
                    'claim_type': 'numerical',
                    'numerical_value': match.group(1),
                    'numerical_context': match.group(2) if len(match.groups()) > 1 else ""
                })
        
        return claims
    
    def _extract_temporal_claims(self, content: str) -> List[Dict[str, Any]]:
        """Extract temporal claims (dates, years, etc.) that need verification."""
        
        temporal_patterns = [
            r'in (\d{4}), (.+?)(?:\.|$)',
            r'(?:since|from|until) (\d{4}), (.+?)(?:\.|$)',
            r'(?:as of|by) (\w+ \d{4}|\d{4}), (.+?)(?:\.|$)',
            r'(\w+ \d{1,2}, \d{4}): (.+?)(?:\.|$)'
        ]
        
        claims = []
        
        for pattern in temporal_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                claim_text = match.group().strip()
                
                claims.append({
                    'claim_id': f"temporal_claim_{len(claims)}",
                    'claim_text': claim_text,
                    'full_context': claim_text,
                    'source_mentioned': None,
                    'position_in_content': match.start(),
                    'pattern_matched': pattern,
                    'claim_type': 'temporal',
                    'temporal_reference': match.group(1),
                    'temporal_context': match.group(2) if len(match.groups()) > 1 else ""
                })
        
        return claims
    
    async def _verify_against_sources(
        self,
        claims: List[Dict[str, Any]],
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify claims against provided source documents."""
        
        if not sources:
            return {
                'verification_score': 0.5,
                'claims_verified': [],
                'sources_checked': 0,
                'message': 'No sources provided for verification'
            }
        
        logger.info(f"Verifying {len(claims)} claims against {len(sources)} sources")
        
        try:
            verified_claims = []
            
            for claim in claims:
                claim_verification = await self._verify_single_claim_against_sources(claim, sources)
                verified_claims.append(claim_verification)
            
            # Calculate overall verification score
            supported_claims = sum(1 for vc in verified_claims if vc['verification_status'] == 'supported')
            verification_score = supported_claims / len(claims) if claims else 1.0
            
            return {
                'verification_score': round(verification_score, 3),
                'claims_verified': verified_claims,
                'total_claims': len(claims),
                'supported_claims': supported_claims,
                'unsupported_claims': len(claims) - supported_claims,
                'sources_checked': len(sources),
                'verification_summary': self._create_verification_summary(verified_claims)
            }
            
        except Exception as e:
            logger.error(f"Source verification failed: {e}")
            return {
                'verification_score': 0.5,
                'error': str(e),
                'claims_verified': [],
                'sources_checked': len(sources)
            }
    
    async def _verify_single_claim_against_sources(
        self,
        claim: Dict[str, Any],
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify a single claim against source documents."""
        
        claim_text = claim['claim_text'].lower()
        supporting_sources = []
        contradicting_sources = []
        
        # Check each source for support/contradiction
        for source in sources:
            source_text = ""
            
            # Extract text from source
            if 'content' in source and source['content']:
                source_text = source['content'].lower()
            elif 'summary' in source and source['summary']:
                source_text = source['summary'].lower()
            elif 'title' in source:
                source_text = source['title'].lower()
            
            if not source_text:
                continue
            
            # Check for support (simple keyword/phrase matching)
            support_score = self._calculate_claim_support_score(claim_text, source_text)
            
            if support_score >= 0.7:
                supporting_sources.append({
                    'source': source,
                    'support_score': support_score,
                    'support_type': 'strong'
                })
            elif support_score >= 0.4:
                supporting_sources.append({
                    'source': source,
                    'support_score': support_score,
                    'support_type': 'moderate'
                })
            
            # Check for contradiction indicators
            contradiction_score = self._check_for_contradiction(claim_text, source_text)
            if contradiction_score >= 0.6:
                contradicting_sources.append({
                    'source': source,
                    'contradiction_score': contradiction_score
                })
        
        # Determine verification status
        if supporting_sources:
            max_support = max(s['support_score'] for s in supporting_sources)
            if max_support >= 0.8:
                verification_status = 'strongly_supported'
            elif max_support >= 0.6:
                verification_status = 'supported'
            else:
                verification_status = 'weakly_supported'
        elif contradicting_sources:
            verification_status = 'contradicted'
        else:
            verification_status = 'unverified'
        
        return {
            'claim_id': claim['claim_id'],
            'claim_text': claim['claim_text'],
            'verification_status': verification_status,
            'supporting_sources': supporting_sources[:3],  # Top 3 supporting sources
            'contradicting_sources': contradicting_sources[:2],  # Top 2 contradicting sources
            'confidence_score': self._calculate_verification_confidence(
                supporting_sources, contradicting_sources
            )
        }
    
    def _calculate_claim_support_score(self, claim_text: str, source_text: str) -> float:
        """Calculate how well a source supports a claim."""
        
        # Tokenize claim and source
        claim_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', claim_text.lower()))
        source_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', source_text.lower()))
        
        # Calculate word overlap
        common_words = claim_words.intersection(source_words)
        word_overlap_score = len(common_words) / len(claim_words) if claim_words else 0
        
        # Check for key phrases in source
        claim_phrases = self._extract_key_phrases(claim_text)
        phrase_support_score = 0
        
        for phrase in claim_phrases:
            if phrase in source_text:
                phrase_support_score += 1
        
        phrase_support_score = phrase_support_score / len(claim_phrases) if claim_phrases else 0
        
        # Check for numerical consistency (if claim contains numbers)
        numerical_consistency = self._check_numerical_consistency(claim_text, source_text)
        
        # Combine scores
        support_score = (
            word_overlap_score * 0.3 +
            phrase_support_score * 0.4 +
            numerical_consistency * 0.3
        )
        
        return round(support_score, 3)
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text for matching."""
        
        # Simple noun phrase extraction
        phrases = []
        
        # Multi-word technical terms
        multi_word_patterns = [
            r'\b(?:machine learning|artificial intelligence|deep learning)\b',
            r'\b(?:data science|computer vision|natural language processing)\b',
            r'\b(?:cloud computing|edge computing|quantum computing)\b',
            r'\b(?:cyber security|information security|data protection)\b'
        ]
        
        for pattern in multi_word_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            phrases.extend([match.lower() for match in matches])
        
        # Important single terms (domain-specific)
        important_terms = re.findall(r'\b(?:AI|ML|API|SDK|SaaS|IoT|5G|blockchain|kubernetes|docker)\b', text, re.IGNORECASE)
        phrases.extend([term.lower() for term in important_terms])
        
        return list(set(phrases))
    
    def _check_numerical_consistency(self, claim_text: str, source_text: str) -> float:
        """Check if numerical values in claim match those in source."""
        
        # Extract numbers from both texts
        claim_numbers = re.findall(r'\d+(?:\.\d+)?', claim_text)
        source_numbers = re.findall(r'\d+(?:\.\d+)?', source_text)
        
        if not claim_numbers:
            return 0.5  # No numbers to verify
        
        if not source_numbers:
            return 0.3  # Claim has numbers but source doesn't
        
        # Check for matching numbers
        claim_nums = set(claim_numbers)
        source_nums = set(source_numbers)
        
        matching_numbers = claim_nums.intersection(source_nums)
        consistency_score = len(matching_numbers) / len(claim_nums)
        
        return consistency_score
    
    def _check_for_contradiction(self, claim_text: str, source_text: str) -> float:
        """Check if source contradicts the claim."""
        
        contradiction_indicators = [
            ('increase', 'decrease'), ('rise', 'fall'), ('improve', 'worsen'),
            ('success', 'failure'), ('effective', 'ineffective'), ('secure', 'insecure'),
            ('accurate', 'inaccurate'), ('true', 'false'), ('correct', 'incorrect')
        ]
        
        contradiction_score = 0
        total_checks = 0
        
        for positive, negative in contradiction_indicators:
            if positive in claim_text and negative in source_text:
                contradiction_score += 1
                total_checks += 1
            elif negative in claim_text and positive in source_text:
                contradiction_score += 1
                total_checks += 1
            elif positive in claim_text or negative in claim_text:
                total_checks += 1
        
        return contradiction_score / total_checks if total_checks > 0 else 0.0
    
    async def _verify_against_external_sources(
        self,
        claims: List[Dict[str, Any]],
        domain_context: str = "general"
    ) -> Dict[str, Any]:
        """Verify claims against external fact-checking sources."""
        
        logger.info(f"Performing external verification for {len(claims)} claims")
        
        external_results = []
        
        try:
            # Use async session for concurrent requests
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                # Verify each claim
                tasks = []
                for claim in claims[:3]:  # Limit to 3 claims for external verification
                    task = self._verify_claim_externally(session, claim, domain_context)
                    tasks.append(task)
                
                # Wait for all verifications
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        external_results.append({
                            'claim_id': claims[i]['claim_id'],
                            'verification_status': 'error',
                            'error': str(result)
                        })
                    else:
                        external_results.append(result)
        
        except Exception as e:
            logger.error(f"External verification failed: {e}")
            return {
                'external_verification_score': 0.5,
                'error': str(e),
                'verifications': []
            }
        
        # Calculate overall external verification score
        successful_verifications = [r for r in external_results if r.get('verification_status') not in ['error', 'not_found']]
        supported_claims = sum(1 for r in successful_verifications if 'supported' in r.get('verification_status', ''))
        
        verification_score = supported_claims / len(successful_verifications) if successful_verifications else 0.5
        
        return {
            'external_verification_score': round(verification_score, 3),
            'verifications': external_results,
            'claims_checked_externally': len(claims),
            'successful_checks': len(successful_verifications),
            'supported_by_external': supported_claims
        }
    
    async def _verify_claim_externally(
        self,
        session: aiohttp.ClientSession,
        claim: Dict[str, Any],
        domain_context: str
    ) -> Dict[str, Any]:
        """Verify a single claim against external fact-checking sources."""
        
        claim_text = claim['claim_text']
        
        # Create search query from claim
        search_terms = self._extract_search_terms(claim_text)
        search_query = ' '.join(search_terms[:5])  # Limit to 5 terms
        
        # Try different fact-checking sources
        for source_name, source_config in self.fact_check_sources.items():
            try:
                # Perform search
                search_url = source_config['search_url'].format(quote(search_query))
                
                async with session.get(search_url) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        verification_result = self._parse_fact_check_results(
                            html_content,
                            claim_text,
                            source_name,
                            source_config['credibility_score']
                        )
                        
                        if verification_result['found_relevant']:
                            return {
                                'claim_id': claim['claim_id'],
                                'verification_status': verification_result['status'],
                                'external_source': source_name,
                                'credibility_score': source_config['credibility_score'],
                                'verification_details': verification_result,
                                'search_query': search_query
                            }
                
                # Rate limiting between requests
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.warning(f"External verification failed for {source_name}: {e}")
                continue
        
        # No verification found
        return {
            'claim_id': claim['claim_id'],
            'verification_status': 'not_found',
            'external_source': None,
            'search_query': search_query,
            'message': 'No external verification found for this claim'
        }
    
    def _extract_search_terms(self, claim_text: str) -> List[str]:
        """Extract key search terms from a claim."""
        
        # Remove common words and extract meaningful terms
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'that', 'this', 'which', 'when', 'where', 'how', 'why',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'
        }
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', claim_text.lower())
        meaningful_words = [word for word in words if word not in stop_words]
        
        # Prioritize longer, more specific terms
        meaningful_words.sort(key=len, reverse=True)
        
        return meaningful_words[:8]  # Top 8 terms
    
    def _parse_fact_check_results(
        self,
        html_content: str,
        claim_text: str,
        source_name: str,
        credibility_score: float
    ) -> Dict[str, Any]:
        """Parse fact-checking website results."""
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Look for fact-check results
        found_relevant = False
        status = 'not_found'
        confidence = 0.0
        
        # Extract search results or fact-check articles
        if source_name == 'snopes':
            results = soup.select('.search-result, .fact-check')
        elif source_name == 'factcheck_org':
            results = soup.select('.post, .entry')
        elif source_name == 'politifact':
            results = soup.select('.statement, .ruling')
        else:
            results = soup.select('article, .result, .post')
        
        # Analyze results for relevance
        for result in results[:5]:  # Check top 5 results
            result_text = result.get_text(strip=True).lower()
            
            # Check if result is relevant to claim
            claim_words = set(claim_text.lower().split())
            result_words = set(result_text.split())
            
            word_overlap = len(claim_words.intersection(result_words)) / len(claim_words) if claim_words else 0
            
            if word_overlap >= 0.3:  # Reasonable overlap threshold
                found_relevant = True
                
                # Look for fact-check ratings
                if any(word in result_text for word in ['true', 'correct', 'accurate', 'verified']):
                    status = 'supported'
                    confidence = credibility_score * 0.8
                elif any(word in result_text for word in ['false', 'incorrect', 'inaccurate', 'debunked']):
                    status = 'contradicted'
                    confidence = credibility_score * 0.9
                elif any(word in result_text for word in ['mixed', 'partially', 'mostly']):
                    status = 'partially_supported'
                    confidence = credibility_score * 0.6
                else:
                    status = 'found_but_unclear'
                    confidence = credibility_score * 0.4
                
                break
        
        return {
            'found_relevant': found_relevant,
            'status': status,
            'confidence': round(confidence, 3),
            'word_overlap_score': round(word_overlap, 3) if found_relevant else 0.0,
            'source_credibility': credibility_score
        }
    
    def _analyze_citation_coverage(self, content: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how well claims are covered by citations."""
        
        # Extract citation markers
        citation_markers = re.findall(r'\[(\d+)\]|\[Citation (\d+)\]', content)
        citation_numbers = []
        
        for marker in citation_markers:
            if marker[0]:
                try:
                    citation_numbers.append(int(marker[0]))
                except ValueError:
                    pass
            elif marker[1]:
                try:
                    citation_numbers.append(int(marker[1]))
                except ValueError:
                    pass
        
        unique_citations = len(set(citation_numbers))
        total_sources = len(sources)
        
        # Calculate citation coverage
        coverage_ratio = unique_citations / total_sources if total_sources > 0 else 0
        
        # Check citation density (citations per paragraph)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        citations_per_paragraph = len(citation_numbers) / len(paragraphs) if paragraphs else 0
        
        # Identify uncited claims
        claims_sections = re.split(r'\[(?:\d+|Citation \d+)\]', content)
        uncited_claims = []
        
        for section in claims_sections:
            # Look for factual statements without citations
            if any(pattern in section.lower() for pattern in ['research shows', 'studies indicate', 'data suggests']):
                if len(section.strip()) > 50:  # Substantial content
                    uncited_claims.append(section.strip()[:100] + "...")
        
        return {
            'citation_coverage_score': round(min(1.0, coverage_ratio * 1.2), 3),  # Slight boost
            'citations_found': len(citation_numbers),
            'unique_citations': unique_citations,
            'sources_available': total_sources,
            'coverage_ratio': round(coverage_ratio, 3),
            'citations_per_paragraph': round(citations_per_paragraph, 2),
            'uncited_claims': uncited_claims[:3],  # Top 3 uncited claims
            'citation_density': 'high' if citations_per_paragraph > 0.5 else 'medium' if citations_per_paragraph > 0.2 else 'low'
        }
    
    def _validate_statistical_claims(self, content: str) -> Dict[str, Any]:
        """Validate statistical and numerical claims in content."""
        
        # Extract statistical patterns
        statistical_patterns = [
            r'(\d+(?:\.\d+)?%)',  # Percentages
            r'(\d+(?:,\d{3})*)\s+(?:people|users|customers|companies)',  # Population figures
            r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:million|billion|trillion)?',  # Financial figures
            r'(\d+(?:\.\d+)?)\s*(?:times|fold)',  # Multipliers
            r'(\d+(?:\.\d+)?)\s*(?:seconds|minutes|hours|days|weeks|months|years)',  # Time periods
        ]
        
        statistical_claims = []
        
        for pattern in statistical_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Get surrounding context
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end]
                
                statistical_claims.append({
                    'value': match.group(1),
                    'context': context,
                    'position': match.start(),
                    'type': self._categorize_statistical_claim(match.group())
                })
        
        # Validate reasonableness of statistical claims
        validation_results = []
        
        for claim in statistical_claims:
            validation = self._validate_statistical_reasonableness(claim)
            validation_results.append(validation)
        
        # Calculate overall statistical validation score
        reasonable_claims = sum(1 for v in validation_results if v['is_reasonable'])
        validation_score = reasonable_claims / len(validation_results) if validation_results else 1.0
        
        return {
            'statistical_validation_score': round(validation_score, 3),
            'statistical_claims_found': len(statistical_claims),
            'reasonable_claims': reasonable_claims,
            'questionable_claims': len(validation_results) - reasonable_claims,
            'validation_details': validation_results[:5],  # Top 5 validations
            'statistical_claim_types': list(set(claim['type'] for claim in statistical_claims))
        }
    
    def _categorize_statistical_claim(self, claim_text: str) -> str:
        """Categorize the type of statistical claim."""
        
        claim_lower = claim_text.lower()
        
        if '%' in claim_text:
            return 'percentage'
        elif '$' in claim_text or 'million' in claim_lower or 'billion' in claim_lower:
            return 'financial'
        elif any(word in claim_lower for word in ['people', 'users', 'customers', 'companies']):
            return 'population'
        elif any(word in claim_lower for word in ['times', 'fold']):
            return 'multiplier'
        elif any(word in claim_lower for word in ['seconds', 'minutes', 'hours', 'days', 'weeks', 'months', 'years']):
            return 'temporal'
        else:
            return 'numerical'
    
    def _validate_statistical_reasonableness(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if a statistical claim seems reasonable."""
        
        value_str = claim['value'].replace(',', '')
        claim_type = claim['type']
        context = claim['context'].lower()
        
        try:
            # Convert to number
            if '%' in value_str:
                value = float(value_str.replace('%', ''))
                
                # Percentage validation
                if value < 0 or value > 100:
                    return {
                        'is_reasonable': False,
                        'issue': f"Percentage {value}% is outside valid range (0-100%)",
                        'confidence': 0.9
                    }
                elif value > 99.9 and 'almost' not in context and 'nearly' not in context:
                    return {
                        'is_reasonable': False,
                        'issue': f"Percentage {value}% seems suspiciously high",
                        'confidence': 0.7
                    }
                
            else:
                value = float(value_str)
                
                # Numerical validation based on type
                if claim_type == 'financial' and value > 1000000000000:  # > 1 trillion
                    return {
                        'is_reasonable': False,
                        'issue': f"Financial figure ${value:,.0f} seems extremely large",
                        'confidence': 0.8
                    }
                elif claim_type == 'population' and value > 8000000000:  # > 8 billion (world population)
                    return {
                        'is_reasonable': False,
                        'issue': f"Population figure {value:,.0f} exceeds world population",
                        'confidence': 0.95
                    }
            
            # If we get here, the claim seems reasonable
            return {
                'is_reasonable': True,
                'value': value,
                'claim_type': claim_type,
                'confidence': 0.8
            }
            
        except ValueError:
            return {
                'is_reasonable': False,
                'issue': f"Could not parse numerical value: {value_str}",
                'confidence': 0.6
            }
    
    def _check_temporal_accuracy(self, content: str) -> Dict[str, Any]:
        """Check temporal accuracy of dates and time references."""
        
        current_year = datetime.now().year
        
        # Extract year references
        year_patterns = [
            r'\b(19|20)(\d{2})\b',  # 4-digit years
            r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',  # MM/DD/YYYY dates
            r'\b(\w+) (\d{1,2}), (\d{4})\b'  # Month DD, YYYY
        ]
        
        temporal_issues = []
        years_found = []
        
        # Check for future dates or impossible dates
        for pattern in year_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                if len(match.groups()) == 2:  # Simple year format
                    year = int(match.group(1) + match.group(2))
                elif len(match.groups()) == 3 and match.group(3).isdigit():  # Date format
                    year = int(match.group(3))
                else:
                    continue
                
                years_found.append(year)
                
                # Check for temporal issues
                if year > current_year:
                    temporal_issues.append({
                        'issue': f"Future date reference: {year}",
                        'context': match.group(),
                        'issue_type': 'future_date'
                    })
                elif year < 1900:
                    temporal_issues.append({
                        'issue': f"Very old date reference: {year}",
                        'context': match.group(),
                        'issue_type': 'very_old_date'
                    })
        
        # Check for version references that might be outdated
        version_patterns = [
            r'(?:version|v\.?)\s*(\d+(?:\.\d+)*)',
            r'(\w+)\s+(\d+(?:\.\d+)*)',  # "Python 3.9", etc.
        ]
        
        version_issues = []
        for pattern in version_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                version_text = match.group()
                
                # Check for known outdated versions
                if any(outdated in version_text.lower() for outdated in ['python 2', 'node.js 12', 'tensorflow 1']):
                    version_issues.append({
                        'issue': f"Potentially outdated version reference: {version_text}",
                        'context': version_text,
                        'issue_type': 'outdated_version'
                    })
        
        # Calculate temporal accuracy score
        total_temporal_refs = len(years_found) + len(version_issues)
        issues_count = len(temporal_issues) + len(version_issues)
        
        temporal_accuracy_score = max(0.0, 1.0 - (issues_count / max(1, total_temporal_refs)))
        
        return {
            'temporal_accuracy_score': round(temporal_accuracy_score, 3),
            'years_referenced': sorted(list(set(years_found))),
            'temporal_issues': temporal_issues,
            'version_issues': version_issues,
            'total_temporal_references': total_temporal_refs,
            'issues_found': issues_count,
            'date_range': f"{min(years_found)}-{max(years_found)}" if years_found else "No dates found"
        }
    
    def _calculate_verification_confidence(
        self,
        supporting_sources: List[Dict[str, Any]],
        contradicting_sources: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for claim verification."""
        
        if not supporting_sources and not contradicting_sources:
            return 0.1  # No evidence found
        
        # Calculate support strength
        if supporting_sources:
            max_support = max(s['support_score'] for s in supporting_sources)
            support_count = len(supporting_sources)
            support_strength = max_support * min(1.0, support_count / 2)  # Diminishing returns
        else:
            support_strength = 0.0
        
        # Calculate contradiction strength
        if contradicting_sources:
            max_contradiction = max(s['contradiction_score'] for s in contradicting_sources)
            contradiction_strength = max_contradiction * 0.8  # Slightly lower weight
        else:
            contradiction_strength = 0.0
        
        # Final confidence calculation
        if contradicting_sources and not supporting_sources:
            confidence = contradiction_strength  # High confidence in contradiction
        elif supporting_sources and not contradicting_sources:
            confidence = support_strength  # Confidence in support
        else:
            # Both support and contradiction found - lower confidence
            confidence = max(support_strength, contradiction_strength) * 0.6
        
        return round(confidence, 3)
    
    def _calculate_fact_check_score(
        self,
        source_verification: Dict[str, Any],
        external_verification: Dict[str, Any],
        citation_coverage: Dict[str, Any],
        statistical_validation: Dict[str, Any],
        temporal_accuracy: Dict[str, Any]
    ) -> float:
        """Calculate overall fact-checking score."""
        
        # Weighted scoring
        weights = {
            'source_verification': 0.35,
            'citation_coverage': 0.25,
            'statistical_validation': 0.20,
            'external_verification': 0.15,
            'temporal_accuracy': 0.05
        }
        
        scores = {
            'source_verification': source_verification.get('verification_score', 0.5),
            'citation_coverage': citation_coverage.get('citation_coverage_score', 0.5),
            'statistical_validation': statistical_validation.get('statistical_validation_score', 0.8),
            'external_verification': external_verification.get('external_verification_score', 0.5),
            'temporal_accuracy': temporal_accuracy.get('temporal_accuracy_score', 0.9)
        }
        
        overall_score = sum(scores[component] * weights[component] for component in weights.keys())
        
        return round(min(1.0, max(0.0, overall_score)), 3)
    
    def _create_verification_summary(self, verified_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary of claim verification results."""
        
        if not verified_claims:
            return {
                'total_claims': 0,
                'verification_distribution': {},
                'confidence_stats': {}
            }
        
        # Count verification statuses
        from collections import Counter
        status_counts = Counter(claim['verification_status'] for claim in verified_claims)
        
        # Calculate confidence statistics
        confidences = [claim['confidence_score'] for claim in verified_claims if 'confidence_score' in claim]
        
        confidence_stats = {}
        if confidences:
            confidence_stats = {
                'avg_confidence': round(sum(confidences) / len(confidences), 3),
                'min_confidence': round(min(confidences), 3),
                'max_confidence': round(max(confidences), 3),
                'high_confidence_claims': sum(1 for c in confidences if c >= 0.8)
            }
        
        return {
            'total_claims': len(verified_claims),
            'verification_distribution': dict(status_counts),
            'confidence_stats': confidence_stats,
            'well_supported_claims': status_counts.get('strongly_supported', 0) + status_counts.get('supported', 0),
            'problematic_claims': status_counts.get('contradicted', 0) + status_counts.get('unverified', 0)
        }
    
    def _generate_fact_check_recommendations(
        self,
        fact_check_score: float,
        source_verification: Dict[str, Any],
        external_verification: Dict[str, Any],
        citation_coverage: Dict[str, Any],
        statistical_validation: Dict[str, Any]
    ) -> List[str]:
        """Generate fact-checking improvement recommendations."""
        
        recommendations = []
        
        # Overall score recommendations
        if fact_check_score < 0.6:
            recommendations.append("Fact-checking score is below threshold - comprehensive review required")
        elif fact_check_score < 0.8:
            recommendations.append("Fact-checking score is acceptable but could be improved")
        else:
            recommendations.append("Fact-checking score is excellent - content appears well-verified")
        
        # Source verification recommendations
        verification_score = source_verification.get('verification_score', 1.0)
        if verification_score < 0.7:
            unsupported_claims = source_verification.get('unsupported_claims', 0)
            if unsupported_claims > 0:
                recommendations.append(f"Verify or add citations for {unsupported_claims} unsupported claims")
        
        # Citation coverage recommendations
        coverage_score = citation_coverage.get('citation_coverage_score', 1.0)
        if coverage_score < 0.5:
            recommendations.append("Improve citation coverage - many claims lack proper source attribution")
        
        uncited_claims = citation_coverage.get('uncited_claims', [])
        if uncited_claims:
            recommendations.append("Add citations for factual claims that currently lack source attribution")
        
        # Statistical validation recommendations
        stat_score = statistical_validation.get('statistical_validation_score', 1.0)
        questionable_claims = statistical_validation.get('questionable_claims', 0)
        if questionable_claims > 0:
            recommendations.append(f"Review {questionable_claims} questionable statistical claims for accuracy")
        
        # External verification recommendations
        if external_verification:
            external_score = external_verification.get('external_verification_score', 0.5)
            if external_score < 0.5:
                recommendations.append("Some claims may benefit from additional external verification")
        
        return recommendations[:6]  # Limit to 6 recommendations