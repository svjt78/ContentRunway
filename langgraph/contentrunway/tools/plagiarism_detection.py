"""Plagiarism detection tools for content similarity and originality checking."""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, List
import logging
import re
import hashlib
from collections import Counter
import asyncio
import aiohttp
from datetime import datetime
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class PlagiarismDetectionTool:
    """Tool for detecting plagiarism and content similarity using semantic embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize plagiarism detection tool.
        
        Args:
            model_name: Sentence transformer model for embeddings
        """
        try:
            # Load sentence transformer model
            self.encoder = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            self.encoder = None
        
        # Similarity thresholds
        self.similarity_thresholds = {
            'exact_match': 0.98,      # Almost identical text
            'high_similarity': 0.85,  # Likely plagiarism
            'moderate_similarity': 0.70,  # Possible paraphrasing
            'low_similarity': 0.50,   # Some similarity but likely original
        }
        
        # Chunk sizes for analysis
        self.sentence_chunk_size = 3  # Analyze 3 sentences at a time
        self.min_chunk_length = 20   # Minimum words per chunk
        
        # Web search for plagiarism checking
        self.search_session = None
        self._init_search_session()
    
    def _init_search_session(self):
        """Initialize HTTP session for web searches."""
        self.search_session = aiohttp.ClientSession(
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            },
            timeout=aiohttp.ClientTimeout(total=10)
        )
    
    async def check_content_originality(
        self,
        content: str,
        sources: List[Dict[str, Any]] = None,
        check_web: bool = True,
        check_milvus: bool = True
    ) -> Dict[str, Any]:
        """
        Check content originality against sources and web content.
        
        Args:
            content: Content to check for plagiarism
            sources: Source documents to check against
            check_web: Whether to perform web-based plagiarism detection
            check_milvus: Whether to check against published content in Milvus
            
        Returns:
            Dictionary with plagiarism analysis results
        """
        logger.info("Starting plagiarism detection analysis")
        
        if not self.encoder:
            return {
                'originality_score': 0.8,  # Conservative fallback
                'error': 'Sentence transformer model not available',
                'analysis_completed': False
            }
        
        try:
            # Step 1: Analyze content structure
            content_analysis = self._analyze_content_structure(content)
            
            # Step 2: Check against provided sources
            source_similarity = await self._check_against_sources(content, sources or [])
            
            # Step 2.5: Check against published content in Milvus (if enabled)
            milvus_similarity = {}
            if check_milvus:
                milvus_similarity = await self._check_against_milvus(content)
            
            # Step 3: Web-based plagiarism check (if enabled)
            web_similarity = {}
            if check_web:
                web_similarity = await self._check_against_web(content)
            
            # Step 4: Analyze text patterns for potential issues
            pattern_analysis = self._analyze_suspicious_patterns(content)
            
            # Step 5: Calculate overall originality score
            originality_score = self._calculate_originality_score(
                source_similarity,
                web_similarity,
                pattern_analysis,
                content_analysis,
                milvus_similarity
            )
            
            # Step 6: Generate detailed report
            report = self._generate_plagiarism_report(
                content_analysis,
                source_similarity,
                web_similarity,
                pattern_analysis,
                originality_score,
                milvus_similarity
            )
            
            logger.info(f"Plagiarism check completed: {originality_score:.3f} originality score")
            
            return {
                'originality_score': originality_score,
                'analysis_completed': True,
                'source_similarity': source_similarity,
                'milvus_similarity': milvus_similarity,
                'web_similarity': web_similarity,
                'pattern_analysis': pattern_analysis,
                'detailed_report': report,
                'recommendations': self._generate_originality_recommendations(
                    originality_score, source_similarity, pattern_analysis
                ),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Plagiarism detection failed: {e}")
            return {
                'originality_score': 0.7,  # Conservative fallback
                'error': str(e),
                'analysis_completed': False
            }
        finally:
            if self.search_session and not self.search_session.closed:
                await self.search_session.close()
    
    def _analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """Analyze content structure for plagiarism detection."""
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) >= 5]
        
        # Split into chunks for analysis
        chunks = self._create_text_chunks(content)
        
        # Calculate text fingerprints
        fingerprints = [self._calculate_text_fingerprint(chunk) for chunk in chunks]
        
        return {
            'total_sentences': len(sentences),
            'analysis_chunks': len(chunks),
            'average_chunk_length': sum(len(chunk.split()) for chunk in chunks) / len(chunks) if chunks else 0,
            'text_fingerprints': fingerprints,
            'content_chunks': chunks[:10]  # Store first 10 chunks for analysis
        }
    
    def _create_text_chunks(self, content: str) -> List[str]:
        """Create overlapping text chunks for similarity analysis."""
        
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        
        # Create overlapping chunks of sentences
        for i in range(0, len(sentences), self.sentence_chunk_size - 1):  # Overlap by 1 sentence
            chunk_sentences = sentences[i:i + self.sentence_chunk_size]
            chunk_text = '. '.join(chunk_sentences)
            
            # Only include chunks with sufficient content
            if len(chunk_text.split()) >= self.min_chunk_length:
                chunks.append(chunk_text)
        
        return chunks
    
    def _calculate_text_fingerprint(self, text: str) -> str:
        """Calculate a hash fingerprint for text chunk."""
        
        # Normalize text (remove punctuation, lowercase, remove extra spaces)
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        normalized = ' '.join(normalized.split())  # Normalize whitespace
        
        # Create SHA-256 hash
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]
    
    async def _check_against_sources(self, content: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check content similarity against provided sources."""
        
        if not sources:
            return {
                'similarity_score': 0.0,
                'max_similarity': 0.0,
                'similar_sources': [],
                'exact_matches': []
            }
        
        try:
            # Prepare content chunks
            content_chunks = self._create_text_chunks(content)
            
            if not content_chunks:
                return {
                    'similarity_score': 0.0,
                    'max_similarity': 0.0,
                    'similar_sources': [],
                    'exact_matches': []
                }
            
            # Extract text from sources
            source_texts = []
            for source in sources:
                source_content = ""
                if 'content' in source and source['content']:
                    source_content = source['content']
                elif 'summary' in source and source['summary']:
                    source_content = source['summary']
                elif 'title' in source:
                    source_content = source['title']
                
                if source_content and len(source_content.strip()) > 50:
                    source_texts.append({
                        'text': source_content,
                        'source_info': {
                            'title': source.get('title', 'Unknown'),
                            'url': source.get('url', ''),
                            'domain': source.get('domain', 'unknown')
                        }
                    })
            
            if not source_texts:
                return {
                    'similarity_score': 0.0,
                    'max_similarity': 0.0,
                    'similar_sources': [],
                    'exact_matches': []
                }
            
            # Encode content chunks and source texts
            all_texts = content_chunks + [s['text'] for s in source_texts]
            embeddings = self.encoder.encode(all_texts)
            
            # Split embeddings
            content_embeddings = embeddings[:len(content_chunks)]
            source_embeddings = embeddings[len(content_chunks):]
            
            # Calculate similarities
            similarity_matrix = cosine_similarity(content_embeddings, source_embeddings)
            
            # Find high similarity matches
            similar_sources = []
            max_similarity = 0.0
            exact_matches = []
            
            for i, content_chunk in enumerate(content_chunks):
                for j, source_info in enumerate(source_texts):
                    similarity = similarity_matrix[i][j]
                    max_similarity = max(max_similarity, similarity)
                    
                    if similarity >= self.similarity_thresholds['high_similarity']:
                        match_info = {
                            'content_chunk': content_chunk[:200] + "..." if len(content_chunk) > 200 else content_chunk,
                            'source_text': source_info['text'][:200] + "..." if len(source_info['text']) > 200 else source_info['text'],
                            'similarity_score': round(similarity, 3),
                            'source_info': source_info['source_info'],
                            'similarity_level': self._categorize_similarity(similarity)
                        }
                        
                        similar_sources.append(match_info)
                        
                        # Check for exact matches
                        if similarity >= self.similarity_thresholds['exact_match']:
                            exact_matches.append(match_info)
            
            # Calculate overall similarity score
            avg_max_similarity = np.mean([np.max(row) for row in similarity_matrix])
            
            return {
                'similarity_score': round(avg_max_similarity, 3),
                'max_similarity': round(max_similarity, 3),
                'similar_sources': similar_sources[:5],  # Top 5 matches
                'exact_matches': exact_matches,
                'sources_analyzed': len(source_texts),
                'chunks_analyzed': len(content_chunks)
            }
            
        except Exception as e:
            logger.error(f"Source similarity check failed: {e}")
            return {
                'similarity_score': 0.0,
                'max_similarity': 0.0,
                'similar_sources': [],
                'exact_matches': [],
                'error': str(e)
            }
    
    async def _check_against_web(self, content: str, max_checks: int = 5) -> Dict[str, Any]:
        """Check content against web sources for plagiarism."""
        
        try:
            # Create distinctive phrases for web searching
            distinctive_phrases = self._extract_distinctive_phrases(content)
            
            web_matches = []
            total_searches = 0
            
            # Search for distinctive phrases
            for phrase in distinctive_phrases[:max_checks]:
                try:
                    search_results = await self._search_phrase_on_web(phrase)
                    total_searches += 1
                    
                    if search_results:
                        for result in search_results:
                            # Calculate similarity with original phrase
                            similarity = await self._calculate_phrase_similarity(phrase, result['snippet'])
                            
                            if similarity >= self.similarity_thresholds['moderate_similarity']:
                                web_matches.append({
                                    'original_phrase': phrase,
                                    'found_text': result['snippet'],
                                    'similarity_score': similarity,
                                    'source_url': result['url'],
                                    'source_title': result['title'],
                                    'similarity_level': self._categorize_similarity(similarity)
                                })
                    
                    # Rate limiting
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.warning(f"Web search failed for phrase '{phrase[:50]}...': {e}")
                    continue
            
            # Calculate web similarity metrics
            if web_matches:
                max_web_similarity = max(match['similarity_score'] for match in web_matches)
                avg_web_similarity = sum(match['similarity_score'] for match in web_matches) / len(web_matches)
            else:
                max_web_similarity = 0.0
                avg_web_similarity = 0.0
            
            return {
                'web_similarity_score': round(avg_web_similarity, 3),
                'max_web_similarity': round(max_web_similarity, 3),
                'web_matches': web_matches[:3],  # Top 3 matches
                'searches_performed': total_searches,
                'phrases_checked': len(distinctive_phrases)
            }
            
        except Exception as e:
            logger.error(f"Web plagiarism check failed: {e}")
            return {
                'web_similarity_score': 0.0,
                'max_web_similarity': 0.0,
                'web_matches': [],
                'error': str(e)
            }
    
    def _extract_distinctive_phrases(self, content: str, max_phrases: int = 8) -> List[str]:
        """Extract distinctive phrases that are good for web searching."""
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        distinctive_phrases = []
        
        for sentence in sentences:
            words = sentence.split()
            
            # Skip very short or very long sentences
            if len(words) < 8 or len(words) > 25:
                continue
            
            # Skip sentences that are mostly common words
            uncommon_words = [
                word for word in words 
                if len(word) > 4 and word.lower() not in [
                    'that', 'this', 'with', 'from', 'they', 'have', 'been',
                    'their', 'which', 'would', 'could', 'should', 'about'
                ]
            ]
            
            if len(uncommon_words) >= 4:  # Must have at least 4 substantial words
                # Clean up the sentence for searching
                clean_sentence = re.sub(r'[^\w\s]', '', sentence).strip()
                if clean_sentence:
                    distinctive_phrases.append(clean_sentence)
        
        # Sort by length (longer phrases are more distinctive)
        distinctive_phrases.sort(key=len, reverse=True)
        
        return distinctive_phrases[:max_phrases]
    
    async def _check_against_milvus(self, content: str) -> Dict[str, Any]:
        """Check content against published content stored in Milvus."""
        try:
            from app.services.vector_service import vector_service
            
            # Use VectorService to find similar published content
            similar_content = await vector_service.find_similar_content(
                content_text=content,
                limit=10
            )
            
            # Calculate similarity metrics
            if similar_content:
                max_milvus_similarity = max(item['similarity_score'] for item in similar_content)
                avg_milvus_similarity = sum(item['similarity_score'] for item in similar_content) / len(similar_content)
                
                # Filter for concerning similarities (>0.75)
                concerning_matches = [
                    item for item in similar_content 
                    if item['similarity_score'] > 0.75
                ]
            else:
                max_milvus_similarity = 0.0
                avg_milvus_similarity = 0.0
                concerning_matches = []
            
            return {
                'milvus_similarity_score': round(avg_milvus_similarity, 3),
                'max_milvus_similarity': round(max_milvus_similarity, 3),
                'milvus_matches': concerning_matches,
                'published_content_checked': len(similar_content),
                'concerning_matches_count': len(concerning_matches)
            }
            
        except Exception as e:
            logger.error(f"Milvus plagiarism check failed: {e}")
            return {
                'milvus_similarity_score': 0.0,
                'max_milvus_similarity': 0.0,
                'milvus_matches': [],
                'published_content_checked': 0,
                'error': str(e)
            }
    
    async def _search_phrase_on_web(self, phrase: str) -> List[Dict[str, Any]]:
        """Search for a phrase on the web to detect potential plagiarism."""
        
        try:
            # Use exact phrase search with quotes
            search_query = f'"{phrase}"'
            search_url = f"https://www.google.com/search?q={search_query}&num=5"
            
            if not self.search_session or self.search_session.closed:
                self._init_search_session()
            
            async with self.search_session.get(search_url) as response:
                if response.status != 200:
                    return []
                
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                results = []
                
                # Extract search results
                for result_elem in soup.select('div.g, .tF2Cxc')[:3]:  # Top 3 results
                    try:
                        # Extract title
                        title_elem = result_elem.select_one('h3, .LC20lb')
                        title = title_elem.get_text(strip=True) if title_elem else ""
                        
                        # Extract URL
                        link_elem = result_elem.select_one('a[href]')
                        url = link_elem.get('href', '') if link_elem else ""
                        
                        # Extract snippet
                        snippet_elem = result_elem.select_one('.VwiC3b, .s3v9rd')
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                        
                        if title and url and snippet:
                            results.append({
                                'title': title,
                                'url': url,
                                'snippet': snippet
                            })
                            
                    except Exception as e:
                        logger.warning(f"Failed to parse search result: {e}")
                        continue
                
                return results
                
        except Exception as e:
            logger.warning(f"Web search failed for phrase: {e}")
            return []
    
    async def _calculate_phrase_similarity(self, original_phrase: str, found_text: str) -> float:
        """Calculate semantic similarity between original phrase and found text."""
        
        if not self.encoder:
            return 0.0
        
        try:
            # Encode both texts
            embeddings = self.encoder.encode([original_phrase, found_text])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _analyze_suspicious_patterns(self, content: str) -> Dict[str, Any]:
        """Analyze text patterns that might indicate plagiarism or AI generation."""
        
        # Check for repetitive phrases
        repetitive_score = self._check_repetitive_patterns(content)
        
        # Check for inconsistent style
        style_consistency = self._check_style_consistency(content)
        
        # Check for unusual vocabulary patterns
        vocabulary_analysis = self._analyze_vocabulary_patterns(content)
        
        # Check for citation anomalies
        citation_analysis = self._analyze_citation_patterns(content)
        
        # Calculate overall suspicion score
        suspicion_score = (
            (1 - repetitive_score) * 0.3 +
            (1 - style_consistency) * 0.3 +
            vocabulary_analysis.get('anomaly_score', 0) * 0.2 +
            citation_analysis.get('anomaly_score', 0) * 0.2
        )
        
        return {
            'suspicion_score': round(suspicion_score, 3),
            'repetitive_patterns_score': repetitive_score,
            'style_consistency_score': style_consistency,
            'vocabulary_analysis': vocabulary_analysis,
            'citation_analysis': citation_analysis,
            'potential_issues': self._identify_potential_issues(
                repetitive_score, style_consistency, vocabulary_analysis, citation_analysis
            )
        }
    
    def _check_repetitive_patterns(self, content: str) -> float:
        """Check for repetitive phrases that might indicate copy-paste behavior."""
        
        # Extract phrases (3-6 words)
        words = content.lower().split()
        phrases = []
        
        for length in range(3, 7):  # 3 to 6 word phrases
            for i in range(len(words) - length + 1):
                phrase = ' '.join(words[i:i + length])
                phrases.append(phrase)
        
        # Count phrase frequencies
        phrase_counts = Counter(phrases)
        
        # Find repetitive phrases (appearing 3+ times)
        repetitive_phrases = [phrase for phrase, count in phrase_counts.items() if count >= 3]
        
        # Calculate repetition score (lower is better)
        total_phrases = len(phrases)
        repetitive_ratio = len(repetitive_phrases) / total_phrases if total_phrases > 0 else 0
        
        # Convert to score (higher is better, less repetitive)
        repetition_score = max(0.0, 1.0 - repetitive_ratio * 10)  # Penalize repetition
        
        return round(repetition_score, 3)
    
    def _check_style_consistency(self, content: str) -> float:
        """Check for style consistency throughout the content."""
        
        # Split content into sections
        sections = re.split(r'\n\s*\n', content)
        sections = [s.strip() for s in sections if len(s.strip()) > 100]  # Substantial sections only
        
        if len(sections) < 3:
            return 1.0  # Can't assess consistency with few sections
        
        try:
            # Analyze style features for each section
            section_features = []
            
            for section in sections:
                features = {
                    'avg_sentence_length': self._calculate_avg_sentence_length(section),
                    'complexity_score': self._calculate_complexity_score(section),
                    'formality_score': self._calculate_formality_score(section),
                    'vocabulary_diversity': len(set(section.lower().split())) / len(section.split()) if section.split() else 0
                }
                section_features.append(features)
            
            # Calculate consistency (low variance = high consistency)
            consistency_scores = []
            
            for feature in ['avg_sentence_length', 'complexity_score', 'formality_score', 'vocabulary_diversity']:
                values = [sf[feature] for sf in section_features]
                if values:
                    std_dev = np.std(values)
                    mean_val = np.mean(values)
                    # Normalize by mean to get coefficient of variation
                    cv = std_dev / mean_val if mean_val > 0 else 0
                    consistency_score = max(0.0, 1.0 - cv)  # Lower variance = higher consistency
                    consistency_scores.append(consistency_score)
            
            overall_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
            
            return round(overall_consistency, 3)
            
        except Exception as e:
            logger.warning(f"Style consistency check failed: {e}")
            return 0.8  # Conservative fallback
    
    def _calculate_avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length for a text section."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        total_words = sum(len(s.split()) for s in sentences)
        return total_words / len(sentences)
    
    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate text complexity score."""
        words = text.split()
        
        if not words:
            return 0.0
        
        # Count complex words (3+ syllables or 7+ characters)
        complex_words = sum(1 for word in words if len(word) >= 7 or self._estimate_syllables(word) >= 3)
        
        return complex_words / len(words)
    
    def _calculate_formality_score(self, text: str) -> float:
        """Calculate formality score of text."""
        text_lower = text.lower()
        
        formal_indicators = [
            'furthermore', 'moreover', 'consequently', 'therefore',
            'however', 'nevertheless', 'accordingly', 'thus'
        ]
        
        informal_indicators = [
            'gonna', 'wanna', 'yeah', 'ok', 'cool', 'awesome',
            'really', 'totally', 'basically', 'literally'
        ]
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
        informal_count = sum(1 for indicator in informal_indicators if indicator in text_lower)
        
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.5
        
        # Calculate formality ratio
        formality_ratio = (formal_count - informal_count + word_count / 100) / (word_count / 100)
        
        return max(0.0, min(1.0, formality_ratio))
    
    def _estimate_syllables(self, word: str) -> int:
        """Simple syllable estimation."""
        word = word.lower().strip('.,!?;:"')
        if not word:
            return 0
        
        vowels = 'aeiouy'
        syllables = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllables += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllables > 1:
            syllables -= 1
        
        return max(1, syllables)
    
    def _analyze_vocabulary_patterns(self, content: str) -> Dict[str, Any]:
        """Analyze vocabulary usage patterns for anomalies."""
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        if not words:
            return {'anomaly_score': 0.0}
        
        # Calculate vocabulary diversity
        unique_words = set(words)
        vocabulary_diversity = len(unique_words) / len(words)
        
        # Check for unusual word frequency distributions
        word_counts = Counter(words)
        
        # Find overused words (excluding common words)
        common_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him',
            'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way',
            'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'
        }
        
        overused_words = []
        total_words = len(words)
        
        for word, count in word_counts.most_common(20):
            if word not in common_words and count / total_words > 0.01:  # >1% frequency
                overused_words.append({'word': word, 'frequency': round(count / total_words * 100, 2)})
        
        # Calculate anomaly score
        anomaly_score = 0.0
        
        # Low vocabulary diversity might indicate copying
        if vocabulary_diversity < 0.4:
            anomaly_score += 0.3
        
        # Too many overused words might indicate poor writing or copying
        if len(overused_words) > 5:
            anomaly_score += 0.2
        
        # Very high vocabulary diversity might indicate multiple sources
        if vocabulary_diversity > 0.8:
            anomaly_score += 0.1
        
        return {
            'anomaly_score': round(min(1.0, anomaly_score), 3),
            'vocabulary_diversity': round(vocabulary_diversity, 3),
            'overused_words': overused_words[:5],
            'unique_words_count': len(unique_words),
            'total_words_count': total_words
        }
    
    def _analyze_citation_patterns(self, content: str) -> Dict[str, Any]:
        """Analyze citation patterns for anomalies."""
        
        # Find all citation markers
        citation_patterns = [
            r'\[(\d+)\]',           # [1]
            r'\[Citation (\d+)\]',  # [Citation 1]
            r'\(([^)]+\s+\d{4})\)', # (Author 2020)
            r'\[([^\]]+)\]'         # [Source Name]
        ]
        
        all_citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, content)
            all_citations.extend(matches)
        
        # Analyze citation distribution
        citation_positions = []
        for pattern in citation_patterns:
            for match in re.finditer(pattern, content):
                position = match.start() / len(content)  # Relative position (0-1)
                citation_positions.append(position)
        
        # Check for citation clustering (might indicate copy-paste)
        anomaly_score = 0.0
        
        if citation_positions:
            # Check if citations are clustered in one area
            citation_positions.sort()
            gaps = [citation_positions[i+1] - citation_positions[i] for i in range(len(citation_positions)-1)]
            
            if gaps:
                max_gap = max(gaps)
                if max_gap > 0.7:  # Large gap without citations
                    anomaly_score += 0.3
        
        # Check citation frequency
        words_per_citation = len(content.split()) / len(all_citations) if all_citations else float('inf')
        
        if words_per_citation < 50:  # Very frequent citations might indicate copying
            anomaly_score += 0.2
        elif words_per_citation > 500:  # Very rare citations might indicate insufficient sourcing
            anomaly_score += 0.1
        
        return {
            'anomaly_score': round(min(1.0, anomaly_score), 3),
            'total_citations': len(all_citations),
            'citation_density': round(len(all_citations) / len(content.split()) * 100, 2) if content.split() else 0,
            'words_per_citation': round(words_per_citation, 1) if words_per_citation != float('inf') else None,
            'citation_distribution': 'clustered' if len(citation_positions) > 0 and gaps and max(gaps) > 0.5 else 'distributed'
        }
    
    def _calculate_originality_score(
        self,
        source_similarity: Dict[str, Any],
        web_similarity: Dict[str, Any],
        pattern_analysis: Dict[str, Any],
        content_analysis: Dict[str, Any],
        milvus_similarity: Dict[str, Any] = None
    ) -> float:
        """Calculate overall content originality score."""
        
        # Start with perfect score
        originality_score = 1.0
        
        # Penalize based on source similarity
        max_source_similarity = source_similarity.get('max_similarity', 0.0)
        if max_source_similarity >= self.similarity_thresholds['high_similarity']:
            originality_score -= 0.4
        elif max_source_similarity >= self.similarity_thresholds['moderate_similarity']:
            originality_score -= 0.2
        
        # Penalize based on web similarity
        max_web_similarity = web_similarity.get('max_web_similarity', 0.0)
        if max_web_similarity >= self.similarity_thresholds['high_similarity']:
            originality_score -= 0.3
        elif max_web_similarity >= self.similarity_thresholds['moderate_similarity']:
            originality_score -= 0.15
        
        # Penalize based on Milvus similarity (published content)
        if milvus_similarity:
            max_milvus_similarity = milvus_similarity.get('max_milvus_similarity', 0.0)
            if max_milvus_similarity >= self.similarity_thresholds['high_similarity']:
                originality_score -= 0.4  # Higher penalty for self-plagiarism
            elif max_milvus_similarity >= self.similarity_thresholds['moderate_similarity']:
                originality_score -= 0.2
        
        # Penalize based on suspicious patterns
        suspicion_score = pattern_analysis.get('suspicion_score', 0.0)
        originality_score -= suspicion_score * 0.2
        
        # Apply minimum threshold
        originality_score = max(0.0, originality_score)
        
        return round(originality_score, 3)
    
    def _categorize_similarity(self, similarity: float) -> str:
        """Categorize similarity score into readable levels."""
        
        if similarity >= self.similarity_thresholds['exact_match']:
            return 'exact_match'
        elif similarity >= self.similarity_thresholds['high_similarity']:
            return 'high_similarity'
        elif similarity >= self.similarity_thresholds['moderate_similarity']:
            return 'moderate_similarity'
        elif similarity >= self.similarity_thresholds['low_similarity']:
            return 'low_similarity'
        else:
            return 'no_similarity'
    
    def _identify_potential_issues(
        self,
        repetitive_score: float,
        style_score: float,
        vocab_analysis: Dict[str, Any],
        citation_analysis: Dict[str, Any]
    ) -> List[str]:
        """Identify specific potential plagiarism issues."""
        
        issues = []
        
        if repetitive_score < 0.7:
            issues.append("High repetitive pattern detected - check for copy-paste behavior")
        
        if style_score < 0.6:
            issues.append("Inconsistent writing style - may indicate multiple sources or authors")
        
        if vocab_analysis.get('vocabulary_diversity', 1.0) < 0.4:
            issues.append("Low vocabulary diversity - may indicate copied content")
        
        if vocab_analysis.get('anomaly_score', 0.0) > 0.3:
            issues.append("Unusual vocabulary patterns detected")
        
        if citation_analysis.get('anomaly_score', 0.0) > 0.3:
            issues.append("Unusual citation patterns detected")
        
        citation_density = citation_analysis.get('citation_density', 0)
        if citation_density < 0.5:
            issues.append("Very low citation density - may indicate insufficient sourcing")
        elif citation_density > 5.0:
            issues.append("Very high citation density - check for over-reliance on sources")
        
        return issues
    
    def _generate_plagiarism_report(
        self,
        content_analysis: Dict[str, Any],
        source_similarity: Dict[str, Any],
        web_similarity: Dict[str, Any],
        pattern_analysis: Dict[str, Any],
        originality_score: float,
        milvus_similarity: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive plagiarism detection report."""
        
        # Determine overall status
        if originality_score >= 0.9:
            status = 'ORIGINAL'
            risk_level = 'low'
        elif originality_score >= 0.8:
            status = 'MOSTLY_ORIGINAL'
            risk_level = 'low'
        elif originality_score >= 0.7:
            status = 'ACCEPTABLE'
            risk_level = 'medium'
        elif originality_score >= 0.6:
            status = 'CONCERNING'
            risk_level = 'medium'
        else:
            status = 'HIGH_RISK'
            risk_level = 'high'
        
        # Collect all issues
        all_issues = []
        all_issues.extend(pattern_analysis.get('potential_issues', []))
        
        if source_similarity.get('exact_matches'):
            all_issues.append(f"Found {len(source_similarity['exact_matches'])} exact matches with sources")
        
        if web_similarity.get('web_matches'):
            all_issues.append(f"Found {len(web_similarity['web_matches'])} similar content on web")
        
        if milvus_similarity and milvus_similarity.get('milvus_matches'):
            all_issues.append(f"Found {len(milvus_similarity['milvus_matches'])} similar published content")
        
        return {
            'originality_status': status,
            'risk_level': risk_level,
            'overall_score': originality_score,
            'content_analyzed': {
                'chunks_processed': content_analysis.get('analysis_chunks', 0),
                'sentences_processed': content_analysis.get('total_sentences', 0)
            },
            'similarity_analysis': {
                'source_max_similarity': source_similarity.get('max_similarity', 0.0),
                'web_max_similarity': web_similarity.get('max_web_similarity', 0.0),
                'milvus_max_similarity': milvus_similarity.get('max_milvus_similarity', 0.0) if milvus_similarity else 0.0,
                'sources_checked': source_similarity.get('sources_analyzed', 0),
                'web_searches_performed': web_similarity.get('searches_performed', 0),
                'published_content_checked': milvus_similarity.get('published_content_checked', 0) if milvus_similarity else 0
            },
            'pattern_analysis_summary': {
                'suspicion_score': pattern_analysis.get('suspicion_score', 0.0),
                'style_consistency': pattern_analysis.get('style_consistency_score', 1.0),
                'repetitive_patterns': pattern_analysis.get('repetitive_patterns_score', 1.0)
            },
            'issues_identified': all_issues,
            'requires_manual_review': risk_level in ['medium', 'high'] or len(all_issues) > 2,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_originality_recommendations(
        self,
        originality_score: float,
        source_similarity: Dict[str, Any],
        pattern_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for improving content originality."""
        
        recommendations = []
        
        if originality_score < 0.7:
            recommendations.append("Review content for potential plagiarism - originality score is below threshold")
        
        # Source similarity recommendations
        similar_sources = source_similarity.get('similar_sources', [])
        if similar_sources:
            recommendations.append("Rewrite sections with high similarity to sources using original analysis")
            recommendations.append("Add more original insights and commentary to distinguish from sources")
        
        exact_matches = source_similarity.get('exact_matches', [])
        if exact_matches:
            recommendations.append("Rewrite exact matches with sources - ensure proper paraphrasing and citation")
        
        # Pattern-based recommendations
        potential_issues = pattern_analysis.get('potential_issues', [])
        if 'High repetitive pattern detected' in ' '.join(potential_issues):
            recommendations.append("Reduce repetitive phrases and vary language throughout content")
        
        if 'Inconsistent writing style' in ' '.join(potential_issues):
            recommendations.append("Review content for style consistency - ensure unified voice")
        
        if 'Low vocabulary diversity' in ' '.join(potential_issues):
            recommendations.append("Expand vocabulary usage and avoid repetitive language")
        
        # General recommendations
        if originality_score >= 0.9:
            recommendations.append("Content originality is excellent - no major concerns")
        elif originality_score >= 0.8:
            recommendations.append("Content originality is good - minor improvements may be beneficial")
        
        return recommendations[:6]  # Limit to 6 recommendations