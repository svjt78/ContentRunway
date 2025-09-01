"""Content analysis tools for text quality assessment and readability metrics."""

import re
import spacy
import textstat
from typing import Dict, Any, List, Optional
from collections import Counter
import logging
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class ContentAnalysisTool:
    """Tool for comprehensive content analysis including readability, style, and quality metrics."""
    
    def __init__(self):
        """Initialize content analysis tool with NLP models."""
        try:
            # Load spaCy model for linguistic analysis
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # TF-IDF vectorizer for keyword analysis
        self.tfidf = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Style scoring parameters
        self.professional_indicators = [
            'analysis', 'research', 'study', 'findings', 'data', 'results',
            'methodology', 'implementation', 'framework', 'strategy',
            'best practices', 'industry', 'professional', 'enterprise'
        ]
        
        self.engagement_indicators = [
            'you', 'your', 'we', 'let\'s', 'imagine', 'consider',
            'example', 'case study', 'real-world', 'practical',
            'actionable', 'tips', 'guide', 'how to'
        ]
    
    def analyze_content_quality(self, content: str, target_audience: str = "professionals") -> Dict[str, Any]:
        """
        Perform comprehensive content quality analysis.
        
        Args:
            content: Text content to analyze
            target_audience: Target audience type
            
        Returns:
            Dictionary with detailed quality metrics
        """
        logger.info("Starting comprehensive content quality analysis")
        
        try:
            # Basic metrics
            basic_metrics = self._calculate_basic_metrics(content)
            
            # Readability analysis
            readability_metrics = self._analyze_readability(content)
            
            # Style analysis
            style_analysis = self._analyze_writing_style(content, target_audience)
            
            # Structure analysis
            structure_metrics = self._analyze_structure(content)
            
            # Keyword analysis
            keyword_analysis = self._analyze_keywords(content)
            
            # Professional tone analysis
            tone_analysis = self._analyze_professional_tone(content)
            
            # Calculate overall quality score
            overall_score = self._calculate_overall_quality_score(
                readability_metrics,
                style_analysis,
                structure_metrics,
                tone_analysis
            )
            
            return {
                'overall_score': overall_score,
                'basic_metrics': basic_metrics,
                'readability': readability_metrics,
                'style_analysis': style_analysis,
                'structure': structure_metrics,
                'keywords': keyword_analysis,
                'tone': tone_analysis,
                'recommendations': self._generate_improvement_recommendations(
                    readability_metrics, style_analysis, structure_metrics, tone_analysis
                ),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Content quality analysis failed: {e}")
            return {
                'overall_score': 0.5,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_basic_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate basic content metrics."""
        
        # Text statistics
        word_count = len(content.split())
        char_count = len(content)
        char_count_no_spaces = len(content.replace(' ', ''))
        
        # Sentence analysis
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        # Paragraph analysis
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        # Average calculations
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        avg_sentences_per_paragraph = sentence_count / paragraph_count if paragraph_count > 0 else 0
        avg_words_per_paragraph = word_count / paragraph_count if paragraph_count > 0 else 0
        
        return {
            'word_count': word_count,
            'character_count': char_count,
            'character_count_no_spaces': char_count_no_spaces,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'average_words_per_sentence': round(avg_words_per_sentence, 1),
            'average_sentences_per_paragraph': round(avg_sentences_per_paragraph, 1),
            'average_words_per_paragraph': round(avg_words_per_paragraph, 1)
        }
    
    def _analyze_readability(self, content: str) -> Dict[str, Any]:
        """Analyze content readability using multiple metrics."""
        
        try:
            # Flesch Reading Ease
            flesch_ease = textstat.flesch_reading_ease(content)
            
            # Flesch-Kincaid Grade Level
            flesch_kincaid = textstat.flesch_kincaid_grade(content)
            
            # Gunning Fog Index
            gunning_fog = textstat.gunning_fog(content)
            
            # SMOG Index
            smog_index = textstat.smog_index(content)
            
            # Automated Readability Index
            ari = textstat.automated_readability_index(content)
            
            # Coleman-Liau Index
            coleman_liau = textstat.coleman_liau_index(content)
            
            # Reading time estimation (200 words per minute)
            reading_time_minutes = max(1, round(len(content.split()) / 200))
            
            # Syllable analysis
            syllable_count = textstat.syllable_count(content)
            avg_syllables_per_word = syllable_count / len(content.split()) if content.split() else 0
            
            # Interpret readability level
            readability_level = self._interpret_flesch_score(flesch_ease)
            
            # Calculate composite readability score (0.0-1.0)
            # Optimal range for professional content: 60-80 Flesch score
            if 60 <= flesch_ease <= 80:
                readability_score = 1.0
            elif 50 <= flesch_ease < 60 or 80 < flesch_ease <= 90:
                readability_score = 0.8
            elif 40 <= flesch_ease < 50 or 90 < flesch_ease <= 100:
                readability_score = 0.6
            else:
                readability_score = 0.4
            
            return {
                'readability_score': readability_score,
                'flesch_reading_ease': flesch_ease,
                'flesch_kincaid_grade': flesch_kincaid,
                'gunning_fog_index': gunning_fog,
                'smog_index': smog_index,
                'automated_readability_index': ari,
                'coleman_liau_index': coleman_liau,
                'reading_time_minutes': reading_time_minutes,
                'syllable_count': syllable_count,
                'average_syllables_per_word': round(avg_syllables_per_word, 2),
                'readability_level': readability_level,
                'reading_difficulty': self._categorize_reading_difficulty(flesch_kincaid)
            }
            
        except Exception as e:
            logger.error(f"Readability analysis failed: {e}")
            return {
                'readability_score': 0.5,
                'error': str(e)
            }
    
    def _interpret_flesch_score(self, score: float) -> str:
        """Interpret Flesch Reading Ease score."""
        if score >= 90:
            return "Very Easy"
        elif score >= 80:
            return "Easy"
        elif score >= 70:
            return "Fairly Easy"
        elif score >= 60:
            return "Standard"
        elif score >= 50:
            return "Fairly Difficult"
        elif score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"
    
    def _categorize_reading_difficulty(self, grade_level: float) -> str:
        """Categorize reading difficulty based on grade level."""
        if grade_level <= 6:
            return "Elementary"
        elif grade_level <= 8:
            return "Middle School"
        elif grade_level <= 12:
            return "High School"
        elif grade_level <= 16:
            return "College Level"
        else:
            return "Graduate Level"
    
    def _analyze_writing_style(self, content: str, target_audience: str) -> Dict[str, Any]:
        """Analyze writing style characteristics."""
        
        content_lower = content.lower()
        words = content.split()
        
        # Voice analysis (active vs passive)
        passive_indicators = ['was', 'were', 'been', 'being', 'be']
        passive_count = sum(1 for word in words if word.lower() in passive_indicators)
        passive_voice_ratio = passive_count / len(words) if words else 0
        
        # Professional tone indicators
        professional_score = sum(1 for indicator in self.professional_indicators 
                                if indicator in content_lower) / len(self.professional_indicators)
        
        # Engagement indicators
        engagement_score = sum(1 for indicator in self.engagement_indicators 
                             if indicator in content_lower) / len(self.engagement_indicators)
        
        # Jargon analysis using spaCy (if available)
        technical_terms = []
        if self.nlp:
            doc = self.nlp(content)
            # Extract technical terms (nouns with capital letters or specific patterns)
            technical_terms = [
                token.text for token in doc 
                if (token.pos_ in ['NOUN', 'PROPN'] and 
                    (token.text[0].isupper() or len(token.text) > 8))
            ]
        
        # Sentence variety analysis
        sentence_lengths = []
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            if sentence.strip():
                sentence_lengths.append(len(sentence.split()))
        
        sentence_variety_score = 0.5
        if sentence_lengths:
            std_dev = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
            avg_length = np.mean(sentence_lengths)
            # Good variety means reasonable standard deviation
            if 5 <= std_dev <= 15 and 15 <= avg_length <= 25:
                sentence_variety_score = 1.0
            elif 3 <= std_dev <= 20 and 10 <= avg_length <= 30:
                sentence_variety_score = 0.8
            else:
                sentence_variety_score = 0.6
        
        # Calculate style appropriateness for target audience
        if target_audience == "professionals":
            # Professional content should balance professionalism with engagement
            audience_fit_score = (professional_score * 0.6 + engagement_score * 0.4)
        elif target_audience == "general":
            # General audience needs more engagement, less jargon
            audience_fit_score = (engagement_score * 0.7 + (1 - professional_score) * 0.3)
        else:
            audience_fit_score = 0.7  # Default
        
        return {
            'style_score': min(1.0, (audience_fit_score + sentence_variety_score) / 2),
            'passive_voice_ratio': round(passive_voice_ratio, 3),
            'professional_tone_score': round(professional_score, 3),
            'engagement_score': round(engagement_score, 3),
            'sentence_variety_score': round(sentence_variety_score, 3),
            'audience_fit_score': round(audience_fit_score, 3),
            'technical_terms_count': len(set(technical_terms)),
            'technical_terms': list(set(technical_terms))[:10],  # Top 10 unique terms
            'sentence_length_stats': {
                'avg': round(np.mean(sentence_lengths), 1) if sentence_lengths else 0,
                'min': min(sentence_lengths) if sentence_lengths else 0,
                'max': max(sentence_lengths) if sentence_lengths else 0,
                'std_dev': round(np.std(sentence_lengths), 1) if len(sentence_lengths) > 1 else 0
            }
        }
    
    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze content structure and organization."""
        
        # Header analysis
        headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        header_levels = [len(h[0]) for h in headers]
        
        # List and formatting analysis
        bullet_lists = len(re.findall(r'^\s*[-*+]\s+', content, re.MULTILINE))
        numbered_lists = len(re.findall(r'^\s*\d+\.\s+', content, re.MULTILINE))
        code_blocks = len(re.findall(r'```[\s\S]*?```', content))
        
        # Paragraph analysis
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        paragraph_lengths = [len(p.split()) for p in paragraphs]
        
        # Link analysis
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        external_links = [link for link in links if 'http' in link[1]]
        
        # Calculate structure score
        structure_score = 0.6  # Base score
        
        # Good use of headers
        if len(headers) >= 3:
            structure_score += 0.15
        elif len(headers) >= 1:
            structure_score += 0.1
        
        # Proper header hierarchy
        if header_levels and max(header_levels) - min(header_levels) <= 2:
            structure_score += 0.1
        
        # Good use of lists
        if bullet_lists > 0 or numbered_lists > 0:
            structure_score += 0.1
        
        # Reasonable paragraph lengths
        if paragraph_lengths:
            avg_para_length = np.mean(paragraph_lengths)
            if 30 <= avg_para_length <= 100:  # Optimal paragraph length
                structure_score += 0.15
            elif 20 <= avg_para_length <= 120:
                structure_score += 0.1
        
        return {
            'structure_score': min(1.0, structure_score),
            'header_count': len(headers),
            'header_levels': header_levels,
            'has_proper_hierarchy': len(set(header_levels)) <= 3 if header_levels else False,
            'paragraph_count': len(paragraphs),
            'average_paragraph_length': round(np.mean(paragraph_lengths), 1) if paragraph_lengths else 0,
            'bullet_lists': bullet_lists,
            'numbered_lists': numbered_lists,
            'code_blocks': code_blocks,
            'external_links_count': len(external_links),
            'has_clear_structure': len(headers) >= 2 and len(paragraphs) >= 3
        }
    
    def _analyze_keywords(self, content: str) -> Dict[str, Any]:
        """Analyze keyword usage and density."""
        
        # Clean content for keyword analysis
        cleaned_content = re.sub(r'[^\w\s]', ' ', content.lower())
        words = cleaned_content.split()
        
        # Word frequency analysis
        word_freq = Counter(words)
        
        # Remove very common words and very short words
        meaningful_words = {
            word: count for word, count in word_freq.items() 
            if len(word) > 3 and count > 1
        }
        
        # Get top keywords
        top_keywords = sorted(meaningful_words.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Keyword density analysis
        total_words = len(words)
        keyword_densities = {
            word: round((count / total_words) * 100, 2) 
            for word, count in top_keywords[:10]
        }
        
        # TF-IDF analysis for important terms
        important_terms = []
        try:
            if len(content.split('.')) > 1:  # Need multiple sentences for TF-IDF
                sentences = [s.strip() for s in content.split('.') if s.strip()]
                if len(sentences) >= 2:
                    tfidf_matrix = self.tfidf.fit_transform(sentences)
                    feature_names = self.tfidf.get_feature_names_out()
                    
                    # Get average TF-IDF scores
                    mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                    top_indices = np.argsort(mean_scores)[-10:][::-1]
                    
                    important_terms = [
                        {'term': feature_names[i], 'score': round(mean_scores[i], 3)}
                        for i in top_indices if mean_scores[i] > 0
                    ]
        except Exception as e:
            logger.warning(f"TF-IDF analysis failed: {e}")
        
        return {
            'total_unique_words': len(set(words)),
            'top_keywords': top_keywords[:10],
            'keyword_densities': keyword_densities,
            'important_terms_tfidf': important_terms,
            'lexical_diversity': len(set(words)) / len(words) if words else 0
        }
    
    def _analyze_professional_tone(self, content: str) -> Dict[str, Any]:
        """Analyze professional tone and formality."""
        
        content_lower = content.lower()
        
        # Formality indicators
        formal_indicators = [
            'furthermore', 'moreover', 'consequently', 'therefore', 'thus',
            'accordingly', 'nevertheless', 'however', 'although', 'whereas'
        ]
        
        informal_indicators = [
            'gonna', 'wanna', 'yeah', 'ok', 'cool', 'awesome', 'super',
            'really really', 'lots of', 'a bunch of', 'totally'
        ]
        
        # Count indicators
        formal_count = sum(1 for indicator in formal_indicators if indicator in content_lower)
        informal_count = sum(1 for indicator in informal_indicators if indicator in content_lower)
        
        # Authority indicators
        authority_indicators = [
            'research shows', 'studies indicate', 'data suggests', 'evidence demonstrates',
            'analysis reveals', 'findings show', 'experts believe', 'according to'
        ]
        
        authority_count = sum(1 for indicator in authority_indicators if indicator in content_lower)
        
        # Personal pronoun analysis
        first_person = len(re.findall(r'\b(i|me|my|myself)\b', content_lower))
        second_person = len(re.findall(r'\b(you|your|yourself)\b', content_lower))
        third_person = len(re.findall(r'\b(he|she|it|they|them|their)\b', content_lower))
        
        word_count = len(content.split())
        
        # Calculate tone scores
        formality_score = min(1.0, (formal_count - informal_count + 5) / 10)  # Normalized
        authority_score = min(1.0, authority_count / max(1, word_count / 200))  # Per ~200 words
        
        # Professional tone composite score
        professional_tone_score = (
            formality_score * 0.4 +
            authority_score * 0.3 +
            min(1.0, first_person / max(1, word_count / 100)) * 0.3  # Some first person is good
        )
        
        return {
            'professional_tone_score': round(professional_tone_score, 3),
            'formality_score': round(formality_score, 3),
            'authority_score': round(authority_score, 3),
            'pronoun_analysis': {
                'first_person_count': first_person,
                'second_person_count': second_person,
                'third_person_count': third_person,
                'first_person_ratio': round(first_person / word_count * 100, 2) if word_count else 0
            },
            'formal_indicators_found': formal_count,
            'informal_indicators_found': informal_count,
            'authority_indicators_found': authority_count
        }
    
    def _calculate_overall_quality_score(
        self,
        readability: Dict[str, Any],
        style: Dict[str, Any],
        structure: Dict[str, Any],
        tone: Dict[str, Any]
    ) -> float:
        """Calculate overall content quality score."""
        
        # Weighted scoring for professional content
        weights = {
            'readability': 0.25,
            'style': 0.30,
            'structure': 0.25,
            'tone': 0.20
        }
        
        scores = {
            'readability': readability.get('readability_score', 0.5),
            'style': style.get('style_score', 0.5),
            'structure': structure.get('structure_score', 0.5),
            'tone': tone.get('professional_tone_score', 0.5)
        }
        
        overall_score = sum(scores[key] * weights[key] for key in weights.keys())
        
        return round(min(1.0, max(0.0, overall_score)), 3)
    
    def _generate_improvement_recommendations(
        self,
        readability: Dict[str, Any],
        style: Dict[str, Any],
        structure: Dict[str, Any],
        tone: Dict[str, Any]
    ) -> List[str]:
        """Generate specific recommendations for content improvement."""
        
        recommendations = []
        
        # Readability recommendations
        flesch_score = readability.get('flesch_reading_ease', 50)
        if flesch_score < 50:
            recommendations.append("Improve readability by shortening sentences and using simpler vocabulary")
        elif flesch_score > 80:
            recommendations.append("Consider adding more technical depth - content may be too simplified for professional audience")
        
        avg_sentence_length = style.get('sentence_length_stats', {}).get('avg', 0)
        if avg_sentence_length > 25:
            recommendations.append("Break up long sentences to improve clarity and flow")
        elif avg_sentence_length < 10:
            recommendations.append("Consider combining short sentences for better flow")
        
        # Style recommendations
        if style.get('style_score', 1.0) < 0.7:
            if style.get('audience_fit_score', 1.0) < 0.7:
                recommendations.append("Adjust tone and vocabulary to better match target audience")
            if style.get('sentence_variety_score', 1.0) < 0.7:
                recommendations.append("Vary sentence lengths and structures for better readability")
        
        # Structure recommendations
        if structure.get('header_count', 0) < 2:
            recommendations.append("Add section headers to improve content organization")
        elif structure.get('header_count', 0) > 8:
            recommendations.append("Consider consolidating sections - too many headers may fragment the content")
        
        if structure.get('paragraph_count', 0) > 20:
            recommendations.append("Consider breaking content into smaller, focused sections")
        
        avg_para_length = structure.get('average_paragraph_length', 0)
        if avg_para_length > 100:
            recommendations.append("Shorten paragraphs for better visual appeal and readability")
        elif avg_para_length < 20:
            recommendations.append("Consider expanding paragraphs with more detailed explanations")
        
        # Tone recommendations
        if tone.get('professional_tone_score', 1.0) < 0.7:
            formal_score = tone.get('formality_score', 0.5)
            if formal_score < 0.5:
                recommendations.append("Increase formality with more professional language and structure")
            
            authority_score = tone.get('authority_score', 0.5)
            if authority_score < 0.5:
                recommendations.append("Add more authoritative references and evidence-based statements")
        
        # Passive voice recommendation
        first_person_ratio = tone.get('pronoun_analysis', {}).get('first_person_ratio', 0)
        if first_person_ratio > 5:
            recommendations.append("Reduce first-person references for more objective tone")
        elif first_person_ratio < 1:
            recommendations.append("Consider adding some personal insights or examples")
        
        return recommendations if recommendations else ["Content quality appears good - no major improvements needed"]
    
    def analyze_content_gaps(self, content: str, expected_topics: List[str]) -> Dict[str, Any]:
        """Analyze gaps between actual content and expected topics."""
        
        content_lower = content.lower()
        
        # Check coverage of expected topics
        topic_coverage = {}
        for topic in expected_topics:
            topic_words = topic.lower().split()
            matches = sum(1 for word in topic_words if word in content_lower)
            coverage_ratio = matches / len(topic_words) if topic_words else 0
            topic_coverage[topic] = coverage_ratio
        
        # Identify missing topics
        missing_topics = [topic for topic, coverage in topic_coverage.items() if coverage < 0.5]
        well_covered_topics = [topic for topic, coverage in topic_coverage.items() if coverage >= 0.8]
        
        # Calculate overall coverage score
        avg_coverage = sum(topic_coverage.values()) / len(topic_coverage) if topic_coverage else 0
        
        return {
            'coverage_score': round(avg_coverage, 3),
            'topic_coverage': topic_coverage,
            'missing_topics': missing_topics,
            'well_covered_topics': well_covered_topics,
            'coverage_gaps': len(missing_topics),
            'recommendations': self._generate_gap_recommendations(missing_topics, well_covered_topics)
        }
    
    def _generate_gap_recommendations(
        self, 
        missing_topics: List[str], 
        covered_topics: List[str]
    ) -> List[str]:
        """Generate recommendations for addressing content gaps."""
        
        recommendations = []
        
        if missing_topics:
            if len(missing_topics) == 1:
                recommendations.append(f"Add content covering: {missing_topics[0]}")
            else:
                recommendations.append(f"Add content covering these missing topics: {', '.join(missing_topics[:3])}")
        
        if len(covered_topics) > len(missing_topics) * 2:
            recommendations.append("Content covers expected topics well - consider deepening analysis")
        
        if not missing_topics and covered_topics:
            recommendations.append("Topic coverage is comprehensive - content addresses all expected areas")
        
        return recommendations if recommendations else ["No significant content gaps identified"]
    
    def validate_citations(self, content: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate citation usage and coverage."""
        
        # Extract citation markers from content
        citation_pattern = r'\[(\d+)\]|\[Citation (\d+)\]|\[(\w+(?:\s+\w+)*)\]'
        citations_found = re.findall(citation_pattern, content)
        
        # Flatten citation matches
        citation_numbers = []
        citation_names = []
        
        for match in citations_found:
            if match[0]:  # Numbered citation [1]
                try:
                    citation_numbers.append(int(match[0]))
                except ValueError:
                    pass
            elif match[1]:  # Named citation [Citation 1]
                try:
                    citation_numbers.append(int(match[1]))
                except ValueError:
                    pass
            elif match[2]:  # Named citation [Source Name]
                citation_names.append(match[2])
        
        # Validate against available sources
        total_sources = len(sources)
        valid_numbered_citations = [n for n in citation_numbers if 1 <= n <= total_sources]
        invalid_citations = [n for n in citation_numbers if n > total_sources or n < 1]
        
        # Calculate citation coverage
        unique_citations = len(set(valid_numbered_citations + citation_names))
        citation_coverage = unique_citations / total_sources if total_sources > 0 else 0
        
        # Analyze citation distribution
        word_count = len(content.split())
        citations_per_100_words = (len(citation_numbers) / word_count * 100) if word_count > 0 else 0
        
        return {
            'citation_score': min(1.0, citation_coverage * 1.2),  # Boost score slightly
            'total_citations_found': len(citation_numbers) + len(citation_names),
            'valid_citations': len(valid_numbered_citations),
            'invalid_citations': invalid_citations,
            'named_citations': citation_names,
            'citation_coverage_ratio': round(citation_coverage, 3),
            'citations_per_100_words': round(citations_per_100_words, 2),
            'source_utilization': f"{unique_citations}/{total_sources} sources cited",
            'has_adequate_citations': citations_per_100_words >= 0.5 and citation_coverage >= 0.3
        }