"""Genre Generation Agent - Generate appropriate genres using LLM."""

import openai
import os
from typing import Dict, Any, List
import json
from ..utils.publisher_logger import PublisherLogger


class GenreGeneratorAgent:
    """Agent for generating appropriate genres for content using LLM."""
    
    def __init__(self):
        self.logger = PublisherLogger()
        self.client = openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Predefined genre examples for reference (IDs will be auto-generated)
        self.common_genres = [
            "AI Research",
            "Insurance Technology", 
            "Digital Transformation",
            "Cybersecurity",
            "Data Analytics",
            "Machine Learning",
            "Business Intelligence",
            "Technical Analysis",
            "Industry Reports",
            "Best Practices",
            "Case Studies",
            "Product Reviews",
            "Implementation Guides",
            "Strategic Planning",
            "Risk Management",
            "Software Testing",
            "Quality Assurance",
            "DevOps Practices",
            "Cloud Computing",
            "Blockchain Technology"
        ]
        
        # Genre generation prompt
        self.genre_generation_prompt = """
        You are an expert content curator and taxonomist for a professional digital publishing platform.
        
        Analyze the following content and determine the most appropriate genre classification:
        
        **Content Information:**
        - Title: {title}
        - Classification: {classification}
        - Domain: {domain}
        - Content Summary: {content_summary}
        - Key Indicators: {key_indicators}
        
        **Common Genre Examples:**
        {common_genres}
        
        **Genre Selection Criteria:**
        1. Choose the most specific and accurate genre that matches the content's subject matter
        2. Consider the content's domain expertise level (IT Insurance, AI Research, Agentic AI)
        3. For technical content, prefer specific technical genres
        4. For business/strategy content, prefer business-focused genres
        5. For mixed content, choose the dominant theme
        6. You can use the common genres above as examples, or create a more specific genre if needed
        
        **Genre Requirements:**
        - Maximum 30 characters
        - Professional and descriptive
        - Industry-appropriate terminology
        - Suitable for {classification} category
        - Clear and specific (avoid overly broad terms like "General" or "Misc")
        
        Respond with JSON format:
        {
            "selected_genre": "<descriptive genre name>",
            "genre_id": <integer 1-999, use hash or deterministic assignment>,
            "confidence_score": <0.0-1.0>,
            "reasoning": "<detailed explanation of why this genre was selected>",
            "alternative_genres": ["<genre1>", "<genre2>", "<genre3>"],
            "is_custom_genre": <true if created new genre, false if using common example>,
            "domain_focus": "<primary domain focus: IT Insurance, AI Research, Agentic AI, or General>"
        }
        """
    
    async def execute(
        self,
        content_dict: Dict[str, Any],
        classification: str,
        classification_analysis: Dict[str, Any],
        state: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute genre generation.
        
        Args:
            content_dict: Content dictionary with title, content, summary
            classification: Blog or Product classification
            classification_analysis: Analysis from category classification
            state: Optional pipeline state
            
        Returns:
            Genre generation results with selected genre
        """
        
        operation_context = {
            "title": content_dict.get('title', 'Untitled'),
            "classification": classification,
            "domain": classification_analysis.get('domain', 'General'),
            "content_length": len(content_dict.get('content', ''))
        }
        
        self.logger.log_operation_start("genre_generation", operation_context)
        
        try:
            # Extract content information
            title = content_dict.get('title', 'Untitled')
            content = content_dict.get('content', content_dict.get('body', ''))
            summary = content_dict.get('summary', content_dict.get('excerpt', ''))
            
            # Create content summary for prompt
            content_summary = summary if summary else content[:500] + "..." if len(content) > 500 else content
            
            # Generate genre
            genre_result = await self._generate_genre(
                title=title,
                classification=classification,
                domain=classification_analysis.get('domain', 'General'),
                content_summary=content_summary,
                key_indicators=classification_analysis.get('key_indicators', [])
            )
            
            # Create comprehensive results
            results = {
                'selected_genre': genre_result['selected_genre'],
                'genre_id': genre_result['genre_id'],
                'confidence_score': genre_result['confidence_score'],
                'reasoning': genre_result['reasoning'],
                'alternative_genres': genre_result.get('alternative_genres', []),
                'is_custom_genre': genre_result.get('is_custom_genre', False),
                'domain_focus': genre_result.get('domain_focus', 'General'),
                'generation_details': {
                    'original_title': title,
                    'classification': classification,
                    'domain': classification_analysis.get('domain', 'General'),
                    'model_used': 'gpt-4',
                    'tokens_used': genre_result.get('tokens_used', 0)
                }
            }
            
            self.logger.log_operation_success(
                "genre_generation",
                {
                    "selected_genre": results['selected_genre'],
                    "genre_id": results['genre_id'],
                    "confidence_score": results['confidence_score'],
                    "is_custom": results['is_custom_genre']
                },
                operation_context
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Genre generation failed: {e}"
            self.logger.log_operation_failure("genre_generation", error_msg, operation_context)
            
            # Check if it's an API key issue
            if "api_key" in str(e).lower() or "unauthorized" in str(e).lower():
                self.logger.log_error("genre_generation", "OpenAI API key missing or invalid. Please set OPENAI_API_KEY environment variable.")
            
            # Return fallback results
            fallback_genre = self._get_fallback_genre(classification, classification_analysis.get('domain', 'General'))
            
            fallback_results = {
                'selected_genre': fallback_genre['name'],
                'genre_id': fallback_genre['id'],
                'confidence_score': 0.5,
                'reasoning': f"Genre generation failed, using fallback: {error_msg}",
                'alternative_genres': [],
                'is_custom_genre': False,
                'domain_focus': classification_analysis.get('domain', 'General'),
                'generation_details': {
                    'original_title': content_dict.get('title', 'Untitled'),
                    'classification': classification,
                    'domain': classification_analysis.get('domain', 'General'),
                    'model_used': 'fallback',
                    'error': error_msg,
                    'fallback_used': True
                }
            }
            
            return fallback_results
    
    async def _generate_genre(
        self,
        title: str,
        classification: str,
        domain: str,
        content_summary: str,
        key_indicators: List[str]
    ) -> Dict[str, Any]:
        """Generate genre using OpenAI API."""
        
        # Format common genres for prompt
        common_genres_text = "\n".join([f"- {genre}" for genre in self.common_genres])
        
        # Create prompt
        prompt = self.genre_generation_prompt.format(
            title=title,
            classification=classification,
            domain=domain,
            content_summary=content_summary,
            key_indicators=", ".join(key_indicators) if key_indicators else "None specified",
            common_genres=common_genres_text
        )
        
        # Make API call
        response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert content curator and taxonomist. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=400
        )
        
        # Parse response
        response_content = response.choices[0].message.content.strip()
        
        try:
            genre_result = json.loads(response_content)
        except json.JSONDecodeError:
            # Fallback parsing for markdown-wrapped JSON
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response_content, re.DOTALL)
            if json_match:
                genre_result = json.loads(json_match.group(1))
            else:
                # Extract key information manually as last resort
                raise Exception("Invalid JSON response from OpenAI")
        
        # Add token usage information
        if response.usage:
            genre_result['tokens_used'] = response.usage.total_tokens
        
        # Validate and clean up the response
        selected_genre = genre_result.get('selected_genre', 'Technical Analysis')
        
        # Generate deterministic ID if not provided or invalid
        if not genre_result.get('genre_id') or not isinstance(genre_result.get('genre_id'), int):
            genre_result['genre_id'] = self._generate_genre_id(selected_genre)
        
        genre_result.setdefault('selected_genre', selected_genre)
        genre_result.setdefault('confidence_score', 0.7)
        genre_result.setdefault('reasoning', 'Genre selected based on content analysis')
        genre_result.setdefault('alternative_genres', [])
        genre_result.setdefault('is_custom_genre', selected_genre not in self.common_genres)
        genre_result.setdefault('domain_focus', 'General')
        
        return genre_result
    
    def _generate_genre_id(self, genre_name: str) -> int:
        """Generate deterministic ID from genre name for consistency."""
        # Use hash function to generate consistent ID from genre name
        import hashlib
        
        # Create hash and convert to deterministic integer in range 1-999
        hash_object = hashlib.md5(genre_name.encode())
        hash_hex = hash_object.hexdigest()
        hash_int = int(hash_hex, 16)
        
        # Map to range 100-999 to avoid conflicts with low IDs
        genre_id = (hash_int % 900) + 100
        
        return genre_id
    
    def _get_fallback_genre(self, classification: str, domain: str) -> Dict[str, Any]:
        """Get fallback genre based on classification and domain."""
        
        # Simple mapping for fallback scenarios
        fallback_mapping = {
            ('Blog', 'IT Insurance'): {'name': 'Insurance Technology', 'id': 100},
            ('Blog', 'AI Research'): {'name': 'AI Research', 'id': 101},
            ('Blog', 'Agentic AI'): {'name': 'AI Research', 'id': 101},
            ('Product', 'IT Insurance'): {'name': 'Product Reviews', 'id': 102},
            ('Product', 'AI Research'): {'name': 'Technical Analysis', 'id': 103},
            ('Product', 'Agentic AI'): {'name': 'Technical Analysis', 'id': 103}
        }
        
        fallback_key = (classification, domain)
        if fallback_key in fallback_mapping:
            return fallback_mapping[fallback_key]
        
        # Default fallback with deterministic IDs
        if classification == 'Blog':
            return {'name': 'Technical Analysis', 'id': 104}
        else:
            return {'name': 'Product Reviews', 'id': 105}
    
    def get_common_genres(self) -> List[str]:
        """Get list of common genre options."""
        return self.common_genres.copy()
    
    async def suggest_genre_for_domain(self, domain: str) -> List[str]:
        """Suggest appropriate genres for a specific domain."""
        
        domain_genre_mapping = {
            'IT Insurance': [
                'Insurance Technology',
                'Digital Transformation', 
                'Risk Management',
                'Business Intelligence',
                'Industry Reports'
            ],
            'AI Research': [
                'AI Research',
                'Machine Learning',
                'Data Analytics',
                'Technical Analysis',
                'Implementation Guides'
            ],
            'Agentic AI': [
                'AI Research',
                'Machine Learning',
                'Technical Analysis',
                'Implementation Guides',
                'Best Practices'
            ]
        }
        
        return domain_genre_mapping.get(domain, [
            'Technical Analysis',
            'Best Practices', 
            'Case Studies',
            'Industry Reports'
        ])