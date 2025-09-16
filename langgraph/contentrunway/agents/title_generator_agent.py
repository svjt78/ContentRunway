"""Title Generation Agent - Generate optimized titles using OpenAI."""

import openai
import os
from typing import Dict, Any, List, Tuple
import json
from ..utils.publisher_logger import PublisherLogger


class TitleGeneratorAgent:
    """Agent for generating optimized titles for content."""
    
    def __init__(self):
        self.logger = PublisherLogger()
        self.client = openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Title generation prompt
        self.title_generation_prompt = """You are an expert content marketing specialist and SEO copywriter. 

Generate 4 compelling, optimized titles for the following content:

**Content Information:**
- Original Title: {original_title}
- Classification: {classification}
- Domain: {domain}
- Content Summary: {content_summary}
- Key Indicators: {key_indicators}

**Title Requirements:**
- Maximum 60 characters for SEO optimization
- Clear, engaging, and descriptive
- Include relevant keywords naturally
- Suitable for {classification} category
- Professional tone appropriate for {domain} domain
- Compelling enough to drive clicks

**Title Types to Generate:**
1. SEO-Optimized: Focus on search engine optimization and keyword inclusion
2. Engagement-Focused: Emphasize curiosity and click-through appeal
3. Descriptive: Clear, straightforward description of content
4. Benefits-Driven: Highlight value proposition and benefits to reader

Respond with ONLY valid JSON in this exact format:
{{
    "titles": [
        {{
            "title": "Complete Testing Guide: Best Practices",
            "type": "seo_optimized",
            "character_count": 36,
            "score": 0.9,
            "reasoning": "SEO-friendly with key terms"
        }},
        {{
            "title": "Testing Secrets Every Developer Needs",
            "type": "engagement_focused", 
            "character_count": 34,
            "score": 0.8,
            "reasoning": "Creates curiosity and targets audience"
        }}
    ],
    "recommended_title": {{
        "title": "Complete Testing Guide: Best Practices",
        "type": "seo_optimized",
        "overall_score": 0.9,
        "reasoning": "Best balance of SEO and readability"
    }}
}}"""
    
    async def execute(
        self,
        content_dict: Dict[str, Any],
        classification: str,
        classification_analysis: Dict[str, Any],
        state: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute title generation.
        
        Args:
            content_dict: Content dictionary with title, content, summary
            classification: Blog or Product classification
            classification_analysis: Analysis from category classification
            state: Optional pipeline state
            
        Returns:
            Title generation results with recommendations
        """
        
        operation_context = {
            "original_title": content_dict.get('title', 'Untitled'),
            "classification": classification,
            "domain": classification_analysis.get('domain', 'General'),
            "content_length": len(content_dict.get('content', ''))
        }
        
        self.logger.log_operation_start("title_generation", operation_context)
        
        try:
            # Extract content information
            original_title = content_dict.get('title', 'Untitled')
            content = content_dict.get('content', content_dict.get('body', ''))
            summary = content_dict.get('summary', content_dict.get('excerpt', ''))
            
            # Create content summary for prompt
            content_summary = summary if summary else content[:500] + "..." if len(content) > 500 else content
            
            # Generate titles
            titles_result = await self._generate_titles(
                original_title=original_title,
                classification=classification,
                domain=classification_analysis.get('domain', 'General'),
                content_summary=content_summary,
                key_indicators=classification_analysis.get('key_indicators', [])
            )
            
            # Create comprehensive result
            result = {
                'original_title': original_title,
                'generated_titles': titles_result['titles'],
                'recommended_title': titles_result['recommended_title']['title'],
                'recommended_reasoning': titles_result['recommended_title']['reasoning'],
                'recommended_score': titles_result['recommended_title']['overall_score'],
                'classification': classification,
                'domain': classification_analysis.get('domain', 'General'),
                'generation_details': titles_result,
                'agent': 'TitleGeneratorAgent'
            }
            
            self.logger.log_operation_success(
                "title_generation",
                {
                    "original_title": original_title,
                    "recommended_title": result['recommended_title'],
                    "titles_generated": len(titles_result['titles']),
                    "recommended_score": result['recommended_score']
                },
                operation_context
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Title generation failed: {e}"
            self.logger.log_operation_failure("title_generation", error_msg, operation_context)
            
            # Return fallback result with enhanced original title
            fallback_title = self._enhance_original_title(
                content_dict.get('title', 'Untitled'),
                classification,
                classification_analysis
            )
            
            fallback_result = {
                'original_title': content_dict.get('title', 'Untitled'),
                'generated_titles': [],
                'recommended_title': fallback_title,
                'recommended_reasoning': f'Title generation failed, using enhanced original: {error_msg}',
                'recommended_score': 0.6,
                'classification': classification,
                'domain': classification_analysis.get('domain', 'General'),
                'agent': 'TitleGeneratorAgent',
                'error': error_msg,
                'fallback_used': True
            }
            
            return fallback_result
    
    async def _generate_titles(
        self,
        original_title: str,
        classification: str,
        domain: str,
        content_summary: str,
        key_indicators: List[str]
    ) -> Dict[str, Any]:
        """Generate 4 optimized titles using OpenAI."""
        
        try:
            # Create generation prompt
            prompt = self.title_generation_prompt.format(
                original_title=original_title,
                classification=classification,
                domain=domain,
                content_summary=content_summary,
                key_indicators=", ".join(key_indicators) if key_indicators else "None specified"
            )
            
            # Make OpenAI API call
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert SEO copywriter and content marketing specialist. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Higher temperature for creativity
                max_tokens=800
            )
            
            # Parse response
            response_content = response.choices[0].message.content.strip()
            
            try:
                titles_result = json.loads(response_content)
            except json.JSONDecodeError as parse_error:
                # Enhanced fallback parsing
                import re
                
                # Try multiple JSON extraction patterns
                patterns = [
                    r'```json\n(.*?)\n```',
                    r'```\n(.*?)\n```',
                    r'{.*}',
                    r'\{[\s\S]*\}'
                ]
                
                titles_result = None
                for pattern in patterns:
                    json_match = re.search(pattern, response_content, re.DOTALL)
                    if json_match:
                        try:
                            extracted_json = json_match.group(1) if pattern.startswith('```') else json_match.group(0)
                            titles_result = json.loads(extracted_json)
                            break
                        except json.JSONDecodeError:
                            continue
                
                if titles_result is None:
                    self.logger.log_warning("generate_titles", f"Failed to parse OpenAI response: {response_content[:200]}...")
                    raise Exception(f"Invalid JSON response from OpenAI: {parse_error}")
            
            # Validate and process titles
            processed_titles = self._process_and_validate_titles(
                titles_result.get('titles', []),
                original_title
            )
            
            # Select best title
            recommended = self._select_best_title(
                processed_titles,
                titles_result.get('recommended_title', {})
            )
            
            return {
                'titles': processed_titles,
                'recommended_title': recommended,
                'model_used': 'gpt-4',
                'tokens_used': response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            error_msg = f"OpenAI title generation failed: {e}"
            self.logger.log_error("generate_titles", error_msg)
            
            # Check if it's an API key issue
            if "api_key" in str(e).lower() or "unauthorized" in str(e).lower():
                self.logger.log_error("generate_titles", "OpenAI API key missing or invalid. Please set OPENAI_API_KEY environment variable.")
            
            raise Exception(error_msg)
    
    def _process_and_validate_titles(
        self,
        titles: List[Dict[str, Any]],
        original_title: str
    ) -> List[Dict[str, Any]]:
        """Process and validate generated titles."""
        
        processed_titles = []
        
        for i, title_data in enumerate(titles):
            try:
                title_text = title_data.get('title', f'Generated Title {i+1}')
                
                # Validate title length
                if len(title_text) > 80:  # Truncate very long titles
                    title_text = title_text[:77] + "..."
                
                processed_title = {
                    'title': title_text,
                    'type': title_data.get('type', 'generated'),
                    'character_count': len(title_text),
                    'score': self._calculate_title_score(title_text, original_title),
                    'reasoning': title_data.get('reasoning', 'Generated title'),
                    'seo_friendly': len(title_text) <= 60,
                    'index': i
                }
                
                processed_titles.append(processed_title)
                
            except Exception as e:
                self.logger.log_warning("process_titles", f"Error processing title {i}: {e}")
                continue
        
        # Ensure we have at least one title
        if not processed_titles:
            processed_titles.append({
                'title': original_title,
                'type': 'fallback',
                'character_count': len(original_title),
                'score': 0.6,
                'reasoning': 'Fallback to original title',
                'seo_friendly': len(original_title) <= 60,
                'index': 0
            })
        
        return processed_titles
    
    def _calculate_title_score(self, title: str, original_title: str) -> float:
        """Calculate quality score for a title."""
        
        score = 0.5  # Base score
        
        # Length scoring (ideal: 30-60 characters)
        length = len(title)
        if 30 <= length <= 60:
            score += 0.2
        elif 20 <= length <= 70:
            score += 0.1
        
        # Keyword presence (simple check)
        title_lower = title.lower()
        original_lower = original_title.lower()
        
        # Check for common important words
        original_words = set(original_lower.split())
        title_words = set(title_lower.split())
        
        # Keyword overlap bonus
        overlap = len(original_words.intersection(title_words))
        if overlap > 0:
            score += min(0.2, overlap * 0.05)
        
        # Avoid common weak words at the beginning
        weak_starters = ['the', 'a', 'an', 'how', 'why', 'what', 'when', 'where']
        if not any(title_lower.startswith(word + ' ') for word in weak_starters):
            score += 0.1
        
        # Cap score at 1.0
        return min(score, 1.0)
    
    def _select_best_title(
        self,
        processed_titles: List[Dict[str, Any]],
        recommended_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Select the best title from generated options."""
        
        if not processed_titles:
            return {
                'title': 'Untitled',
                'type': 'fallback',
                'overall_score': 0.5,
                'reasoning': 'No titles generated'
            }
        
        # If we have a specific recommendation, try to find it
        if recommended_data.get('title'):
            recommended_title = recommended_data['title']
            for title_data in processed_titles:
                if title_data['title'] == recommended_title:
                    return {
                        'title': recommended_title,
                        'type': title_data.get('type', 'recommended'),
                        'overall_score': recommended_data.get('overall_score', title_data['score']),
                        'reasoning': recommended_data.get('reasoning', 'AI recommended choice')
                    }
        
        # Otherwise, select highest scoring title
        best_title = max(processed_titles, key=lambda t: t['score'])
        
        return {
            'title': best_title['title'],
            'type': best_title['type'],
            'overall_score': best_title['score'],
            'reasoning': f"Highest scoring title: {best_title['reasoning']}"
        }
    
    def _enhance_original_title(
        self,
        original_title: str,
        classification: str,
        classification_analysis: Dict[str, Any]
    ) -> str:
        """Enhance original title as fallback when generation fails."""
        
        title = original_title.strip()
        
        # Add domain context if missing
        domain = classification_analysis.get('domain', '')
        if domain and domain.lower() not in title.lower():
            # Only add if title is not too long
            if len(title) < 50:
                if classification == 'Product':
                    title = f"{title} - {domain} Solution"
                else:
                    title = f"{title}: {domain} Insights"
        
        # Ensure it's not too long
        if len(title) > 70:
            title = title[:67] + "..."
        
        return title
    
    async def generate_multiple_titles(
        self,
        content_list: List[Dict[str, Any]],
        classifications: List[str],
        classification_analyses: List[Dict[str, Any]],
        state: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Generate titles for multiple content items."""
        
        self.logger.log_operation_start(
            "generate_multiple_titles",
            {"content_count": len(content_list)}
        )
        
        results = []
        
        for i, (content_dict, classification, analysis) in enumerate(zip(content_list, classifications, classification_analyses)):
            try:
                result = await self.execute(content_dict, classification, analysis, state)
                result['batch_index'] = i
                results.append(result)
                
            except Exception as e:
                self.logger.log_error(
                    "generate_multiple_titles",
                    f"Failed to generate title for item {i}: {e}"
                )
                
                # Add fallback result
                results.append({
                    'original_title': content_dict.get('title', 'Untitled'),
                    'recommended_title': content_dict.get('title', 'Untitled'),
                    'recommended_reasoning': f'Batch generation failed: {e}',
                    'recommended_score': 0.5,
                    'batch_index': i,
                    'error': str(e),
                    'agent': 'TitleGeneratorAgent'
                })
        
        self.logger.log_operation_success(
            "generate_multiple_titles",
            {"processed_count": len(results), "success_count": len([r for r in results if not r.get('error')])},
            {"content_count": len(content_list)}
        )
        
        return results
    
    def get_title_analytics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get analytics for a title generation result."""
        
        original = result.get('original_title', '')
        recommended = result.get('recommended_title', '')
        
        analytics = {
            'original_length': len(original),
            'recommended_length': len(recommended),
            'length_optimized': len(recommended) <= 60,
            'title_changed': original.lower() != recommended.lower(),
            'improvement_score': result.get('recommended_score', 0.5),
            'generation_successful': not result.get('error'),
            'titles_generated': len(result.get('generated_titles', [])),
            'seo_friendly': len(recommended) <= 60,
            'character_savings': len(original) - len(recommended) if len(original) > len(recommended) else 0
        }
        
        return analytics