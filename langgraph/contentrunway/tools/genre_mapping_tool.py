"""Genre Mapping Tool - Map content to appropriate DigitalDossier genres."""

import openai
import os
from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime, timedelta
from ..utils.publisher_logger import PublisherLogger


class GenreMappingTool:
    """Tool for mapping content to appropriate DigitalDossier genres."""
    
    def __init__(self, api_tool):
        self.logger = PublisherLogger()
        self.api_tool = api_tool
        self.client = openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Genre cache for performance
        self._genre_cache = None
        self._cache_timestamp = None
        self._cache_expiry_hours = 24
        
        # Genre mapping prompt
        self.genre_mapping_prompt = """
        You are an expert content curator for a digital publishing platform.
        
        Given the following content and available genres, select the BEST matching genre for this content.
        
        **Content Analysis:**
        - Title: {title}
        - Classification: {classification}
        - Domain: {domain}
        - Key Indicators: {key_indicators}
        - Content Summary: {content_summary}
        
        **Available Genres:**
        {available_genres}
        
        **Selection Criteria:**
        1. Choose the genre that best matches the content's subject matter
        2. Consider the content's domain (IT Insurance, AI Research, Agentic AI)
        3. For technical content, prefer specific technical genres over general ones
        4. For business/strategy content, prefer business-focused genres
        5. If no perfect match exists, choose the closest relevant genre
        
        Respond with a JSON object:
        {
            "selected_genre_id": <integer>,
            "selected_genre_name": "<name>",
            "confidence_score": <0.0-1.0>,
            "reasoning": "<explanation of why this genre was selected>",
            "alternatives": [<list of alternative genre IDs if applicable>]
        }
        
        If no existing genre is suitable, respond with:
        {
            "selected_genre_id": null,
            "selected_genre_name": null,
            "confidence_score": 0.0,
            "reasoning": "No suitable existing genre found",
            "suggested_new_genre": "<suggested name for new genre>",
            "new_genre_description": "<description for new genre>"
        }
        """
    
    async def get_genres_with_cache(self) -> List[Dict[str, Any]]:
        """Get genres from API with caching."""
        
        # Check cache validity
        if (self._genre_cache is not None and 
            self._cache_timestamp is not None and
            datetime.now() - self._cache_timestamp < timedelta(hours=self._cache_expiry_hours)):
            
            self.logger.log_info("get_genres", "Using cached genres")
            return self._genre_cache
        
        # Fetch fresh genres
        try:
            genres = await self.api_tool.fetch_genres()
            
            # Update cache
            self._genre_cache = genres
            self._cache_timestamp = datetime.now()
            
            self.logger.log_info(
                "get_genres",
                f"Fetched and cached {len(genres)} genres"
            )
            
            return genres
            
        except Exception as e:
            error_msg = f"Failed to fetch genres: {e}"
            self.logger.log_error("get_genres", error_msg)
            
            # Return cached genres if available, even if expired
            if self._genre_cache is not None:
                self.logger.log_warning(
                    "get_genres",
                    "Using expired cache due to fetch failure"
                )
                return self._genre_cache
            
            raise Exception(error_msg)
    
    async def map_content_to_genre(
        self,
        title: str,
        classification: str,
        classification_analysis: Dict[str, Any],
        content_summary: str = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Map content to the best matching genre.
        
        Args:
            title: Content title
            classification: Blog or Product
            classification_analysis: Analysis from content classification
            content_summary: Optional content summary
            
        Returns:
            Tuple of (genre_id, mapping_analysis)
        """
        
        operation_context = {
            "title": title,
            "classification": classification,
            "domain": classification_analysis.get("domain", "General")
        }
        
        self.logger.log_operation_start("map_content_to_genre", operation_context)
        
        try:
            # Get available genres
            genres = await self.get_genres_with_cache()
            
            if not genres:
                raise Exception("No genres available from API")
            
            # Format genres for prompt
            genre_list = []
            for genre in genres:
                genre_info = f"ID: {genre['id']}, Name: {genre['name']}"
                if 'description' in genre:
                    genre_info += f", Description: {genre['description']}"
                genre_list.append(genre_info)
            
            available_genres_text = "\n".join(genre_list)
            
            # Prepare content summary
            if not content_summary:
                content_summary = f"Domain: {classification_analysis.get('domain', 'General')}"
            
            # Create mapping prompt
            prompt = self.genre_mapping_prompt.format(
                title=title,
                classification=classification,
                domain=classification_analysis.get("domain", "General"),
                key_indicators=", ".join(classification_analysis.get("key_indicators", [])),
                content_summary=content_summary[:500] if content_summary else "No summary available",
                available_genres=available_genres_text
            )
            
            # Make OpenAI API call
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert content curator. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=400
            )
            
            # Parse response
            response_content = response.choices[0].message.content.strip()
            
            try:
                mapping_result = json.loads(response_content)
            except json.JSONDecodeError:
                # Fallback parsing
                import re
                json_match = re.search(r'```json\n(.*?)\n```', response_content, re.DOTALL)
                if json_match:
                    mapping_result = json.loads(json_match.group(1))
                else:
                    raise Exception("Invalid JSON response from OpenAI")
            
            # Process mapping result
            selected_genre_id = mapping_result.get("selected_genre_id")
            
            if selected_genre_id is None:
                # No suitable genre found, need to create new one
                self.logger.log_warning(
                    "map_content_to_genre",
                    "No suitable existing genre found",
                    operation_context
                )
                
                # Use first genre as fallback
                fallback_genre_id = genres[0]['id'] if genres else 1
                
                analysis = {
                    "selected_genre_id": fallback_genre_id,
                    "selected_genre_name": genres[0]['name'] if genres else "General",
                    "confidence_score": 0.3,
                    "reasoning": "No suitable genre found, using fallback",
                    "suggested_new_genre": mapping_result.get("suggested_new_genre", ""),
                    "new_genre_description": mapping_result.get("new_genre_description", ""),
                    "requires_new_genre": True,
                    "model_used": "gpt-4",
                    "fallback_used": True
                }
                
                return fallback_genre_id, analysis
            
            # Validate selected genre ID exists
            valid_genre_ids = [g['id'] for g in genres]
            if selected_genre_id not in valid_genre_ids:
                # Use first genre as fallback
                selected_genre_id = genres[0]['id']
                mapping_result["selected_genre_name"] = genres[0]['name']
                mapping_result["reasoning"] += " (Invalid genre ID, using fallback)"
                mapping_result["fallback_used"] = True
            
            # Create analysis
            analysis = {
                "selected_genre_id": selected_genre_id,
                "selected_genre_name": mapping_result.get("selected_genre_name", "Unknown"),
                "confidence_score": float(mapping_result.get("confidence_score", 0.7)),
                "reasoning": mapping_result.get("reasoning", ""),
                "alternatives": mapping_result.get("alternatives", []),
                "model_used": "gpt-4",
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "requires_new_genre": False,
                "fallback_used": mapping_result.get("fallback_used", False)
            }
            
            self.logger.log_operation_success(
                "map_content_to_genre",
                {
                    "genre_id": selected_genre_id,
                    "genre_name": analysis["selected_genre_name"],
                    "confidence": analysis["confidence_score"]
                },
                operation_context
            )
            
            return selected_genre_id, analysis
            
        except Exception as e:
            error_msg = f"Genre mapping failed: {e}"
            self.logger.log_operation_failure("map_content_to_genre", error_msg, operation_context)
            
            # Return fallback genre
            try:
                genres = await self.get_genres_with_cache()
                fallback_genre_id = genres[0]['id'] if genres else 1
                fallback_genre_name = genres[0]['name'] if genres else "General"
            except:
                fallback_genre_id = 1
                fallback_genre_name = "General"
            
            fallback_analysis = {
                "selected_genre_id": fallback_genre_id,
                "selected_genre_name": fallback_genre_name,
                "confidence_score": 0.5,
                "reasoning": f"Genre mapping failed, using fallback: {error_msg}",
                "model_used": "fallback",
                "error": error_msg,
                "fallback_used": True
            }
            
            return fallback_genre_id, fallback_analysis
    
    async def find_genre_by_name(self, genre_name: str) -> Optional[Dict[str, Any]]:
        """Find genre by name (case-insensitive)."""
        
        try:
            genres = await self.get_genres_with_cache()
            
            for genre in genres:
                if genre['name'].lower() == genre_name.lower():
                    return genre
            
            return None
            
        except Exception as e:
            self.logger.log_error("find_genre_by_name", f"Failed to find genre: {e}")
            return None
    
    async def suggest_new_genre(
        self,
        title: str,
        classification: str,
        classification_analysis: Dict[str, Any],
        existing_genres: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Suggest a new genre based on content analysis."""
        
        operation_context = {
            "title": title,
            "classification": classification,
            "existing_genres_count": len(existing_genres)
        }
        
        self.logger.log_operation_start("suggest_new_genre", operation_context)
        
        try:
            # Create suggestion prompt
            existing_genre_names = [g['name'] for g in existing_genres]
            
            prompt = f"""
            Based on this content, suggest a new genre that would be appropriate but doesn't exist in the current genres.
            
            Content:
            - Title: {title}
            - Classification: {classification}
            - Domain: {classification_analysis.get('domain', 'General')}
            - Key Indicators: {', '.join(classification_analysis.get('key_indicators', []))}
            
            Existing Genres: {', '.join(existing_genre_names)}
            
            Respond with JSON:
            {{
                "suggested_name": "<genre name>",
                "description": "<description>",
                "reasoning": "<why this genre is needed>"
            }}
            """
            
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a content categorization expert. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            response_content = response.choices[0].message.content.strip()
            
            try:
                suggestion = json.loads(response_content)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'```json\n(.*?)\n```', response_content, re.DOTALL)
                if json_match:
                    suggestion = json.loads(json_match.group(1))
                else:
                    raise Exception("Invalid JSON response from OpenAI")
            
            self.logger.log_operation_success(
                "suggest_new_genre",
                {"suggested_name": suggestion.get("suggested_name", "")},
                operation_context
            )
            
            return suggestion
            
        except Exception as e:
            error_msg = f"New genre suggestion failed: {e}"
            self.logger.log_operation_failure("suggest_new_genre", error_msg, operation_context)
            
            # Return fallback suggestion
            return {
                "suggested_name": f"{classification_analysis.get('domain', 'General')} Content",
                "description": f"Content related to {classification_analysis.get('domain', 'general topics')}",
                "reasoning": f"Fallback suggestion due to error: {error_msg}",
                "error": error_msg
            }
    
    def clear_cache(self):
        """Clear the genre cache."""
        self._genre_cache = None
        self._cache_timestamp = None
        self.logger.log_info("clear_cache", "Genre cache cleared")