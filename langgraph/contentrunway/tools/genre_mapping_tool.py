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
        
        Analyze the content and determine if it fits into existing genres or needs a new genre.
        
        **Content Analysis:**
        - Title: {title}
        - Classification: {classification}
        - Domain: {domain}
        - Key Indicators: {key_indicators}
        - Content Summary: {content_summary}
        
        **Available Genres:**
        {available_genres}
        
        **Instructions:**
        1. If content has a GOOD match (80%+ fit) with existing genres, select the best one
        2. If content has only POOR matches (<70% fit), suggest creating a new genre
        3. Be specific - prefer exact topic matches over broad categories
        4. For technical content like "Quantum Computing", "Blockchain", "Machine Learning" - these should be separate genres, not lumped into "AI"
        
        **Response Format (JSON only, no markdown):**
        For existing genre match:
        {{
            "selected_genre_id": <integer>,
            "selected_genre_name": "<name>", 
            "confidence_score": <0.0-1.0>,
            "reasoning": "<why this genre fits well>",
            "create_new_genre": false
        }}
        
        For new genre needed:
        {{
            "selected_genre_id": null,
            "selected_genre_name": null,
            "confidence_score": 0.0,
            "reasoning": "<why existing genres don't fit>",
            "create_new_genre": true,
            "suggested_new_genre": "<specific genre name>",
            "new_genre_description": "<clear description of what this genre covers>"
        }}
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
                    {"role": "system", "content": "You are an expert content curator. Respond ONLY with valid JSON. No markdown, no explanations, just JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            # Parse response
            response_content = response.choices[0].message.content.strip()
            self.logger.log_info("map_content_to_genre", f"LLM Response: {response_content}")
            
            mapping_result = None
            
            # Try multiple parsing strategies
            try:
                # Strategy 1: Direct JSON parsing
                mapping_result = json.loads(response_content)
            except json.JSONDecodeError:
                try:
                    # Strategy 2: Extract from markdown code blocks
                    import re
                    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_content, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1).strip()
                        mapping_result = json.loads(json_content)
                    else:
                        # Strategy 3: Find JSON object boundaries
                        start = response_content.find('{')
                        end = response_content.rfind('}') + 1
                        if start != -1 and end != 0:
                            json_content = response_content[start:end]
                            mapping_result = json.loads(json_content)
                        else:
                            raise json.JSONDecodeError("No JSON found", response_content, 0)
                except json.JSONDecodeError as e:
                    self.logger.log_error("map_content_to_genre", f"Failed to parse LLM response: {response_content}")
                    # Return fallback response instead of raising exception
                    mapping_result = {
                        "selected_genre_id": None,
                        "selected_genre_name": None,
                        "confidence_score": 0.0,
                        "reasoning": f"Failed to parse LLM response: {str(e)}",
                        "create_new_genre": True,
                        "suggested_new_genre": f"{classification_analysis.get('domain', 'General')} Content",
                        "new_genre_description": f"Content related to {classification_analysis.get('domain', 'general topics')}"
                    }
            
            # Process mapping result
            selected_genre_id = mapping_result.get("selected_genre_id")
            create_new_genre = mapping_result.get("create_new_genre", False)
            
            # Check if we should create a new genre
            if selected_genre_id is None or create_new_genre:
                # Generate new genre
                self.logger.log_info(
                    "map_content_to_genre",
                    "Creating new genre based on LLM recommendation",
                    operation_context
                )
                
                suggested_new_genre = mapping_result.get("suggested_new_genre")
                if not suggested_new_genre:
                    # Create a more specific genre name based on content
                    domain = classification_analysis.get('domain', 'General')
                    key_indicators = classification_analysis.get('key_indicators', [])
                    
                    # Generate smart genre name based on content
                    if 'quantum' in title.lower() or any('quantum' in k.lower() for k in key_indicators):
                        suggested_new_genre = "Quantum Computing"
                    elif 'blockchain' in title.lower() or any('blockchain' in k.lower() for k in key_indicators):
                        suggested_new_genre = "Blockchain Technology"
                    elif 'machine learning' in title.lower() or any('machine learning' in k.lower() for k in key_indicators):
                        suggested_new_genre = "Machine Learning"
                    elif domain == 'IT Insurance':
                        suggested_new_genre = "Insurance Technology"
                    elif domain == 'AI Research':
                        suggested_new_genre = "AI Research"
                    elif domain == 'Agentic AI':
                        suggested_new_genre = "Agentic AI"
                    else:
                        suggested_new_genre = f"{domain} Technology"
                
                new_genre_description = mapping_result.get("new_genre_description", 
                    f"Content covering {suggested_new_genre.lower()} topics and applications")
                
                # Generate deterministic genre ID (100-999 range for ContentRunway)
                import hashlib
                genre_name_hash = hashlib.md5(suggested_new_genre.encode()).hexdigest()
                new_genre_id = 100 + (int(genre_name_hash, 16) % 900)  # 100-999 range
                
                analysis = {
                    "selected_genre_id": new_genre_id,
                    "selected_genre_name": suggested_new_genre,
                    "selected_genre": suggested_new_genre,  # For API compatibility
                    "confidence_score": 0.9,  # High confidence for new genre creation
                    "reasoning": mapping_result.get("reasoning", f"Created new specific genre: {suggested_new_genre}"),
                    "domain_focus": classification_analysis.get("domain", "General"),
                    "requires_new_genre": True,
                    "model_used": "gpt-4",
                    "fallback_used": False,
                    "new_genre_metadata": {
                        "id": new_genre_id,
                        "name": suggested_new_genre,
                        "description": new_genre_description,
                        "domain": classification_analysis.get("domain", "General"),
                        "isAutoGenerated": True,
                        "createdBy": "ContentRunway-LLM"
                    }
                }
                
                return new_genre_id, analysis
            
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
                "selected_genre": mapping_result.get("selected_genre_name", "Unknown"),  # For API compatibility
                "confidence_score": float(mapping_result.get("confidence_score", 0.7)),
                "reasoning": mapping_result.get("reasoning", ""),
                "domain_focus": classification_analysis.get("domain", "General"),
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
            
            # Check if it's an API key issue
            if "api_key" in str(e).lower() or "unauthorized" in str(e).lower():
                self.logger.log_error("map_content_to_genre", "OpenAI API key missing or invalid. Please set OPENAI_API_KEY environment variable.")
            
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