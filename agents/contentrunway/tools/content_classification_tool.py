"""Content Classification Tool - OpenAI-based content analysis for Blog vs Product classification."""

import openai
import os
from typing import Dict, Any, Tuple
import json
from ..utils.publisher_logger import PublisherLogger

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class ContentClassificationTool:
    """Tool for classifying content as Blog or Product using OpenAI."""
    
    def __init__(self):
        self.logger = PublisherLogger()
        
        # Get API key with proper error handling
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable in your .env file."
            )
        
        self.client = openai.AsyncOpenAI(api_key=api_key)
        
        # Classification prompts
        self.classification_prompt = """You are an expert content classifier for a digital publishing platform. 

Analyze the following content and classify it as either "Blog" or "Product" based on these criteria:

**Product Classification:**
- Content that describes, reviews, or discusses specific products, platforms, services, or tools
- Content that includes product specifications, features, or comparisons
- Content that provides tutorials or guides for using specific products
- Content that discusses product strategies, implementations, or use cases

**Blog Classification:**
- General informational or educational content
- Industry insights, trends, or analysis that don't focus on specific products
- Conceptual discussions about technologies, methodologies, or practices
- Opinion pieces, thought leadership, or general commentary
- Educational content about broad topics rather than specific products

**Content Domains to Consider:**
- IT Insurance (regulatory compliance, digital transformation, insurtech)
- AI Research (technical AI content, ML developments, LLM integrations)
- Agentic AI (multi-agent systems, LangGraph, agent orchestration)

Respond with ONLY a valid JSON object in this exact format:
{{
    "classification": "Blog",
    "confidence_score": 0.85,
    "reasoning": "Brief explanation of classification decision",
    "key_indicators": ["testing", "best practices", "guide"],
    "domain": "Technical"
}}

Content to classify:

Title: {title}

Content: {content}"""
    
    async def classify_content(
        self,
        title: str,
        content: str,
        summary: str = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Classify content as Blog or Product.
        
        Args:
            title: Content title
            content: Main content text
            summary: Optional summary text
            
        Returns:
            Tuple of (classification, detailed_analysis)
        """
        
        operation_context = {
            "title": title,
            "content_length": len(content),
            "has_summary": bool(summary)
        }
        
        self.logger.log_operation_start("classify_content", operation_context)
        
        # CRITICAL: Validate input data
        if not content or len(content.strip()) < 100:
            error_msg = f"Classification failed: Content too short ({len(content)} chars). Content: '{content[:100]}'"
            self.logger.log_error("classify_content", error_msg)
            raise ValueError(error_msg)
        
        if not title or title.strip() == "":
            error_msg = f"Classification failed: Invalid title: '{title}'"
            self.logger.log_error("classify_content", error_msg)
            raise ValueError(error_msg)
        
        self.logger.log_info("classify_content", f"🔧 DEBUG: Starting classification - Title: '{title}', Content: {len(content)} chars")
        self.logger.log_info("classify_content", f"🔧 DEBUG: Content preview: {content[:200]}")
        
        try:
            # Prepare content for classification
            full_content = content
            if summary:
                full_content = f"Summary: {summary}\n\nContent: {content}"
            
            # Truncate content if too long (keep first 3000 chars to stay within token limits)
            if len(full_content) > 3000:
                full_content = full_content[:3000] + "..."
            
            # Create classification prompt
            prompt = self.classification_prompt.format(
                title=title,
                content=full_content
            )
            
            # Make OpenAI API call
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert content classifier. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse response
            response_content = response.choices[0].message.content.strip()
            
            try:
                classification_result = json.loads(response_content)
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
                
                classification_result = None
                for pattern in patterns:
                    json_match = re.search(pattern, response_content, re.DOTALL)
                    if json_match:
                        try:
                            extracted_json = json_match.group(1) if pattern.startswith('```') else json_match.group(0)
                            classification_result = json.loads(extracted_json)
                            break
                        except json.JSONDecodeError:
                            continue
                
                if classification_result is None:
                    self.logger.log_warning("classify_content", f"Failed to parse OpenAI response: {response_content[:200]}...")
                    raise Exception(f"Invalid JSON response from OpenAI: {parse_error}")
            
            # Validate classification result
            classification = classification_result.get("classification", "Blog")
            if classification not in ["Blog", "Product"]:
                self.logger.log_warning("classify_content", f"🔧 DEBUG: Invalid classification '{classification}', defaulting to Blog")
                classification = "Blog"  # Default fallback
            
            confidence_score = float(classification_result.get("confidence_score", 0.7))
            
            # CRITICAL: Ensure confidence is valid
            if confidence_score <= 0.0 or confidence_score > 1.0:
                self.logger.log_warning("classify_content", f"🔧 DEBUG: Invalid confidence score {confidence_score}, defaulting to 0.7")
                confidence_score = 0.7
            
            self.logger.log_info("classify_content", f"🔧 DEBUG: Classification successful - Type: '{classification}', Confidence: {confidence_score}")
            
            # Create detailed analysis
            analysis = {
                "classification": classification,
                "confidence_score": confidence_score,
                "reasoning": classification_result.get("reasoning", ""),
                "key_indicators": classification_result.get("key_indicators", []),
                "domain": classification_result.get("domain", "General"),
                "model_used": "gpt-4",
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }
            
            self.logger.log_operation_success(
                "classify_content",
                {
                    "classification": classification,
                    "confidence": confidence_score,
                    "domain": analysis["domain"]
                },
                operation_context
            )
            
            return classification, analysis
            
        except Exception as e:
            error_msg = f"Content classification failed: {e}"
            self.logger.log_operation_failure("classify_content", error_msg, operation_context)
            
            # Enhanced error logging
            import traceback
            self.logger.log_error("classify_content", f"🔧 DEBUG: Full traceback: {traceback.format_exc()}")
            self.logger.log_error("classify_content", f"🔧 DEBUG: Error type: {type(e)}")
            self.logger.log_error("classify_content", f"🔧 DEBUG: Error message: {str(e)}")
            self.logger.log_error("classify_content", f"🔧 DEBUG: Title: '{title}'")
            self.logger.log_error("classify_content", f"🔧 DEBUG: Content length: {len(content)}")
            
            # Check if it's an API key issue
            if "api_key" in str(e).lower() or "unauthorized" in str(e).lower():
                self.logger.log_error("classify_content", "🔧 CRITICAL: OpenAI API key missing or invalid!")
                raise ValueError(f"OpenAI API key issue: {e}")
            
            # Return fallback classification
            fallback_analysis = {
                "classification": "Blog",
                "confidence_score": 0.5,
                "reasoning": f"Classification failed, defaulting to Blog: {error_msg}",
                "key_indicators": [],
                "domain": "General",
                "model_used": "fallback",
                "error": error_msg,
                "fallback_used": True
            }
            
            self.logger.log_warning("classify_content", f"🔧 DEBUG: Using fallback classification: Blog with 0.5 confidence")
            
            return "Blog", fallback_analysis
    
    async def classify_from_pipeline_content(self, content_dict: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Classify content from pipeline content dictionary."""
        
        title = content_dict.get('title', 'Untitled')
        content = content_dict.get('content', content_dict.get('body', ''))
        summary = content_dict.get('summary', content_dict.get('excerpt', ''))
        
        return await self.classify_content(title, content, summary)
    
    def get_category_folder_mapping(self, classification: str) -> str:
        """Get folder name for cover image selection based on classification."""
        
        folder_mapping = {
            "Blog": "blog",
            "Product": "product"
        }
        
        return folder_mapping.get(classification, "blog")  # Default to blog
    
    async def batch_classify_content(self, content_items: list) -> list:
        """Classify multiple content items in batch."""
        
        results = []
        
        for item in content_items:
            try:
                if isinstance(item, dict):
                    classification, analysis = await self.classify_from_pipeline_content(item)
                else:
                    # Assume it's a tuple of (title, content)
                    title, content = item
                    classification, analysis = await self.classify_content(title, content)
                
                results.append({
                    "classification": classification,
                    "analysis": analysis
                })
                
            except Exception as e:
                self.logger.log_error(
                    "batch_classify_content",
                    f"Failed to classify item: {e}"
                )
                
                results.append({
                    "classification": "Blog",
                    "analysis": {
                        "classification": "Blog",
                        "confidence_score": 0.5,
                        "reasoning": f"Batch classification failed: {e}",
                        "error": str(e)
                    }
                })
        
        return results