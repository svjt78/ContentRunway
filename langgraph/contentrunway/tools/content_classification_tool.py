"""Content Classification Tool - OpenAI-based content analysis for Blog vs Product classification."""

import openai
import os
from typing import Dict, Any, Tuple
import json
from ..utils.publisher_logger import PublisherLogger


class ContentClassificationTool:
    """Tool for classifying content as Blog or Product using OpenAI."""
    
    def __init__(self):
        self.logger = PublisherLogger()
        self.client = openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Classification prompts
        self.classification_prompt = """
        You are an expert content classifier for a digital publishing platform. 
        
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
        
        Respond with a JSON object containing:
        {
            "classification": "Blog" or "Product",
            "confidence_score": 0.0-1.0,
            "reasoning": "Brief explanation of classification decision",
            "key_indicators": ["list", "of", "key", "words", "or", "phrases", "that", "influenced", "decision"],
            "domain": "Primary domain (IT Insurance, AI Research, Agentic AI, or General)"
        }
        
        Content to classify:
        
        Title: {title}
        
        Content: {content}
        """
    
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
            except json.JSONDecodeError:
                # Fallback parsing if JSON is wrapped in code blocks
                import re
                json_match = re.search(r'```json\n(.*?)\n```', response_content, re.DOTALL)
                if json_match:
                    classification_result = json.loads(json_match.group(1))
                else:
                    raise Exception("Invalid JSON response from OpenAI")
            
            # Validate classification result
            classification = classification_result.get("classification", "Blog")
            if classification not in ["Blog", "Product"]:
                classification = "Blog"  # Default fallback
            
            confidence_score = float(classification_result.get("confidence_score", 0.7))
            
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
            
            # Return fallback classification
            fallback_analysis = {
                "classification": "Blog",
                "confidence_score": 0.5,
                "reasoning": f"Classification failed, defaulting to Blog: {error_msg}",
                "key_indicators": [],
                "domain": "General",
                "model_used": "fallback",
                "error": error_msg
            }
            
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