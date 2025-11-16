"""Category Classification Agent - Classify content as Blog or Product using OpenAI."""

from typing import Dict, Any, Tuple
from ..tools.content_classification_tool import ContentClassificationTool
from ..utils.publisher_logger import PublisherLogger


class CategoryClassifierAgent:
    """Agent for classifying content as Blog or Product."""
    
    def __init__(self):
        self.logger = PublisherLogger()
        self.classification_tool = ContentClassificationTool()
    
    async def execute(
        self,
        content_dict: Dict[str, Any],
        state: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute category classification.
        
        Args:
            content_dict: Content dictionary with title, content, summary
            state: Optional pipeline state for additional context
            
        Returns:
            Classification results with analysis
        """
        
        operation_context = {
            "has_title": bool(content_dict.get('title')),
            "has_content": bool(content_dict.get('content')),
            "has_summary": bool(content_dict.get('summary'))
        }
        
        self.logger.log_operation_start("category_classification", operation_context)
        
        try:
            # Extract content components
            title = content_dict.get('title', 'Untitled')
            content = content_dict.get('content', content_dict.get('body', ''))
            summary = content_dict.get('summary', content_dict.get('excerpt', ''))
            
            # Perform classification
            classification, analysis = await self.classification_tool.classify_content(
                title=title,
                content=content,
                summary=summary
            )
            
            # Create comprehensive result
            result = {
                'classification': classification,
                'confidence_score': analysis['confidence_score'],
                'reasoning': analysis['reasoning'],
                'key_indicators': analysis['key_indicators'],
                'domain': analysis['domain'],
                'analysis': analysis,
                'input_title': title,
                'input_content_length': len(content),
                'agent': 'CategoryClassifierAgent'
            }
            
            self.logger.log_operation_success(
                "category_classification",
                {
                    "classification": classification,
                    "confidence": analysis['confidence_score'],
                    "domain": analysis['domain']
                },
                operation_context
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Category classification failed: {e}"
            self.logger.log_operation_failure("category_classification", error_msg, operation_context)
            
            # Return fallback result
            fallback_result = {
                'classification': 'Blog',
                'confidence_score': 0.5,
                'reasoning': f'Classification failed, defaulting to Blog: {error_msg}',
                'key_indicators': [],
                'domain': 'General',
                'analysis': {
                    'classification': 'Blog',
                    'confidence_score': 0.5,
                    'reasoning': f'Agent execution failed: {error_msg}',
                    'error': error_msg,
                    'fallback_used': True
                },
                'agent': 'CategoryClassifierAgent',
                'error': error_msg
            }
            
            return fallback_result
    
    async def classify_multiple_contents(
        self,
        content_list: list,
        state: Dict[str, Any] = None
    ) -> list:
        """
        Classify multiple content items.
        
        Args:
            content_list: List of content dictionaries
            state: Optional pipeline state
            
        Returns:
            List of classification results
        """
        
        self.logger.log_operation_start(
            "classify_multiple_contents",
            {"content_count": len(content_list)}
        )
        
        results = []
        
        for i, content_dict in enumerate(content_list):
            try:
                result = await self.execute(content_dict, state)
                result['batch_index'] = i
                results.append(result)
                
            except Exception as e:
                self.logger.log_error(
                    "classify_multiple_contents",
                    f"Failed to classify item {i}: {e}"
                )
                
                # Add error result
                results.append({
                    'classification': 'Blog',
                    'confidence_score': 0.5,
                    'reasoning': f'Batch classification failed: {e}',
                    'batch_index': i,
                    'error': str(e),
                    'agent': 'CategoryClassifierAgent'
                })
        
        self.logger.log_operation_success(
            "classify_multiple_contents",
            {"processed_count": len(results), "success_count": len([r for r in results if not r.get('error')])},
            {"content_count": len(content_list)}
        )
        
        return results
    
    def get_folder_mapping(self, classification: str) -> str:
        """Get folder name for cover image selection based on classification."""
        return self.classification_tool.get_category_folder_mapping(classification)
    
    def validate_classification_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and standardize classification result.
        
        Args:
            result: Classification result from execute()
            
        Returns:
            Validated result
        """
        
        # Ensure classification is valid
        if result.get('classification') not in ['Blog', 'Product']:
            result['classification'] = 'Blog'
            result['confidence_score'] = min(result.get('confidence_score', 0.5), 0.5)
            result['reasoning'] += ' (Invalid classification corrected to Blog)'
        
        # Ensure confidence score is valid
        confidence = result.get('confidence_score', 0.5)
        if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
            result['confidence_score'] = 0.5
            result['reasoning'] += ' (Invalid confidence score corrected)'
        
        # Ensure required fields exist
        result.setdefault('key_indicators', [])
        result.setdefault('domain', 'General')
        result.setdefault('reasoning', 'No reasoning provided')
        
        return result
    
    async def get_classification_summary(self, results: list) -> Dict[str, Any]:
        """
        Generate summary statistics for multiple classification results.
        
        Args:
            results: List of classification results
            
        Returns:
            Summary statistics
        """
        
        if not results:
            return {
                'total_items': 0,
                'blog_count': 0,
                'product_count': 0,
                'average_confidence': 0.0,
                'domains': {},
                'errors': 0
            }
        
        blog_count = sum(1 for r in results if r.get('classification') == 'Blog')
        product_count = sum(1 for r in results if r.get('classification') == 'Product')
        error_count = sum(1 for r in results if r.get('error'))
        
        # Calculate average confidence for successful classifications
        successful_results = [r for r in results if not r.get('error')]
        avg_confidence = 0.0
        if successful_results:
            avg_confidence = sum(r.get('confidence_score', 0) for r in successful_results) / len(successful_results)
        
        # Count domains
        domain_counts = {}
        for result in successful_results:
            domain = result.get('domain', 'General')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        summary = {
            'total_items': len(results),
            'blog_count': blog_count,
            'product_count': product_count,
            'error_count': error_count,
            'success_rate': (len(results) - error_count) / len(results) if results else 0,
            'average_confidence': round(avg_confidence, 3),
            'domains': domain_counts,
            'classification_distribution': {
                'Blog': blog_count,
                'Product': product_count
            }
        }
        
        return summary