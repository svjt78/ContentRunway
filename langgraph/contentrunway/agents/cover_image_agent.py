"""Cover Image Agent - Select and process cover images for content."""

from typing import Dict, Any
from ..tools.cover_image_processor_tool import CoverImageProcessorTool
from ..tools.dalle_image_generator_tool import DalleImageGeneratorTool
from ..utils.publisher_logger import PublisherLogger


class CoverImageAgent:
    """Agent for selecting and processing cover images."""
    
    def __init__(self):
        self.logger = PublisherLogger()
        self.image_processor = CoverImageProcessorTool()
        self.dalle_generator = DalleImageGeneratorTool()
    
    async def execute(
        self,
        content_dict: Dict[str, Any],
        classification: str,
        classification_analysis: Dict[str, Any],
        title: str = None,
        state: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute cover image selection and processing.
        
        Args:
            content_dict: Content dictionary with title, content, summary
            classification: Blog or Product classification
            classification_analysis: Analysis from category classification
            title: Optional override title (from title generation)
            state: Optional pipeline state
            
        Returns:
            Cover image processing results
        """
        
        # Use provided title or extract from content
        final_title = title or content_dict.get('title', 'Untitled')
        
        operation_context = {
            "classification": classification,
            "title": final_title,
            "domain": classification_analysis.get('domain', 'General'),
            "has_key_indicators": bool(classification_analysis.get('key_indicators'))
        }
        
        self.logger.log_operation_start("cover_image_selection", operation_context)
        
        try:
            # Select and process cover image with title replacement (conservative approach)
            image_result = await self.image_processor.select_and_process_image(
                category=classification,
                content_analysis=classification_analysis,
                content_title=content_dict.get('title', 'Untitled'),  # Use original title for selection
                remove_text=False,  # Use conservative overlay approach instead of aggressive text removal
                replacement_title=final_title  # Use generated title for replacement
            )
            
            # Create comprehensive result
            result = {
                'image_data': image_result['image_data'],
                'image_base64': image_result['image_base64'],
                'filename': image_result['filename'],
                'mime_type': image_result['mime_type'],
                'size_bytes': image_result['size_bytes'],
                'is_placeholder': image_result.get('is_placeholder', False),
                'selection_info': image_result.get('selection_info', {}),
                'dimensions': image_result.get('dimensions', (0, 0)),
                'text_removed': image_result.get('text_removed', True),
                'classification': classification,
                'title_used': final_title,
                'domain': classification_analysis.get('domain', 'General'),
                'agent': 'CoverImageAgent'
            }
            
            # Add API-ready object
            result['api_object'] = self.image_processor.create_api_image_object(
                image_result['image_data'],
                image_result['filename']
            )
            
            self.logger.log_operation_success(
                "cover_image_selection",
                {
                    "filename": result['filename'],
                    "size_bytes": result['size_bytes'],
                    "dimensions": f"{result['dimensions'][0]}x{result['dimensions'][1]}",
                    "is_placeholder": result['is_placeholder']
                },
                operation_context
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Cover image selection failed: {e}"
            self.logger.log_operation_failure("cover_image_selection", error_msg, operation_context)
            
            # Create fallback result with minimal placeholder
            try:
                placeholder_result = self.image_processor._create_placeholder_image()
                
                fallback_result = {
                    'image_data': placeholder_result['image_data'],
                    'image_base64': placeholder_result['image_base64'],
                    'filename': 'fallback_cover.png',
                    'mime_type': 'image/png',
                    'size_bytes': len(placeholder_result['image_data']),
                    'is_placeholder': True,
                    'dimensions': (800, 600),
                    'text_removed': False,
                    'classification': classification,
                    'title_used': final_title,
                    'domain': classification_analysis.get('domain', 'General'),
                    'agent': 'CoverImageAgent',
                    'error': error_msg,
                    'fallback_used': True
                }
                
                # Add API-ready object
                fallback_result['api_object'] = self.image_processor.create_api_image_object(
                    placeholder_result['image_data'],
                    'fallback_cover.png'
                )
                
                return fallback_result
                
            except Exception as fallback_error:
                # Ultimate fallback with minimal data
                minimal_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01`\x00\x00\x00\x00\x00\x00IEND\xaeB`\x82'
                
                return {
                    'image_data': minimal_data,
                    'image_base64': 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWPY9+8fAzYABgIDvvP1lN0AAAAASUVORK5CYII=',
                    'filename': 'minimal_cover.png',
                    'mime_type': 'image/png',
                    'size_bytes': len(minimal_data),
                    'is_placeholder': True,
                    'dimensions': (1, 1),
                    'text_removed': False,
                    'classification': classification,
                    'title_used': final_title,
                    'agent': 'CoverImageAgent',
                    'error': f"{error_msg}; Fallback also failed: {fallback_error}",
                    'critical_fallback_used': True,
                    'api_object': {
                        'data': 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWPY9+8fAzYABgIDvvP1lN0AAAAASUVORK5CYII=',
                        'filename': 'minimal_cover.png',
                        'mimeType': 'image/png'
                    }
                }
    
    async def select_multiple_images(
        self,
        content_list: list,
        classifications: list,
        classification_analyses: list,
        titles: list = None,
        state: Dict[str, Any] = None
    ) -> list:
        """
        Select cover images for multiple content items.
        
        Args:
            content_list: List of content dictionaries
            classifications: List of classifications
            classification_analyses: List of classification analyses
            titles: Optional list of optimized titles
            state: Optional pipeline state
            
        Returns:
            List of cover image results
        """
        
        self.logger.log_operation_start(
            "select_multiple_images",
            {"content_count": len(content_list)}
        )
        
        results = []
        
        for i, (content_dict, classification, analysis) in enumerate(zip(content_list, classifications, classification_analyses)):
            try:
                title = titles[i] if titles and i < len(titles) else None
                
                result = await self.execute(content_dict, classification, analysis, title, state)
                result['batch_index'] = i
                results.append(result)
                
            except Exception as e:
                self.logger.log_error(
                    "select_multiple_images",
                    f"Failed to select image for item {i}: {e}"
                )
                
                # Add minimal fallback result
                minimal_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01`\x00\x00\x00\x00\x00\x00IEND\xaeB`\x82'
                
                results.append({
                    'image_data': minimal_data,
                    'image_base64': 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWPY9+8fAzYABgIDvvP1lN0AAAAASUVORK5CYII=',
                    'filename': f'error_cover_{i}.png',
                    'mime_type': 'image/png',
                    'size_bytes': len(minimal_data),
                    'is_placeholder': True,
                    'batch_index': i,
                    'error': str(e),
                    'agent': 'CoverImageAgent',
                    'api_object': {
                        'data': 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWPY9+8fAzYABgIDvvP1lN0AAAAASUVORK5CYII=',
                        'filename': f'error_cover_{i}.png',
                        'mimeType': 'image/png'
                    }
                })
        
        self.logger.log_operation_success(
            "select_multiple_images",
            {"processed_count": len(results), "success_count": len([r for r in results if not r.get('error')])},
            {"content_count": len(content_list)}
        )
        
        return results
    
    def validate_image_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and standardize image result.
        
        Args:
            result: Image result from execute()
            
        Returns:
            Validated result
        """
        
        # Ensure required fields exist
        if not result.get('image_data') or not result.get('image_base64'):
            # Create minimal fallback
            minimal_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01`\x00\x00\x00\x00\x00\x00IEND\xaeB`\x82'
            
            result['image_data'] = minimal_data
            result['image_base64'] = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWPY9+8fAzYABgIDvvP1lN0AAAAASUVORK5CYII='
            result['is_placeholder'] = True
            result['validation_error'] = 'Missing image data, using minimal fallback'
        
        # Ensure required metadata
        result.setdefault('filename', 'cover.png')
        result.setdefault('mime_type', 'image/png')
        result.setdefault('size_bytes', len(result.get('image_data', b'')))
        result.setdefault('is_placeholder', False)
        result.setdefault('dimensions', (0, 0))
        
        # Ensure API object exists
        if not result.get('api_object'):
            result['api_object'] = {
                'data': result['image_base64'],
                'filename': result['filename'],
                'mimeType': result['mime_type']
            }
        
        return result
    
    def get_image_analytics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get analytics for an image selection result."""
        
        analytics = {
            'is_placeholder': result.get('is_placeholder', False),
            'text_removed': result.get('text_removed', False),
            'size_bytes': result.get('size_bytes', 0),
            'dimensions': result.get('dimensions', (0, 0)),
            'selection_successful': not result.get('error'),
            'api_ready': bool(result.get('api_object')),
            'filename': result.get('filename', 'unknown'),
            'classification_used': result.get('classification', 'unknown'),
            'domain_used': result.get('domain', 'unknown')
        }
        
        # Calculate image quality metrics
        width, height = analytics['dimensions']
        analytics['pixel_count'] = width * height
        analytics['aspect_ratio'] = width / height if height > 0 else 0
        analytics['is_standard_size'] = width >= 800 and height >= 600
        
        return analytics
    
    async def get_available_images_summary(self) -> Dict[str, Any]:
        """Get summary of available cover images."""
        
        try:
            from pathlib import Path
            
            base_dir = Path("docs/cover-image")
            
            summary = {
                'blog_images': 0,
                'product_images': 0,
                'total_images': 0,
                'blog_dir_exists': False,
                'product_dir_exists': False,
                'supported_formats': list(self.image_processor.supported_formats)
            }
            
            # Check blog directory
            blog_dir = base_dir / "blog"
            if blog_dir.exists():
                summary['blog_dir_exists'] = True
                blog_images = self.image_processor._get_available_images(blog_dir)
                summary['blog_images'] = len(blog_images)
            
            # Check product directory
            product_dir = base_dir / "product"
            if product_dir.exists():
                summary['product_dir_exists'] = True
                product_images = self.image_processor._get_available_images(product_dir)
                summary['product_images'] = len(product_images)
            
            summary['total_images'] = summary['blog_images'] + summary['product_images']
            
            self.logger.log_info(
                "available_images_summary",
                f"Found {summary['total_images']} images total",
                summary
            )
            
            return summary
            
        except Exception as e:
            self.logger.log_error("available_images_summary", f"Failed to get summary: {e}")
            
            return {
                'blog_images': 0,
                'product_images': 0,
                'total_images': 0,
                'blog_dir_exists': False,
                'product_dir_exists': False,
                'error': str(e)
            }