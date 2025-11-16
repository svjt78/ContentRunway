"""Cover Image Processor Tool - Handle image selection and text removal."""

import os
import base64
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from PIL import Image
import io

from ..utils.publisher_logger import PublisherLogger
# DALL-E generation temporarily disabled for cost optimization
# from .dalle_image_generator_tool import DalleImageGeneratorTool


class CoverImageProcessorTool:
    """Tool for selecting and processing cover images."""
    
    def __init__(self):
        self.logger = PublisherLogger()
        # DALL-E generation temporarily disabled for cost optimization
        # self.dalle_generator = DalleImageGeneratorTool()
        
        # Supported image formats
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    async def select_cover_image(
        self,
        category: str,
        content_analysis: Dict[str, Any],
        content_title: str
    ) -> Dict[str, Any]:
        """
        Select cover image from local files (DALL-E generation disabled for cost optimization).
        
        Args:
            category: Blog or Product
            content_analysis: Analysis from content classification
            content_title: Title of the content
            
        Returns:
            Dictionary with selected image info or placeholder
        """
        
        operation_context = {
            "category": category,
            "content_title": content_title,
            "domain": content_analysis.get("domain", "General")
        }
        
        self.logger.log_operation_start("select_cover_image", operation_context)
        
        try:
            # Use local image selection instead of DALL-E generation
            image_dir = Path("docs/cover-image") / category.lower()
            
            if image_dir.exists() and image_dir.is_dir():
                # Get available images from directory
                available_images = self._get_available_images(image_dir)
                
                if available_images:
                    # Select random image from available images
                    selected_image = random.choice(available_images)
                    
                    image_info = {
                        "selected_image_path": str(selected_image),
                        "category": category,
                        "category_folder": category.lower(),
                        "filename": selected_image.name,
                        "available_count": len(available_images),
                        "selection_method": "local_file_random"
                    }
                    
                    self.logger.log_operation_success(
                        "select_cover_image",
                        {
                            "selected_image": selected_image.name,
                            "available_count": len(available_images)
                        },
                        operation_context
                    )
                    
                    return image_info
            
            # No images available - return placeholder info
            self.logger.log_operation_failure(
                "select_cover_image", 
                f"No images found in {image_dir}", 
                operation_context
            )
            return self._create_placeholder_image_info(category.lower(), f"No images found in {image_dir}")
            
        except Exception as e:
            error_msg = f"Image selection failed: {e}"
            self.logger.log_operation_failure("select_cover_image", error_msg, operation_context)
            
            # Return placeholder info for backward compatibility
            return self._create_placeholder_image_info(category.lower(), error_msg)
    
    def _get_available_images(self, image_dir: Path) -> List[Path]:
        """Get available image files from directory."""
        try:
            if not image_dir.exists():
                return []
            
            available_images = []
            for file_path in image_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                    available_images.append(file_path)
            
            return available_images
        except Exception as e:
            self.logger.log_error("_get_available_images", f"Failed to scan directory {image_dir}: {e}")
            return []
    
    def _select_best_image(
        self,
        available_images: List[Path],
        content_analysis: Dict[str, Any],
        content_title: str
    ) -> Path:
        """Legacy method kept for backward compatibility."""
        return None  # No longer needed with DALL-E generation
    
    async def process_cover_image(
        self,
        image_path: str,
        remove_text: bool = True,
        replacement_title: str = None
    ) -> Dict[str, Any]:
        """
        Process cover image - now handles both file paths and pre-generated DALL-E results.
        
        Args:
            image_path: Path to image file OR special marker for DALL-E generated images
            remove_text: Ignored for DALL-E images (no text removal needed)
            replacement_title: Ignored for DALL-E images (title already integrated)
            
        Returns:
            Dictionary with processed image data
        """
        
        operation_context = {
            "image_path": image_path,
            "remove_text": remove_text,
            "replacement_title": replacement_title
        }
        
        self.logger.log_operation_start("process_cover_image", operation_context)
        
        # For DALL-E generated images, we don't need to process from file
        # The image data is already processed and ready
        if image_path and "dalle_generated" in str(image_path):
            self.logger.log_info("process_cover_image", "DALL-E generated image, no processing needed")
            # Return a placeholder result - actual processing happens in select_and_process_image
            return {
                "dalle_processed": True,
                "message": "DALL-E image processed in select_and_process_image method"
            }
        
        try:
            # Legacy file processing (kept for backward compatibility)
            if image_path:
                image_path = Path(image_path)
                
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                
                # Load and process image
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    original_size = img.size
                    
                    # Resize if too large
                    max_size = 1200
                    if max(img.size) > max_size:
                        ratio = max_size / max(img.size)
                        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Convert to PNG format
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG', optimize=True)
                    img_data = img_buffer.getvalue()
                    
                    result = {
                        "image_data": img_data,
                        "image_base64": base64.b64encode(img_data).decode('utf-8'),
                        "filename": f"{image_path.stem}_processed.png",
                        "mime_type": "image/png",
                        "size_bytes": len(img_data),
                        "dimensions": img.size,
                        "original_dimensions": original_size,
                        "text_removed": False,  # No text removal for legacy images
                        "processed_at": self._get_timestamp()
                    }
                    
                    self.logger.log_operation_success(
                        "process_cover_image",
                        {
                            "size_bytes": len(img_data),
                            "dimensions": f"{img.size[0]}x{img.size[1]}"
                        },
                        operation_context
                    )
                    
                    return result
            else:
                raise ValueError("No image path provided")
                
        except Exception as e:
            error_msg = f"Image processing failed: {e}"
            self.logger.log_operation_failure("process_cover_image", error_msg, operation_context)
            raise Exception(error_msg)
    
    async def _remove_text_from_image(self, image: Image.Image, replacement_title: str = None) -> Image.Image:
        """Legacy method kept for backward compatibility - no longer used with DALL-E."""
        return image  # No text removal needed with DALL-E generated images
    
    def _detect_text_regions(self, cv_image) -> Optional[Any]:
        """Legacy method kept for backward compatibility."""
        return None  # No longer needed with DALL-E generation
    
    def _detect_text_regions_with_positions(self, cv_image) -> Tuple[List[Tuple[int, int, int, int]], Optional[Any]]:
        """Legacy method kept for backward compatibility."""
        return [], None  # No longer needed with DALL-E generation
    
    def _get_system_fonts(self) -> List[str]:
        """Legacy method kept for backward compatibility."""
        return []  # No longer needed with DALL-E generation
    
    def _get_font(self, font_size: int, font_name: str = None):
        """Legacy method kept for backward compatibility."""
        return None  # No longer needed with DALL-E generation
    
    def _calculate_optimal_font_size(self, title: str, image_width: int, image_height: int, target_region: Tuple[int, int, int, int] = None) -> int:
        """Legacy method kept for backward compatibility."""
        return 36  # Default font size, no longer needed with DALL-E generation
    
    def _get_text_color_with_contrast(self, image: Image.Image, text_position: Tuple[int, int]) -> Tuple[int, int, int]:
        """Legacy method kept for backward compatibility."""
        return (255, 255, 255)  # Default white, no longer needed with DALL-E generation
    
    def _add_title_to_image(self, image: Image.Image, title: str, detected_text_regions: List[Tuple[int, int, int, int]]) -> Image.Image:
        """Legacy method kept for backward compatibility."""
        return image  # No text overlay needed with DALL-E generated images
    
    def _create_placeholder_image_info(self, category_folder: str, error_msg: str = None) -> Dict[str, Any]:
        """Create placeholder image info when generation fails."""
        
        return {
            "selected_image_path": None,
            "category": category_folder.title(),
            "category_folder": category_folder,
            "filename": None,
            "available_count": 0,
            "selection_method": "placeholder",
            "error": error_msg or f"Image generation failed for {category_folder} category"
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    async def select_and_process_image(
        self,
        category: str,
        content_analysis: Dict[str, Any],
        content_title: str,
        remove_text: bool = True,
        replacement_title: str = None
    ) -> Dict[str, Any]:
        """
        Generate cover image using DALL-E API - maintains backward compatible interface.
        
        Args:
            category: Blog or Product category
            content_analysis: Content analysis from classification
            content_title: Original content title (used for image generation)
            remove_text: Ignored for DALL-E images (no text removal needed)
            replacement_title: Title to use for generation (if None, uses content_title)
        
        Returns:
            Dictionary with processed image data ready for API upload
        """
        
        # Use replacement title if provided, otherwise use content title
        final_title = replacement_title or content_title
        
        operation_context = {
            "category": category,
            "content_title": content_title,
            "replacement_title": final_title,
            "remove_text": remove_text  # Will be ignored but logged for compatibility
        }
        
        self.logger.log_operation_start("select_and_process_image", operation_context)
        
        try:
            # Use local image selection instead of DALL-E generation
            image_dir = Path("docs/cover-image") / category.lower()
            
            if image_dir.exists() and image_dir.is_dir():
                # Get available images from directory
                available_images = self._get_available_images(image_dir)
                
                if available_images:
                    # Select random image from available images
                    selected_image = random.choice(available_images)
                    
                    # Process the selected image
                    image_result = await self.process_cover_image(
                        str(selected_image),
                        remove_text=remove_text,
                        replacement_title=final_title
                    )
                    
                    # Add selection info
                    image_result["selection_info"] = {
                        "method": "local_file_random",
                        "selected_image": selected_image.name,
                        "available_count": len(available_images)
                    }
                    
                    self.logger.log_operation_success(
                        "select_and_process_image",
                        {
                            "filename": image_result["filename"],
                            "size_bytes": image_result["size_bytes"],
                            "selected_from": len(available_images)
                        },
                        operation_context
                    )
                    
                    return image_result
            
            # No local images found - create placeholder
            self.logger.log_operation_failure(
                "select_and_process_image",
                f"No images found in {image_dir}",
                operation_context
            )
            
            # Return placeholder as fallback
            placeholder_result = self._create_placeholder_image()
            
            return {
                "image_data": placeholder_result["image_data"],
                "image_base64": placeholder_result["image_base64"],
                "filename": f"{category.lower()}_placeholder.png",
                "mime_type": "image/png",
                "size_bytes": len(placeholder_result["image_data"]),
                "is_placeholder": True,
                "error": f"No images found in {image_dir}",
                "text_removed": False,
                "dimensions": (800, 600),
                "selection_info": {
                    "method": "placeholder_generated",
                    "reason": "no_local_images_found"
                }
            }
            
        except Exception as e:
            error_msg = f"Image selection and processing failed: {e}"
            self.logger.log_operation_failure("select_and_process_image", error_msg, operation_context)
            
            # Return placeholder as fallback
            placeholder_result = self._create_placeholder_image()
            
            return {
                "image_data": placeholder_result["image_data"],
                "image_base64": placeholder_result["image_base64"],
                "filename": "error_placeholder.png",
                "mime_type": "image/png",
                "size_bytes": len(placeholder_result["image_data"]),
                "is_placeholder": True,
                "error": error_msg,
                "text_removed": False,
                "dimensions": (800, 600)
            }
    
    def _create_placeholder_image(self) -> Dict[str, Any]:
        """Create a simple placeholder image."""
        
        try:
            # Create a simple 800x600 placeholder image
            img = Image.new('RGB', (800, 600), color='#f8f9fa')
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_data = img_buffer.getvalue()
            
            return {
                "image_data": img_data,
                "image_base64": base64.b64encode(img_data).decode('utf-8')
            }
            
        except Exception as e:
            self.logger.log_error("create_placeholder_image", f"Failed to create placeholder: {e}")
            # Return minimal base64 encoded 1x1 pixel image as ultimate fallback
            minimal_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01`\x00\x00\x00\x00\x00\x00IEND\xaeB`\x82'
            
            return {
                "image_data": minimal_png,
                "image_base64": base64.b64encode(minimal_png).decode('utf-8')
            }
    
    def create_api_image_object(self, image_data: bytes, filename: str = "cover.png") -> Dict[str, str]:
        """Create image object for DigitalDossier API."""
        
        return {
            "data": base64.b64encode(image_data).decode('utf-8'),
            "filename": filename,
            "mimeType": "image/png"
        }