"""Cover Image Processor Tool - Handle image selection and text removal."""

import os
import base64
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageFilter
import io

from ..utils.publisher_logger import PublisherLogger


class CoverImageProcessorTool:
    """Tool for selecting and processing cover images."""
    
    def __init__(self):
        self.logger = PublisherLogger()
        self.base_cover_dir = Path("docs/cover-image")
        
        # Supported image formats
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        # Text removal settings
        self.text_detection_config = {
            'text_threshold': 0.7,
            'link_threshold': 0.4,
            'low_text': 0.4
        }
    
    async def select_cover_image(
        self,
        category: str,
        content_analysis: Dict[str, Any],
        content_title: str
    ) -> Dict[str, Any]:
        """
        Select appropriate cover image based on category and content analysis.
        
        Args:
            category: Blog or Product
            content_analysis: Analysis from content classification
            content_title: Title of the content
            
        Returns:
            Dictionary with selected image info
        """
        
        operation_context = {
            "category": category,
            "content_title": content_title,
            "domain": content_analysis.get("domain", "General")
        }
        
        self.logger.log_operation_start("select_cover_image", operation_context)
        
        try:
            # Determine folder based on category
            category_folder = "blog" if category.lower() == "blog" else "product"
            image_dir = self.base_cover_dir / category_folder
            
            # Check if directory exists
            if not image_dir.exists():
                self.logger.log_warning(
                    "select_cover_image",
                    f"Directory does not exist: {image_dir}",
                    operation_context
                )
                
                # Create directory and return placeholder info
                image_dir.mkdir(parents=True, exist_ok=True)
                return self._create_placeholder_image_info(category_folder)
            
            # Get available images
            available_images = self._get_available_images(image_dir)
            
            if not available_images:
                self.logger.log_warning(
                    "select_cover_image",
                    f"No images found in {image_dir}",
                    operation_context
                )
                return self._create_placeholder_image_info(category_folder)
            
            # Select best image based on content analysis
            selected_image_path = self._select_best_image(
                available_images,
                content_analysis,
                content_title
            )
            
            # Create image info
            image_info = {
                "selected_image_path": str(selected_image_path),
                "category": category,
                "category_folder": category_folder,
                "filename": selected_image_path.name,
                "available_count": len(available_images),
                "selection_method": "content_based"
            }
            
            self.logger.log_operation_success(
                "select_cover_image",
                {
                    "selected_image": selected_image_path.name,
                    "available_count": len(available_images)
                },
                operation_context
            )
            
            return image_info
            
        except Exception as e:
            error_msg = f"Cover image selection failed: {e}"
            self.logger.log_operation_failure("select_cover_image", error_msg, operation_context)
            
            # Return placeholder
            return self._create_placeholder_image_info("blog")
    
    def _get_available_images(self, image_dir: Path) -> List[Path]:
        """Get list of available image files."""
        
        images = []
        
        try:
            for file_path in image_dir.iterdir():
                if (file_path.is_file() and 
                    file_path.suffix.lower() in self.supported_formats):
                    images.append(file_path)
            
            return sorted(images)  # Sort for consistent selection
            
        except Exception as e:
            self.logger.log_warning("get_available_images", f"Error scanning directory: {e}")
            return []
    
    def _select_best_image(
        self,
        available_images: List[Path],
        content_analysis: Dict[str, Any],
        content_title: str
    ) -> Path:
        """Select best image based on content analysis."""
        
        try:
            # For now, use keyword-based selection and randomization
            # In the future, this could be enhanced with image analysis
            
            key_indicators = content_analysis.get("key_indicators", [])
            domain = content_analysis.get("domain", "").lower()
            
            # Score images based on filename matching
            image_scores = []
            
            for image_path in available_images:
                score = 0
                filename = image_path.stem.lower()
                
                # Domain matching
                if domain and domain in filename:
                    score += 10
                
                # Key indicator matching
                for indicator in key_indicators:
                    if indicator.lower() in filename:
                        score += 5
                
                # Title word matching
                title_words = content_title.lower().split()
                for word in title_words:
                    if len(word) > 3 and word in filename:
                        score += 3
                
                image_scores.append((image_path, score))
            
            # Sort by score (highest first)
            image_scores.sort(key=lambda x: x[1], reverse=True)
            
            # If top score is 0, select randomly
            if image_scores[0][1] == 0:
                selected_image = random.choice(available_images)
                self.logger.log_info(
                    "select_best_image",
                    f"Random selection: {selected_image.name}"
                )
            else:
                # Select highest scoring image
                selected_image = image_scores[0][0]
                self.logger.log_info(
                    "select_best_image",
                    f"Score-based selection: {selected_image.name} (score: {image_scores[0][1]})"
                )
            
            return selected_image
            
        except Exception as e:
            self.logger.log_warning("select_best_image", f"Error in selection logic: {e}")
            return available_images[0] if available_images else None
    
    async def process_cover_image(
        self,
        image_path: str,
        remove_text: bool = True
    ) -> Dict[str, Any]:
        """
        Process cover image (resize, format, remove text).
        
        Args:
            image_path: Path to the image file
            remove_text: Whether to remove text from image
            
        Returns:
            Dictionary with processed image data
        """
        
        operation_context = {
            "image_path": image_path,
            "remove_text": remove_text
        }
        
        self.logger.log_operation_start("process_cover_image", operation_context)
        
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                original_size = img.size
                
                # Resize if too large (max 1200x1200 for cover images)
                max_size = 1200
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Remove text if requested
                if remove_text:
                    img = await self._remove_text_from_image(img)
                
                # Convert to PNG format
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG', optimize=True)
                img_data = img_buffer.getvalue()
                
                # Create result
                result = {
                    "image_data": img_data,
                    "image_base64": base64.b64encode(img_data).decode('utf-8'),
                    "filename": f"{image_path.stem}_processed.png",
                    "mime_type": "image/png",
                    "size_bytes": len(img_data),
                    "dimensions": img.size,
                    "original_dimensions": original_size,
                    "text_removed": remove_text,
                    "processed_at": self._get_timestamp()
                }
                
                self.logger.log_operation_success(
                    "process_cover_image",
                    {
                        "size_bytes": len(img_data),
                        "dimensions": f"{img.size[0]}x{img.size[1]}",
                        "text_removed": remove_text
                    },
                    operation_context
                )
                
                return result
                
        except Exception as e:
            error_msg = f"Image processing failed: {e}"
            self.logger.log_operation_failure("process_cover_image", error_msg, operation_context)
            raise Exception(error_msg)
    
    async def _remove_text_from_image(self, image: Image.Image) -> Image.Image:
        """
        Remove text from image using computer vision techniques.
        
        This is a simplified implementation. In production, you might want to use
        more sophisticated text detection and removal techniques.
        """
        
        try:
            # Convert PIL image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Create a mask for text regions
            text_mask = self._detect_text_regions(cv_image)
            
            if text_mask is not None and np.any(text_mask):
                # Inpaint text regions
                inpainted = cv2.inpaint(cv_image, text_mask, 3, cv2.INPAINT_TELEA)
                
                # Convert back to PIL
                result_image = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
                
                self.logger.log_info("remove_text_from_image", "Text removal completed")
                return result_image
            else:
                self.logger.log_info("remove_text_from_image", "No text detected, returning original")
                return image
                
        except Exception as e:
            self.logger.log_warning("remove_text_from_image", f"Text removal failed: {e}")
            return image  # Return original image if text removal fails
    
    def _detect_text_regions(self, cv_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect text regions in image.
        
        This is a simplified implementation using basic computer vision techniques.
        For better results, consider using EAST text detector or other deep learning models.
        """
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Apply threshold to get binary image
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours that might be text
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create mask
            mask = np.zeros(gray.shape, dtype=np.uint8)
            
            # Filter contours that look like text
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Text-like characteristics: reasonable aspect ratio, not too small/large
                aspect_ratio = w / float(h)
                area = cv2.contourArea(contour)
                
                if (0.1 < aspect_ratio < 10.0 and 
                    50 < area < 5000 and
                    10 < w < 500 and 
                    8 < h < 100):
                    
                    # Draw filled rectangle on mask
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            
            return mask if np.any(mask) else None
            
        except Exception as e:
            self.logger.log_warning("detect_text_regions", f"Text detection failed: {e}")
            return None
    
    def _create_placeholder_image_info(self, category_folder: str) -> Dict[str, Any]:
        """Create placeholder image info when no images are available."""
        
        return {
            "selected_image_path": None,
            "category": category_folder.title(),
            "category_folder": category_folder,
            "filename": None,
            "available_count": 0,
            "selection_method": "placeholder",
            "error": f"No images available in {category_folder} folder"
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
        remove_text: bool = True
    ) -> Dict[str, Any]:
        """
        Combined method to select and process cover image.
        
        Returns:
            Dictionary with processed image data ready for API upload
        """
        
        operation_context = {
            "category": category,
            "content_title": content_title,
            "remove_text": remove_text
        }
        
        self.logger.log_operation_start("select_and_process_image", operation_context)
        
        try:
            # Select image
            image_info = await self.select_cover_image(category, content_analysis, content_title)
            
            if not image_info.get("selected_image_path"):
                # Create a simple placeholder image
                placeholder_result = self._create_placeholder_image()
                
                result = {
                    "image_data": placeholder_result["image_data"],
                    "image_base64": placeholder_result["image_base64"],
                    "filename": "placeholder_cover.png",
                    "mime_type": "image/png",
                    "size_bytes": len(placeholder_result["image_data"]),
                    "is_placeholder": True,
                    "selection_info": image_info
                }
                
                self.logger.log_warning(
                    "select_and_process_image",
                    "Using placeholder image",
                    operation_context
                )
                
                return result
            
            # Process selected image
            processed_image = await self.process_cover_image(
                image_info["selected_image_path"],
                remove_text
            )
            
            # Combine results
            result = {
                **processed_image,
                "selection_info": image_info,
                "is_placeholder": False
            }
            
            self.logger.log_operation_success(
                "select_and_process_image",
                {
                    "filename": result["filename"],
                    "size_bytes": result["size_bytes"]
                },
                operation_context
            )
            
            return result
            
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
                "error": error_msg
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