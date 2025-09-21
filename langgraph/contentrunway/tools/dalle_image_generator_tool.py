"""DALL-E Image Generator Tool - Generate cover images using OpenAI DALL-E API."""

import os
import base64
import io
from typing import Dict, Any, Optional
from PIL import Image
import openai
from openai import OpenAI

from ..utils.publisher_logger import PublisherLogger


class DalleImageGeneratorTool:
    """Tool for generating cover images using OpenAI DALL-E API."""
    
    def __init__(self):
        self.logger = PublisherLogger()
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=api_key)
        
        # Image generation settings
        self.dalle_config = {
            'model': 'dall-e-3',
            'size': '1024x1024',
            'quality': 'standard',
            'style': 'vivid',  # More dynamic and engaging for covers
            'n': 1  # Generate 1 image per request
        }
        
        # Domain-specific prompt templates
        self.domain_prompts = {
            'it insurance': {
                'blog': 'Professional blog cover image about {title}, IT insurance and cybersecurity theme, digital shield icons, modern corporate design, blue and white color scheme, clean typography space for title overlay',
                'product': 'Product marketing cover image for {title} in IT insurance industry, cybersecurity elements, professional technology design, corporate blue theme with geometric patterns'
            },
            'ai research': {
                'blog': 'Academic blog cover image for {title} about AI research, neural network visualization, data flow diagrams, modern scientific design, teal and purple gradients, space for title text',
                'product': 'AI research product cover for {title}, machine learning visualizations, algorithmic patterns, futuristic tech design, gradient backgrounds with data elements'
            },
            'agentic ai': {
                'blog': 'Technical blog cover for {title} about agentic AI systems, interconnected agent networks, workflow diagrams, modern tech design, green and blue color scheme',
                'product': 'Agentic AI product cover for {title}, multi-agent system visualization, collaborative AI networks, sophisticated technical design with interconnected nodes'
            },
            'general': {
                'blog': 'Professional blog cover image for {title}, clean modern design, corporate style, neutral colors with subtle tech elements, space for title overlay',
                'product': 'Product marketing cover for {title}, sleek professional design, modern technology theme, engaging visual elements'
            }
        }
    
    async def generate_cover_image(
        self,
        title: str,
        category: str,
        domain: str = 'general',
        content_analysis: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a cover image using DALL-E API.
        
        Args:
            title: Content title
            category: 'blog' or 'product'
            domain: Content domain (IT Insurance, AI Research, Agentic AI, etc.)
            content_analysis: Optional content analysis for prompt enhancement
            
        Returns:
            Dictionary with generated image data
        """
        
        operation_context = {
            "title": title,
            "category": category,
            "domain": domain
        }
        
        self.logger.log_operation_start("generate_cover_image", operation_context)
        
        try:
            # Generate prompt based on domain and category
            prompt = self._create_image_prompt(title, category, domain, content_analysis)
            
            self.logger.log_info("generate_cover_image", f"Generated prompt: {prompt}")
            
            # Call DALL-E API
            response = self.client.images.generate(
                model=self.dalle_config['model'],
                prompt=prompt,
                size=self.dalle_config['size'],
                quality=self.dalle_config['quality'],
                style=self.dalle_config['style'],
                n=self.dalle_config['n']
            )
            
            # Get the generated image URL
            image_url = response.data[0].url
            
            # Download and process the image
            image_data = await self._download_and_process_image(image_url, title)
            
            # Create result
            result = {
                "image_data": image_data['image_bytes'],
                "image_base64": image_data['image_base64'],
                "filename": f"{self._sanitize_filename(title)}_cover.png",
                "mime_type": "image/png",
                "size_bytes": len(image_data['image_bytes']),
                "dimensions": (1024, 1024),
                "is_placeholder": False,
                "dalle_prompt": prompt,
                "dalle_model": self.dalle_config['model'],
                "generated_at": self._get_timestamp()
            }
            
            self.logger.log_operation_success(
                "generate_cover_image",
                {
                    "filename": result["filename"],
                    "size_bytes": result["size_bytes"],
                    "prompt_length": len(prompt)
                },
                operation_context
            )
            
            return result
            
        except Exception as e:
            error_msg = f"DALL-E image generation failed: {e}"
            self.logger.log_operation_failure("generate_cover_image", error_msg, operation_context)
            
            # Return placeholder as fallback
            return self._create_placeholder_result(title, error_msg)
    
    def _create_image_prompt(
        self,
        title: str,
        category: str,
        domain: str,
        content_analysis: Dict[str, Any] = None
    ) -> str:
        """Create optimized prompt for DALL-E based on content."""
        
        # Normalize domain for lookup
        domain_key = domain.lower() if domain else 'general'
        if domain_key not in self.domain_prompts:
            domain_key = 'general'
        
        # Get category (default to blog if not recognized)
        category_key = category.lower() if category.lower() in ['blog', 'product'] else 'blog'
        
        # Get base prompt template
        base_prompt = self.domain_prompts[domain_key][category_key]
        
        # Format with title
        prompt = base_prompt.format(title=title)
        
        # Enhance prompt with content analysis if available
        if content_analysis:
            key_indicators = content_analysis.get('key_indicators', [])
            if key_indicators:
                # Add relevant keywords to prompt
                keywords = ', '.join(key_indicators[:3])  # Use top 3 indicators
                prompt += f", incorporating themes of {keywords}"
        
        # Add quality and style directives
        prompt += ", high quality, professional, no text overlay, suitable for cover image"
        
        return prompt
    
    async def _download_and_process_image(self, image_url: str, title: str) -> Dict[str, Any]:
        """Download image from URL and convert to required format."""
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        image_bytes = await response.read()
                        
                        # Convert to PNG if necessary and ensure correct format
                        with Image.open(io.BytesIO(image_bytes)) as img:
                            # Convert to RGB if necessary
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # Save as PNG
                            img_buffer = io.BytesIO()
                            img.save(img_buffer, format='PNG', optimize=True)
                            final_image_bytes = img_buffer.getvalue()
                            
                            return {
                                'image_bytes': final_image_bytes,
                                'image_base64': base64.b64encode(final_image_bytes).decode('utf-8')
                            }
                    else:
                        raise Exception(f"Failed to download image: HTTP {response.status}")
                        
        except ImportError:
            # Fallback to requests if aiohttp not available
            import requests
            
            response = requests.get(image_url, timeout=30)
            if response.status_code == 200:
                image_bytes = response.content
                
                # Convert to PNG if necessary
                with Image.open(io.BytesIO(image_bytes)) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG', optimize=True)
                    final_image_bytes = img_buffer.getvalue()
                    
                    return {
                        'image_bytes': final_image_bytes,
                        'image_base64': base64.b64encode(final_image_bytes).decode('utf-8')
                    }
            else:
                raise Exception(f"Failed to download image: HTTP {response.status_code}")
        
        except Exception as e:
            self.logger.log_error("download_and_process_image", f"Failed to download/process image: {e}")
            raise
    
    def _create_placeholder_result(self, title: str, error_msg: str) -> Dict[str, Any]:
        """Create placeholder result when DALL-E generation fails."""
        
        try:
            # Create a simple 1024x1024 placeholder image
            img = Image.new('RGB', (1024, 1024), color='#f8f9fa')
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_data = img_buffer.getvalue()
            
            return {
                "image_data": img_data,
                "image_base64": base64.b64encode(img_data).decode('utf-8'),
                "filename": f"{self._sanitize_filename(title)}_placeholder.png",
                "mime_type": "image/png",
                "size_bytes": len(img_data),
                "dimensions": (1024, 1024),
                "is_placeholder": True,
                "error": error_msg,
                "generated_at": self._get_timestamp()
            }
            
        except Exception as e:
            # Ultimate fallback with minimal PNG
            self.logger.log_error("create_placeholder_result", f"Failed to create placeholder: {e}")
            minimal_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01`\x00\x00\x00\x00\x00\x00IEND\xaeB`\x82'
            
            return {
                "image_data": minimal_png,
                "image_base64": base64.b64encode(minimal_png).decode('utf-8'),
                "filename": "minimal_placeholder.png",
                "mime_type": "image/png",
                "size_bytes": len(minimal_png),
                "dimensions": (1, 1),
                "is_placeholder": True,
                "error": f"{error_msg}; Placeholder creation also failed: {e}",
                "critical_fallback": True,
                "generated_at": self._get_timestamp()
            }
    
    def _sanitize_filename(self, title: str) -> str:
        """Sanitize title for use as filename."""
        
        import re
        
        # Remove or replace invalid characters
        sanitized = re.sub(r'[^\w\s-]', '', title)
        sanitized = re.sub(r'[-\s]+', '_', sanitized)
        
        # Limit length
        if len(sanitized) > 50:
            sanitized = sanitized[:50]
        
        return sanitized.lower() or 'cover'
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def create_api_image_object(self, image_data: bytes, filename: str = "cover.png") -> Dict[str, str]:
        """Create image object for DigitalDossier API - same interface as original tool."""
        
        return {
            "data": base64.b64encode(image_data).decode('utf-8'),
            "filename": filename,
            "mimeType": "image/png"
        }
    
    async def get_generation_summary(self) -> Dict[str, Any]:
        """Get summary of DALL-E generation capabilities."""
        
        return {
            'model': self.dalle_config['model'],
            'max_resolution': self.dalle_config['size'],
            'supported_domains': list(self.domain_prompts.keys()),
            'supported_categories': ['blog', 'product'],
            'quality_settings': {
                'model': self.dalle_config['model'],
                'quality': self.dalle_config['quality'],
                'style': self.dalle_config['style']
            },
            'fallback_enabled': True
        }