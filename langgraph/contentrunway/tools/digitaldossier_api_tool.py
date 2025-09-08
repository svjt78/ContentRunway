"""DigitalDossier API Tool - Handle all API communication with digitaldossier.us."""

import os
import base64
import httpx
from typing import Any, Dict, List, Optional
import asyncio
from ..utils.publisher_logger import PublisherLogger


class DigitalDossierAPITool:
    """Tool for interacting with DigitalDossier.us API."""
    
    def __init__(self):
        self.logger = PublisherLogger()
        self.api_token = os.getenv('DIGITALDOSSIER_API_TOKEN')
        self.base_url = os.getenv('DIGITALDOSSIER_BASE_URL', 'https://digitaldossier.us')
        self.admin_email = os.getenv('DIGITALDOSSIER_ADMIN_EMAIL')
        self.admin_password = os.getenv('DIGITALDOSSIER_ADMIN_PASSWORD')
        
        # Validate configuration
        self._validate_config()
        
        # HTTP client with timeout
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            headers={
                'Authorization': f'Bearer {self.api_token}',
                'Content-Type': 'application/json'
            }
        )
    
    def _validate_config(self):
        """Validate required environment variables."""
        if not self.api_token:
            raise ValueError("DIGITALDOSSIER_API_TOKEN environment variable is required")
        
        if not self.admin_email:
            raise ValueError("DIGITALDOSSIER_ADMIN_EMAIL environment variable is required")
        
        if not self.admin_password:
            raise ValueError("DIGITALDOSSIER_ADMIN_PASSWORD environment variable is required")
    
    async def fetch_genres(self) -> List[Dict[str, Any]]:
        """Fetch available genres from the API."""
        self.logger.log_operation_start("fetch_genres")
        
        try:
            response = await self.client.get(f"{self.base_url}/api/genres")
            response.raise_for_status()
            
            genres = response.json()
            self.logger.log_operation_success(
                "fetch_genres",
                {"genres_count": len(genres)},
                {"base_url": self.base_url}
            )
            
            return genres
            
        except httpx.HTTPError as e:
            error_msg = f"HTTP error fetching genres: {e}"
            self.logger.log_operation_failure("fetch_genres", error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error fetching genres: {e}"
            self.logger.log_operation_failure("fetch_genres", error_msg)
            raise Exception(error_msg)
    
    async def upload_document(
        self,
        title: str,
        category: str,
        genre_id: int,
        cover_image: Dict[str, str],
        pdf_file: Dict[str, str],
        summary: Optional[str] = None,
        content: Optional[str] = None
    ) -> Dict[str, Any]:
        """Upload document to DigitalDossier API."""
        
        operation_context = {
            "title": title,
            "category": category,
            "genre_id": genre_id,
            "has_cover_image": bool(cover_image),
            "has_pdf": bool(pdf_file)
        }
        
        self.logger.log_operation_start("upload_document", operation_context)
        
        try:
            # Prepare upload payload
            payload = {
                "title": title,
                "author": "Suvojit Dutta",
                "category": category,
                "genreId": genre_id,
                "coverImage": cover_image,
                "file": pdf_file
            }
            
            # Add optional fields
            if summary:
                payload["summary"] = summary
            if content:
                payload["content"] = content
            
            # Make API request
            response = await self.client.post(
                f"{self.base_url}/api/upload/programmatic",
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            
            self.logger.log_operation_success(
                "upload_document",
                result,
                operation_context
            )
            
            return result
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code} error uploading document: {e.response.text}"
            self.logger.log_operation_failure("upload_document", error_msg, operation_context)
            raise Exception(error_msg)
        except httpx.HTTPError as e:
            error_msg = f"HTTP error uploading document: {e}"
            self.logger.log_operation_failure("upload_document", error_msg, operation_context)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error uploading document: {e}"
            self.logger.log_operation_failure("upload_document", error_msg, operation_context)
            raise Exception(error_msg)
    
    async def batch_upload_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Upload multiple documents using batch API."""
        
        self.logger.log_operation_start(
            "batch_upload_documents",
            {"document_count": len(documents)}
        )
        
        try:
            # Prepare batch payload
            payload = {"documents": documents}
            
            response = await self.client.post(
                f"{self.base_url}/api/upload/programmatic/batch",
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            
            self.logger.log_operation_success(
                "batch_upload_documents",
                {"results_count": len(result)},
                {"document_count": len(documents)}
            )
            
            return result
            
        except httpx.HTTPError as e:
            error_msg = f"HTTP error in batch upload: {e}"
            self.logger.log_operation_failure("batch_upload_documents", error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error in batch upload: {e}"
            self.logger.log_operation_failure("batch_upload_documents", error_msg)
            raise Exception(error_msg)
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test API connection and authentication."""
        self.logger.log_operation_start("test_connection")
        
        try:
            # Test with genres endpoint (lightweight)
            response = await self.client.get(f"{self.base_url}/api/genres")
            
            if response.status_code == 200:
                result = {
                    "status": "success",
                    "message": "Connection successful",
                    "base_url": self.base_url,
                    "genres_available": len(response.json())
                }
            else:
                result = {
                    "status": "error",
                    "message": f"Connection failed with status {response.status_code}",
                    "base_url": self.base_url
                }
            
            self.logger.log_operation_success("test_connection", result)
            return result
            
        except Exception as e:
            error_msg = f"Connection test failed: {e}"
            result = {
                "status": "error",
                "message": error_msg,
                "base_url": self.base_url
            }
            
            self.logger.log_operation_failure("test_connection", error_msg)
            return result
    
    def create_cover_image_object(self, image_data: bytes, filename: str = "cover.png") -> Dict[str, str]:
        """Create cover image object for API upload."""
        return {
            "data": base64.b64encode(image_data).decode('utf-8'),
            "filename": filename,
            "mimeType": "image/png"
        }
    
    def create_pdf_file_object(self, pdf_data: bytes, filename: str = "document.pdf") -> Dict[str, str]:
        """Create PDF file object for API upload."""
        return {
            "data": base64.b64encode(pdf_data).decode('utf-8'),
            "filename": filename,
            "mimeType": "application/pdf"
        }
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()