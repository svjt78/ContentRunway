"""DigitalDossier API Tool - Handle all API communication with digitaldossier.us."""

import os
import base64
import copy
import httpx
from typing import Any, Dict, List, Optional
import asyncio
from ..utils.publisher_logger import PublisherLogger
from ..utils.jwt_auth import DigitalDossierJWTAuth


class DigitalDossierAPITool:
    """Tool for interacting with DigitalDossier.us API."""
    
    def __init__(self):
        self.logger = PublisherLogger()
        self.base_url = os.getenv('DIGITALDOSSIER_BASE_URL', 'http://localhost:3003')
        
        # Initialize JWT authentication
        self.auth = DigitalDossierJWTAuth(
            base_url=self.base_url,
            admin_email=os.getenv('DIGITALDOSSIER_ADMIN_EMAIL'),
            admin_password=os.getenv('DIGITALDOSSIER_ADMIN_PASSWORD')
        )
        
        # HTTP client with timeout
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0)
        )
    
    
    
    async def fetch_genres(self) -> List[Dict[str, Any]]:
        """Fetch available genres from the API."""
        self.logger.log_operation_start("fetch_genres")
        
        try:
            # Use JWT authentication for genres
            headers = await self.auth.get_authenticated_headers()
            response = await self.client.get(
                f"{self.base_url}/api/genres",
                headers=headers
            )
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
        cover_image: Optional[Dict[str, str]],
        pdf_file: Dict[str, str],
        summary: Optional[str] = None,
        content: Optional[str] = None,
        genre_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Upload document to DigitalDossier API."""
        
        operation_context = {
            "title": title,
            "category": category,
            "genre_id": genre_id,
            "has_cover_image": bool(cover_image),
            "has_pdf_file": bool(pdf_file),
            "has_genre_metadata": bool(genre_metadata)
        }
        
        self.logger.log_operation_start("upload_document", operation_context)
        
        try:
            # Prepare upload payload with correct field structure
            payload = {
                "title": title,
                "author": "Suvojit Dutta",
                "category": category,
                "genreId": genre_id,
                "pdfFile": pdf_file  # Fixed: Use 'pdfFile' instead of 'file'
            }
            
            # Add cover image only if present and valid
            self.logger.log_info("upload_document", f"🔧 DEBUG: Cover image check - cover_image: {cover_image}")
            self.logger.log_info("upload_document", f"🔧 DEBUG: Cover image type: {type(cover_image)}")
            if cover_image:
                self.logger.log_info("upload_document", f"🔧 DEBUG: Cover image keys: {list(cover_image.keys())}")
                self.logger.log_info("upload_document", f"🔧 DEBUG: Cover image data present: {bool(cover_image.get('data'))}")
                self.logger.log_info("upload_document", f"🔧 DEBUG: Cover image data length: {len(cover_image.get('data', ''))}")
            
            # Check if cover image should be included (including placeholder images)
            should_include_cover = False
            
            if cover_image:
                self.logger.log_info("upload_document", f"🔧 DEBUG: Cover image provided: {cover_image.get('filename')}")
                
                # Always include cover images if provided (server requires them)
                # Check for regular images with data
                if cover_image.get('data'):
                    should_include_cover = True
                    self.logger.log_info("upload_document", f"🔧 DEBUG: Cover image has data ({len(cover_image['data'])} chars), will include")
                
                # Check for placeholder images - be more lenient
                elif 'placeholder' in cover_image.get('filename', '').lower():
                    should_include_cover = True
                    self.logger.log_info("upload_document", f"🔧 DEBUG: Placeholder cover image detected, will include")
                    # Ensure placeholder has minimal base64 data
                    if not cover_image.get('data'):
                        cover_image['data'] = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAFFeaR9cAAAAABJRU5ErkJggg=="
                        self.logger.log_info("upload_document", f"🔧 DEBUG: Added minimal base64 data to placeholder")
                
                # If we have any cover image object, try to use it (server validation will catch issues)
                else:
                    should_include_cover = True
                    self.logger.log_info("upload_document", f"🔧 DEBUG: Cover image object provided, will attempt to include")
            
            if should_include_cover:
                self.logger.log_info("upload_document", f"🔧 DEBUG: Cover image validation passed, adding to payload")
                
                # Ensure proper MIME type detection
                if not cover_image.get('mimeType'):
                    # Default to PNG if not specified
                    cover_image['mimeType'] = 'image/png'
                
                # Validate MIME type
                valid_mime_types = ['image/png', 'image/jpeg', 'image/jpg']
                if cover_image.get('mimeType') not in valid_mime_types:
                    self.logger.log_warning("upload_document", f"Invalid MIME type {cover_image.get('mimeType')}, defaulting to image/png")
                    cover_image['mimeType'] = 'image/png'
                
                payload["coverImage"] = cover_image
                self.logger.log_info("upload_document", f"🔧 DEBUG: Cover image added to payload successfully")
            else:
                # Server requires cover image, so create a default placeholder if none provided
                self.logger.log_warning("upload_document", f"🔧 DEBUG: No cover image provided, creating default placeholder")
                default_placeholder = {
                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAFFeaR9cAAAAABJRU5ErkJggg==",
                    "filename": "default-placeholder-cover.png",
                    "mimeType": "image/png"
                }
                payload["coverImage"] = default_placeholder
                self.logger.log_info("upload_document", f"🔧 DEBUG: Default placeholder cover image added to payload")
            
            # Add genre metadata for auto-creation if provided
            if genre_metadata:
                payload["genre"] = {
                    "id": genre_id,
                    "name": genre_metadata.get("selected_genre", "Auto-Generated Genre"),
                    "description": genre_metadata.get("reasoning", "Auto-generated genre based on content analysis"),
                    "isAutoGenerated": True,
                    "confidence": genre_metadata.get("confidence_score", 0.7),
                    "domain": genre_metadata.get("domain_focus", "General"),
                    "createdBy": "ContentRunway-LLM"
                }
            
            # Note: All uploads are PDF-only, no summary or content text needed
            
            # Log payload structure for debugging
            debug_payload = {
                "title": title,
                "category": category,
                "genreId": genre_id,
                "has_coverImage": bool(payload.get("coverImage")),
                "coverImage_mimeType": payload.get("coverImage", {}).get("mimeType"),
                "has_pdfFile": bool(payload.get("pdfFile")),
                "pdfFile_mimeType": payload.get("pdfFile", {}).get("mimeType"),
                "has_genre_metadata": bool(payload.get("genre")),
                "genre_name": payload.get("genre", {}).get("name") if payload.get("genre") else None
            }
            self.logger.log_info("upload_document", f"Upload payload structure: {debug_payload}")
            
            # Always use JWT-based authentication for uploads (explicitly avoid API token path)
            headers = await self.auth.get_authenticated_headers()
            self.logger.log_info("upload_document", f"🔧 DEBUG: Using JWT authentication; headers: {list(headers.keys())}")
            
            # Log complete request details for debugging
            upload_url = f"{self.base_url}/api/upload/programmatic"
            self.logger.log_info("upload_document", f"🔧 DEBUG: Upload URL: {upload_url}")
            self.logger.log_info("upload_document", f"🔧 DEBUG: Request headers: {headers}")
            self.logger.log_info("upload_document", f"🔧 DEBUG: Payload size: {len(str(payload))} characters")
            
            # Validate PDF object presence and minimal size BEFORE creating safe payload
            try:
                pdf_obj = payload.get('pdfFile', {})
                pdf_b64 = pdf_obj.get('data', '')
                if not pdf_b64:
                    raise ValueError("PDF base64 missing")
                # Validate by decoding base64 and checking raw byte size
                decoded = base64.b64decode(pdf_b64, validate=True)
                if len(decoded) < 1024:
                    raise ValueError(f"PDF payload too small after decode (bytes={len(decoded)})")
            except Exception as e:
                raise Exception(f"PDF validation failed before upload: {e}")
            
            # Log payload without sensitive data - use DEEP COPY to avoid mutation
            safe_payload = copy.deepcopy(payload)
            if 'coverImage' in safe_payload and 'data' in safe_payload['coverImage']:
                safe_payload['coverImage']['data'] = f"[BASE64_DATA_{len(payload['coverImage']['data'])}_CHARS]"
            if 'pdfFile' in safe_payload and 'data' in safe_payload['pdfFile']:
                safe_payload['pdfFile']['data'] = f"[BASE64_DATA_{len(payload['pdfFile']['data'])}_CHARS]"
            self.logger.log_info("upload_document", f"🔧 DEBUG: Safe payload structure: {safe_payload}")

            self.logger.log_info("upload_document", f"🔧 DEBUG: Making POST request to {upload_url}")
            
            response = await self.client.post(
                upload_url,
                json=payload,
                headers=headers
            )
            
            self.logger.log_info("upload_document", f"🔧 DEBUG: Response status: {response.status_code}")
            self.logger.log_info("upload_document", f"🔧 DEBUG: Response headers: {dict(response.headers)}")
            
            # Log response content for debugging
            response_text = response.text
            self.logger.log_info("upload_document", f"🔧 DEBUG: Response text: {response_text}")
            
            response.raise_for_status()
            result = response.json()

            # Post-upload sanity: require id or url to consider success
            _data = result.get('data', result)
            if not (_data.get('id') or _data.get('url') or _data.get('document_url') or _data.get('pdfUrl')):
                self.logger.log_warning("upload_document", f"Upload returned 200 but no id/url in response: {result}")
            
            self.logger.log_info("upload_document", f"🔧 DEBUG: Parsed JSON response: {result}")
            
            self.logger.log_operation_success(
                "upload_document",
                result,
                operation_context
            )
            
            return result
            
        except httpx.HTTPStatusError as e:
            # Enhanced error logging for HTTP status errors
            error_details = {
                "status_code": e.response.status_code,
                "response_text": e.response.text,
                "response_headers": dict(e.response.headers),
                "request_url": str(e.response.url),
                "request_method": e.response.request.method
            }
            self.logger.log_error("upload_document", f"🔧 DEBUG: HTTP Status Error Details: {error_details}")
            
            error_msg = f"HTTP {e.response.status_code} error uploading document: {e.response.text}"
            self.logger.log_operation_failure("upload_document", error_msg, operation_context)
            raise Exception(error_msg)
        except httpx.HTTPError as e:
            self.logger.log_error("upload_document", f"🔧 DEBUG: HTTP Error Details: {str(e)}")
            self.logger.log_error("upload_document", f"🔧 DEBUG: HTTP Error Type: {type(e)}")
            
            error_msg = f"HTTP error uploading document: {e}"
            self.logger.log_operation_failure("upload_document", error_msg, operation_context)
            raise Exception(error_msg)
        except Exception as e:
            # Enhanced logging for unexpected errors
            import traceback
            self.logger.log_error("upload_document", f"🔧 DEBUG: Unexpected Error Details: {str(e)}")
            self.logger.log_error("upload_document", f"🔧 DEBUG: Unexpected Error Type: {type(e)}")
            self.logger.log_error("upload_document", f"🔧 DEBUG: Full Traceback: {traceback.format_exc()}")
            
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
            # Test with genres endpoint using JWT authentication
            headers = await self.auth.get_authenticated_headers()
            response = await self.client.get(
                f"{self.base_url}/api/genres", 
                headers=headers
            )
            
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
        
        # Determine MIME type from filename extension
        import os
        file_ext = os.path.splitext(filename)[1].lower()
        
        mime_type_mapping = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg'
        }
        
        mime_type = mime_type_mapping.get(file_ext, 'image/png')
        
        # Validate image data
        if not image_data:
            raise ValueError("Image data cannot be empty")
        
        # Ensure we have valid base64 data
        try:
            base64_data = base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to encode image data to base64: {e}")
        
        return {
            "data": base64_data,
            "filename": filename,
            "mimeType": mime_type
        }
    
    def create_pdf_file_object(self, pdf_data_or_b64: Any, filename: str = "document.pdf") -> Dict[str, str]:
        """Create PDF file object for API upload.

        Accepts either raw PDF bytes or a base64-encoded string and normalizes to base64.
        """
        b64: str
        if isinstance(pdf_data_or_b64, (bytes, bytearray)):
            b64 = base64.b64encode(pdf_data_or_b64).decode('utf-8')
        elif isinstance(pdf_data_or_b64, str):
            # If it's already base64, keep it; otherwise, encode as UTF-8 bytes (last resort)
            try:
                _ = base64.b64decode(pdf_data_or_b64, validate=True)
                b64 = pdf_data_or_b64
            except Exception:
                b64 = base64.b64encode(pdf_data_or_b64.encode('utf-8')).decode('utf-8')
        else:
            raise ValueError(f"Unsupported pdf_data type: {type(pdf_data_or_b64)}")

        return {
            "data": b64,
            "filename": filename,
            "mimeType": "application/pdf"
        }
    
    async def get_document(self, document_id: int) -> Dict[str, Any]:
        """Get document details by ID for verification."""
        self.logger.log_operation_start("get_document", {"document_id": document_id})
        
        try:
            response = await self.client.get(f"{self.base_url}/api/documents/{document_id}")
            
            if response.status_code == 200:
                result = {
                    "success": True,
                    "data": response.json()
                }
            else:
                result = {
                    "success": False,
                    "error": f"Document not found or access denied (HTTP {response.status_code})",
                    "status_code": response.status_code
                }
            
            self.logger.log_operation_success(
                "get_document",
                result,
                {"document_id": document_id}
            )
            
            return result
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code} error getting document: {e.response.text}"
            result = {
                "success": False,
                "error": error_msg,
                "status_code": e.response.status_code
            }
            self.logger.log_operation_failure("get_document", error_msg, {"document_id": document_id})
            return result
        except httpx.HTTPError as e:
            error_msg = f"HTTP error getting document: {e}"
            result = {
                "success": False,
                "error": error_msg
            }
            self.logger.log_operation_failure("get_document", error_msg, {"document_id": document_id})
            return result
        except Exception as e:
            error_msg = f"Unexpected error getting document: {e}"
            result = {
                "success": False,
                "error": error_msg
            }
            self.logger.log_operation_failure("get_document", error_msg, {"document_id": document_id})
            return result
    
    async def close(self):
        """Close HTTP client and JWT auth client."""
        await self.client.aclose()
        await self.auth.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
