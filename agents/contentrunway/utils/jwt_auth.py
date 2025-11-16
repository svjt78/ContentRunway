"""JWT Authentication utility for DigitalDossier API."""

import os
import httpx
import json
from typing import Dict, Optional
from datetime import datetime, timedelta
from ..utils.publisher_logger import PublisherLogger


class DigitalDossierJWTAuth:
    """JWT-based authentication for DigitalDossier API with auto-renewal."""
    
    def __init__(self, base_url: str = None, admin_email: str = None, admin_password: str = None):
        self.logger = PublisherLogger()
        
        # Configuration from environment or parameters
        self.base_url = base_url or os.getenv('DIGITALDOSSIER_BASE_URL', 'http://localhost:3003')
        self.admin_email = admin_email or os.getenv('DIGITALDOSSIER_ADMIN_EMAIL')
        self.admin_password = admin_password or os.getenv('DIGITALDOSSIER_ADMIN_PASSWORD')
        
        # Validate required credentials
        self._validate_config()
        
        # JWT state
        self._jwt_token = None
        self._jwt_expires_at = None
        
        # HTTP client for authentication requests
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    
    def _validate_config(self):
        """Validate required environment variables for JWT authentication."""
        if not self.admin_email:
            raise ValueError("DIGITALDOSSIER_ADMIN_EMAIL environment variable is required for JWT authentication")
        
        if not self.admin_password:
            raise ValueError("DIGITALDOSSIER_ADMIN_PASSWORD environment variable is required for JWT authentication")
        
        if not self.base_url:
            raise ValueError("DIGITALDOSSIER_BASE_URL environment variable is required")
    
    async def get_authenticated_headers(self) -> Dict[str, str]:
        """Get headers with valid JWT token, auto-renewing if necessary."""
        token = await self._ensure_valid_jwt()
        return {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
    
    async def _ensure_valid_jwt(self) -> str:
        """Ensure JWT is valid, refresh if needed."""
        operation_context = {
            "base_url": self.base_url,
            "admin_email": self.admin_email,
            "has_current_token": bool(self._jwt_token)
        }
        
        if self._needs_renewal():
            self.logger.log_operation_start("jwt_renewal", operation_context)
            await self._authenticate()
            self.logger.log_operation_success(
                "jwt_renewal",
                {
                    "new_expires_at": self._jwt_expires_at.isoformat() if self._jwt_expires_at else None,
                    "token_length": len(self._jwt_token) if self._jwt_token else 0
                },
                operation_context
            )
        
        return self._jwt_token
    
    def _needs_renewal(self) -> bool:
        """Check if JWT needs renewal (5 minute buffer for safety)."""
        if not self._jwt_token or not self._jwt_expires_at:
            return True
        
        # Renew 5 minutes before expiry to avoid edge cases
        buffer_time = timedelta(minutes=5)
        return datetime.now() > (self._jwt_expires_at - buffer_time)
    
    async def _authenticate(self) -> None:
        """Login to DigitalDossier and get fresh JWT token."""
        operation_context = {
            "login_url": f"{self.base_url}/api/auth/login",
            "admin_email": self.admin_email
        }
        
        try:
            self.logger.log_operation_start("jwt_login", operation_context)
            
            # Prepare login payload
            login_payload = {
                "email": self.admin_email,
                "password": self.admin_password
            }
            
            # Make login request
            self.logger.log_info("jwt_login", f"🔧 DEBUG: Attempting login to {self.base_url}/api/auth/login")
            
            response = await self.client.post(
                f"{self.base_url}/api/auth/login",
                json=login_payload,
                headers={"Content-Type": "application/json"}
            )
            
            self.logger.log_info("jwt_login", f"🔧 DEBUG: Login response status: {response.status_code}")
            self.logger.log_info("jwt_login", f"🔧 DEBUG: Login response headers: {dict(response.headers)}")
            
            response.raise_for_status()
            auth_data = response.json()
            
            self.logger.log_info("jwt_login", f"🔧 DEBUG: Auth response keys: {list(auth_data.keys())}")
            
            # Extract JWT token
            self._jwt_token = auth_data.get('access_token')
            self.logger.log_info("jwt_login", f"🔧 DEBUG: JWT token obtained: {bool(self._jwt_token)}")
            if not self._jwt_token:
                raise ValueError("No access_token in login response")
            
            # Decode JWT expiry
            self._jwt_expires_at = self._decode_jwt_expiry(self._jwt_token)
            
            self.logger.log_operation_success(
                "jwt_login",
                {
                    "token_received": bool(self._jwt_token),
                    "expires_at": self._jwt_expires_at.isoformat() if self._jwt_expires_at else None,
                    "user_id": auth_data.get('user_id'),
                    "email": auth_data.get('email')
                },
                operation_context
            )
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code} error during JWT login: {e.response.text}"
            self.logger.log_operation_failure("jwt_login", error_msg, operation_context)
            raise Exception(error_msg)
        except httpx.HTTPError as e:
            error_msg = f"HTTP error during JWT login: {e}"
            self.logger.log_operation_failure("jwt_login", error_msg, operation_context)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"JWT authentication failed: {e}"
            self.logger.log_operation_failure("jwt_login", error_msg, operation_context)
            raise Exception(error_msg)
    
    def _decode_jwt_expiry(self, token: str) -> Optional[datetime]:
        """Extract expiration timestamp from JWT payload."""
        try:
            # Split JWT and decode payload (base64)
            import base64
            
            # JWT format: header.payload.signature
            parts = token.split('.')
            if len(parts) != 3:
                self.logger.log_warning("jwt_decode", "Invalid JWT format", {"token_parts": len(parts)})
                return None
            
            # Decode payload (add padding if needed)
            payload_b64 = parts[1]
            # Add padding for base64 decoding
            padding = 4 - (len(payload_b64) % 4)
            if padding != 4:
                payload_b64 += '=' * padding
            
            payload_bytes = base64.urlsafe_b64decode(payload_b64)
            payload = json.loads(payload_bytes.decode('utf-8'))
            
            # Extract expiry timestamp
            exp_timestamp = payload.get('exp')
            if exp_timestamp:
                expiry = datetime.fromtimestamp(exp_timestamp)
                self.logger.log_info(
                    "jwt_decode",
                    f"JWT expires at: {expiry.isoformat()}",
                    {"exp_timestamp": exp_timestamp}
                )
                return expiry
            
            self.logger.log_warning("jwt_decode", "No expiry found in JWT payload", payload)
            return None
            
        except Exception as e:
            self.logger.log_error("jwt_decode", f"Failed to decode JWT expiry: {e}")
            # Return a default expiry (1 hour from now) if decoding fails
            return datetime.now() + timedelta(hours=1)
    
    async def test_authentication(self) -> Dict[str, any]:
        """Test JWT authentication by making a simple API call."""
        operation_context = {"test_endpoint": f"{self.base_url}/api/genres"}
        
        try:
            self.logger.log_operation_start("jwt_auth_test", operation_context)
            
            headers = await self.get_authenticated_headers()
            
            response = await self.client.get(
                f"{self.base_url}/api/genres",
                headers=headers
            )
            
            if response.status_code == 200:
                genres = response.json()
                result = {
                    "status": "success",
                    "message": "JWT authentication working",
                    "genres_count": len(genres),
                    "expires_at": self._jwt_expires_at.isoformat() if self._jwt_expires_at else None
                }
                
                self.logger.log_operation_success("jwt_auth_test", result, operation_context)
                return result
            else:
                error_msg = f"Authentication test failed with status {response.status_code}: {response.text}"
                self.logger.log_operation_failure("jwt_auth_test", error_msg, operation_context)
                return {
                    "status": "error",
                    "message": error_msg,
                    "status_code": response.status_code
                }
                
        except Exception as e:
            error_msg = f"JWT authentication test failed: {e}"
            self.logger.log_operation_failure("jwt_auth_test", error_msg, operation_context)
            return {
                "status": "error",
                "message": error_msg
            }
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    def get_token_status(self) -> Dict[str, any]:
        """Get current JWT token status for monitoring."""
        if not self._jwt_token or not self._jwt_expires_at:
            return {
                "has_token": False,
                "expires_at": None,
                "expires_in_minutes": None,
                "needs_renewal": True
            }
        
        now = datetime.now()
        expires_in = self._jwt_expires_at - now
        expires_in_minutes = int(expires_in.total_seconds() / 60)
        
        return {
            "has_token": True,
            "expires_at": self._jwt_expires_at.isoformat(),
            "expires_in_minutes": expires_in_minutes,
            "needs_renewal": self._needs_renewal(),
            "token_length": len(self._jwt_token) if self._jwt_token else 0
        }