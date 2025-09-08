# Programmatic Document Upload Solution

## Overview

This document outlines a comprehensive approach for creating programmatic document upload functionality that reuses all existing manual upload infrastructure while providing secure API access for external applications.

## Current Manual Upload Flow Analysis

### Existing Infrastructure
- **Frontend**: `pages/dashboard.js` with `UploadModal.js` component
- **API Endpoint**: `POST /api/upload` 
- **Authentication**: JWT tokens with SuperUser role validation
- **File Storage**: AWS S3 with structured prefixes (`content-images/`, `content-pdfs/`)
- **Database**: Polymorphic content system (Blog, Book, Product tables)
- **Field Processing**: Auto-generated slugs, genre relationships, file key management

### Current Upload Process
1. Admin logs in via dashboard with SuperUser credentials
2. Form submission with FormData (multipart/form-data)
3. Files uploaded to S3, keys stored in database
4. Database record created with relationships (Genre, Tags)
5. Automatic slug generation and uniqueness validation

## Proposed Solution Architecture

### 1. API Token Authentication System

Create a secure token-based authentication system specifically for programmatic access:

```javascript
// New endpoint: POST /api/auth/api-token
// Generates long-lived API tokens for external systems
```

**Implementation Approach:**
- Create `ApiToken` model in Prisma schema
- Generate secure tokens with configurable permissions
- Implement middleware to validate API tokens separately from user JWT
- Store token metadata (creation date, last used, permissions)

### 2. Enhanced Upload API Endpoint

Extend the existing `/api/upload` endpoint or create a new dedicated endpoint:

```javascript
// Enhanced: POST /api/upload/programmatic
// Or extend existing: POST /api/upload
```

**Key Features:**
- Support both FormData (multipart) and JSON payloads
- File upload via base64 encoding or URL references
- Batch upload capabilities for multiple documents
- Comprehensive validation and error handling

### 3. Required Implementation Components

#### A. New API Token Model
```prisma
model ApiToken {
  id          String    @id @default(cuid())
  name        String    // Human-readable token name
  token       String    @unique // Hashed token value
  permissions String[]  // e.g., ["upload", "read", "delete"]
  isActive    Boolean   @default(true)
  lastUsed    DateTime?
  createdAt   DateTime  @default(now())
  expiresAt   DateTime?
  @@map("api_token")
}
```

#### B. Authentication Middleware
- Token validation utility function
- Permission checking system
- Rate limiting and usage tracking
- Audit logging for security

#### C. Upload Service Layer
- Abstracted upload logic for reuse
- File processing utilities (validation, S3 upload)
- Database transaction management
- Error handling and rollback mechanisms

### 4. API Endpoint Specifications

#### Authentication Endpoint
```
POST /api/auth/api-token
Headers: Authorization: Bearer <admin-jwt-token>
Body: {
  "name": "External App Integration",
  "permissions": ["upload"],
  "expiresAt": "2025-12-31T23:59:59Z" // Optional
}
Response: { "token": "api_xxxxxxxxxxxx", "id": "token-id" }
```

#### Programmatic Upload Endpoint
```
POST /api/upload/programmatic
Headers: 
  Authorization: Bearer api_xxxxxxxxxxxx
  Content-Type: application/json

Body: {
  "title": "Document Title",
  "author": "Author Name",
  "category": "Blog|Book|Product", 
  "genreId": 1,
  "summary": "Optional summary",
  "content": "Text content (optional)",
  "coverImage": {
    "data": "base64-encoded-image-data",
    "filename": "cover.jpg",
    "mimeType": "image/jpeg"
  },
  "pdfFile": {
    "data": "base64-encoded-pdf-data", 
    "filename": "document.pdf",
    "mimeType": "application/pdf"
  }
}
```

#### Alternative: URL-based File Upload
```
POST /api/upload/from-urls
Body: {
  "title": "Document Title",
  "author": "Author Name", 
  "category": "Blog|Book|Product",
  "genreId": 1,
  "coverImageUrl": "https://example.com/image.jpg",
  "pdfUrl": "https://example.com/document.pdf"
}
```

### 5. Implementation Strategy

#### Phase 1: Core Infrastructure
1. **Create API Token System**
   - Add ApiToken model to Prisma schema
   - Implement token generation and validation
   - Create token management endpoints

2. **Authentication Middleware**
   - Token validation utility
   - Permission checking system
   - Integration with existing auth flow

#### Phase 2: Upload Service Abstraction
1. **Extract Upload Logic**
   - Create reusable upload service from existing `/api/upload`
   - Abstract file processing, S3 upload, and database operations
   - Implement transaction management

2. **File Processing Enhancement**
   - Support for base64 file uploads
   - URL-based file fetching
   - File validation and security checks

#### Phase 3: API Endpoints
1. **Programmatic Upload Endpoint**
   - New endpoint leveraging abstracted upload service
   - JSON payload support with base64 files
   - Comprehensive error handling

2. **Batch Upload Support**
   - Multiple document upload in single request
   - Transaction management for batch operations
   - Progress tracking and partial failure handling

#### Phase 4: Additional Features
1. **Webhook Support**
   - Optional webhooks for upload completion notifications
   - Configurable webhook endpoints per API token

2. **Enhanced Validation**
   - File type and size validation
   - Content scanning for security
   - Duplicate detection

### 6. Security Considerations

#### Token Security
- Secure token generation using cryptographically strong randomness
- Token hashing before database storage
- Configurable expiration times
- Token rotation capabilities

#### File Upload Security
- File type validation and whitelist enforcement
- File size limits
- Virus scanning integration (optional)
- S3 bucket security policies

#### Access Control
- Permission-based access control for API tokens
- Rate limiting per token
- Audit logging for all API operations
- IP restriction capabilities (optional)

### 7. Error Handling & Monitoring

#### Comprehensive Error Responses
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Genre ID is required",
    "details": {
      "field": "genreId",
      "value": null,
      "constraints": ["required", "integer"]
    }
  }
}
```

#### Monitoring & Observability
- API usage metrics per token
- Upload success/failure rates
- File processing times
- Storage usage tracking

### 8. External App Integration Examples

#### JavaScript/Node.js Example
```javascript
const uploadDocument = async (doc) => {
  const response = await fetch('https://digitaldossier.us/api/upload/programmatic', {
    method: 'POST',
    headers: {
      'Authorization': 'Bearer api_xxxxxxxxxxxx',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      title: doc.title,
      author: doc.author,
      category: 'Book',
      genreId: 1,
      coverImage: {
        data: doc.coverImageBase64,
        filename: 'cover.jpg',
        mimeType: 'image/jpeg'
      },
      pdfFile: {
        data: doc.pdfBase64,
        filename: 'document.pdf', 
        mimeType: 'application/pdf'
      }
    })
  });
  
  return response.json();
};
```

#### Python Example
```python
import requests
import base64

def upload_document(title, author, cover_path, pdf_path, genre_id=1):
    with open(cover_path, 'rb') as cover, open(pdf_path, 'rb') as pdf:
        payload = {
            'title': title,
            'author': author,
            'category': 'Book',
            'genreId': genre_id,
            'coverImage': {
                'data': base64.b64encode(cover.read()).decode(),
                'filename': 'cover.jpg',
                'mimeType': 'image/jpeg'
            },
            'pdfFile': {
                'data': base64.b64encode(pdf.read()).decode(),
                'filename': f'{title}.pdf',
                'mimeType': 'application/pdf'
            }
        }
    
    response = requests.post(
        'https://digitaldossier.us/api/upload/programmatic',
        json=payload,
        headers={'Authorization': 'Bearer api_xxxxxxxxxxxx'}
    )
    
    return response.json()
```

### 9. Benefits of This Approach

#### Reusability
- Leverages all existing upload infrastructure
- Maintains consistency with manual upload process
- Uses same validation rules and business logic
- Preserves file organization and naming conventions

#### Security
- Token-based authentication separate from user sessions
- Permission-based access control
- Audit trail for all programmatic uploads
- No exposure of user credentials

#### Scalability
- Supports high-volume uploads via API
- Batch processing capabilities
- Asynchronous processing options
- Integration with external workflow systems

#### Maintainability
- Single source of truth for upload logic
- Consistent error handling across manual and programmatic flows
- Centralized file management and S3 integration
- Easy to extend with additional features

### 10. Migration & Deployment Strategy

#### Development Phase
1. Implement API token system in development environment
2. Create abstracted upload service
3. Build and test programmatic upload endpoints
4. Validate with external app integration

#### Production Deployment
1. Database migration for ApiToken model
2. Deploy new API endpoints with feature flags
3. Generate initial API tokens for authorized external apps
4. Monitor and optimize based on usage patterns

### 11. Future Enhancements

#### Advanced Features
- Webhook notifications for upload events
- Bulk operations (update, delete)
- Content versioning support
- Metadata extraction from uploaded files

#### Integration Options
- SDKs for popular programming languages
- CLI tool for command-line uploads
- Zapier/automation platform integrations
- Direct database import tools

## Next Steps

1. Review and approve this solution approach
2. Create detailed implementation plan with timeline
3. Set up development environment for API token system
4. Begin implementation with Phase 1 components
5. Test integration with sample external application

This solution provides a robust, secure, and scalable approach for programmatic document uploads while maintaining full compatibility with existing manual upload functionality.