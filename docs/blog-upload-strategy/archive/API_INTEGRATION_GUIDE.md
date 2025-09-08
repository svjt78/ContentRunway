# API Integration Guide for Programmatic Document Upload

## Overview

This guide provides comprehensive instructions for external applications to programmatically upload documents (blogs, books, products) to the Digital Dossier platform using secure API endpoints.

## Authentication

### 1. API Token Generation

**Endpoint:** `POST /api/auth/api-token`

**Headers:**
```
Authorization: Bearer <admin-jwt-token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "name": "External App Integration",
  "permissions": ["upload", "read", "delete"],
  "expiresAt": "2025-12-31T23:59:59Z"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "token": "api_xxxxxxxxxxxxxxxxxxxxxxxx",
    "id": "clx1234567890",
    "name": "External App Integration",
    "permissions": ["upload", "read", "delete"],
    "expiresAt": "2025-12-31T23:59:59Z"
  }
}
```

### 2. API Token Usage

Include the API token in all subsequent requests:

```
Authorization: Bearer api_xxxxxxxxxxxxxxxxxxxxxxxx
```

## Content Categories & Requirements

### Available Categories
- **Blog**: Text-focused content with articles and posts
- **Book**: Long-form content with PDF documents  
- **Product**: Product-related content and documentation

### Required Fields for All Categories
- `title` (string, required): Document title
- `author` (string, optional): Author name
- `category` (string, required): "Blog", "Book", or "Product"
- `genreId` (integer, required): Genre ID from genre reference table
- `coverImage` (object, required): Cover image file data
- `pdfFile` (object, optional): PDF document file data

### Optional Fields
- `content` (string): Text content (auto-generated from PDF if not provided)
- `summary` (string): Brief description (auto-generated if not provided)
- `tags` (array): Tag names for categorization

## Genre Reference Data

### Get Available Genres

**Endpoint:** `GET /api/genres`

**Headers:**
```
Authorization: Bearer api_xxxxxxxxxxxxxxxxxxxxxxxx
```

**Response:**
```json
[
  { "id": 1, "name": "AI" },
  { "id": 2, "name": "Insurance" },
  { "id": 3, "name": "Technology" },
  { "id": 4, "name": "Finance" },
  { "id": 5, "name": "Healthcare" }
]
```

**Important:** Always use the exact `genreId` from this endpoint. Genre relationships are enforced at the database level.

## File Upload Specifications

### File Format Requirements

**Cover Images:**
- **Formats**: JPEG, PNG
- **Size Limit**: 10MB
- **Recommended Dimensions**: 800x1200px (book cover aspect ratio)

**PDF Documents:**
- **Format**: PDF only
- **Size Limit**: 50MB
- **Content**: Will be indexed for search functionality

### Base64 Encoding Format

Files must be base64-encoded with metadata:

```json
{
  "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
  "filename": "cover.jpg",
  "mimeType": "image/jpeg"
}
```

## API Endpoints

### 1. Single Document Upload

**Endpoint:** `POST /api/upload/programmatic`

**Headers:**
```
Authorization: Bearer api_xxxxxxxxxxxxxxxxxxxxxxxx
Content-Type: application/json
```

**Request Body:**
```json
{
  "title": "Introduction to Machine Learning",
  "author": "Dr. Jane Smith",
  "category": "Book",
  "genreId": 1,
  "summary": "A comprehensive guide to machine learning fundamentals",
  "content": "Optional text content here...",
  "coverImage": {
    "data": "base64-encoded-image-data",
    "filename": "ml-book-cover.jpg",
    "mimeType": "image/jpeg"
  },
  "pdfFile": {
    "data": "base64-encoded-pdf-data",
    "filename": "ml-introduction.pdf",
    "mimeType": "application/pdf"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": 123,
    "title": "Introduction to Machine Learning",
    "slug": "introduction-to-machine-learning",
    "author": "Dr. Jane Smith",
    "category": "Book",
    "genreId": 1,
    "genre": "AI",
    "summary": "A comprehensive guide to machine learning fundamentals",
    "coverKey": "content-images/clx123_cover.jpg",
    "pdfKey": "content-pdfs/clx123_document.pdf",
    "coverUrl": "https://bucket.s3.region.amazonaws.com/content-images/clx123_cover.jpg",
    "pdfUrl": "https://bucket.s3.region.amazonaws.com/content-pdfs/clx123_document.pdf",
    "createdAt": "2025-09-02T20:41:28.000Z",
    "updatedAt": "2025-09-02T20:41:28.000Z"
  }
}
```

### 2. Batch Document Upload

**Endpoint:** `POST /api/upload/programmatic/batch`

**Headers:**
```
Authorization: Bearer api_xxxxxxxxxxxxxxxxxxxxxxxx
Content-Type: application/json
```

**Request Body:**
```json
{
  "documents": [
    {
      "title": "AI Ethics Guide",
      "author": "Dr. John Doe",
      "category": "Book",
      "genreId": 1,
      "coverImage": { "data": "base64-data", "filename": "ethics.jpg", "mimeType": "image/jpeg" },
      "pdfFile": { "data": "base64-data", "filename": "ethics.pdf", "mimeType": "application/pdf" }
    },
    {
      "title": "Insurance Innovation",
      "author": "Mary Johnson",
      "category": "Blog", 
      "genreId": 2,
      "coverImage": { "data": "base64-data", "filename": "innovation.jpg", "mimeType": "image/jpeg" },
      "content": "Text content for blog post..."
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "success": true,
      "data": { /* document 1 data */ }
    },
    {
      "success": true, 
      "data": { /* document 2 data */ }
    }
  ],
  "summary": {
    "total": 2,
    "successful": 2,
    "failed": 0
  }
}
```

## Implementation Examples

### JavaScript/Node.js

```javascript
class DigitalDossierAPI {
  constructor(apiToken, baseUrl = 'http://localhost:3003') {
    this.apiToken = apiToken;
    this.baseUrl = baseUrl;
  }

  async getGenres() {
    const response = await fetch(`${this.baseUrl}/api/genres`, {
      headers: { 'Authorization': `Bearer ${this.apiToken}` }
    });
    return response.json();
  }

  async uploadSingleDocument(document) {
    const response = await fetch(`${this.baseUrl}/api/upload/programmatic`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(document)
    });
    return response.json();
  }

  async uploadBatchDocuments(documents) {
    const response = await fetch(`${this.baseUrl}/api/upload/programmatic/batch`, {
      method: 'POST', 
      headers: {
        'Authorization': `Bearer ${this.apiToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ documents })
    });
    return response.json();
  }

  // Helper function to convert file to base64
  async fileToBase64(filePath) {
    const fs = require('fs');
    const file = fs.readFileSync(filePath);
    return file.toString('base64');
  }
}

// Usage Example
const api = new DigitalDossierAPI('api_xxxxxxxxxxxxxxxxxxxxxxxx');

const uploadExample = async () => {
  // Get available genres
  const genres = await api.getGenres();
  const aiGenreId = genres.find(g => g.name === 'AI')?.id;

  // Upload single document
  const coverBase64 = await api.fileToBase64('./cover.jpg');
  const pdfBase64 = await api.fileToBase64('./document.pdf');

  const result = await api.uploadSingleDocument({
    title: 'AI Innovation Report',
    author: 'Tech Team',
    category: 'Book',
    genreId: aiGenreId,
    coverImage: {
      data: coverBase64,
      filename: 'ai-report-cover.jpg',
      mimeType: 'image/jpeg'
    },
    pdfFile: {
      data: pdfBase64,
      filename: 'ai-innovation-report.pdf',
      mimeType: 'application/pdf'
    }
  });

  console.log('Upload result:', result);
};
```

### Python

```python
import requests
import base64
import json

class DigitalDossierAPI:
    def __init__(self, api_token, base_url='http://localhost:3003'):
        self.api_token = api_token
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {api_token}'}

    def get_genres(self):
        response = requests.get(f'{self.base_url}/api/genres', headers=self.headers)
        response.raise_for_status()
        return response.json()

    def upload_single_document(self, document):
        response = requests.post(
            f'{self.base_url}/api/upload/programmatic',
            headers={**self.headers, 'Content-Type': 'application/json'},
            json=document
        )
        response.raise_for_status()
        return response.json()

    def upload_batch_documents(self, documents):
        response = requests.post(
            f'{self.base_url}/api/upload/programmatic/batch',
            headers={**self.headers, 'Content-Type': 'application/json'},
            json={'documents': documents}
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def file_to_base64(file_path):
        with open(file_path, 'rb') as file:
            return base64.b64encode(file.read()).decode('utf-8')

# Usage Example
api = DigitalDossierAPI('api_xxxxxxxxxxxxxxxxxxxxxxxx')

# Get available genres
genres = api.get_genres()
ai_genre_id = next((g['id'] for g in genres if g['name'] == 'AI'), None)

# Upload single document
with open('cover.jpg', 'rb') as cover, open('document.pdf', 'rb') as pdf:
    document = {
        'title': 'Python AI Guide',
        'author': 'Python Team',
        'category': 'Book',
        'genreId': ai_genre_id,
        'coverImage': {
            'data': base64.b64encode(cover.read()).decode(),
            'filename': 'python-ai-cover.jpg',
            'mimeType': 'image/jpeg'
        },
        'pdfFile': {
            'data': base64.b64encode(pdf.read()).decode(),
            'filename': 'python-ai-guide.pdf',
            'mimeType': 'application/pdf'
        }
    }

result = api.upload_single_document(document)
print(f"Uploaded: {result['data']['title']} -> {result['data']['slug']}")
```

## Error Handling

### Common Error Responses

**Authentication Error:**
```json
{
  "success": false,
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or expired API token"
  }
}
```

**Validation Error:**
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

**File Upload Error:**
```json
{
  "success": false,
  "error": {
    "code": "FILE_UPLOAD_ERROR",
    "message": "Invalid file format for cover image",
    "details": {
      "field": "coverImage",
      "allowedTypes": ["image/jpeg", "image/png"]
    }
  }
}
```

### Batch Upload Error Handling

Batch uploads return individual results for each document:

```json
{
  "success": true,
  "results": [
    {
      "success": true,
      "data": { /* successful upload data */ }
    },
    {
      "success": false,
      "error": {
        "code": "VALIDATION_ERROR",
        "message": "Title is required"
      }
    }
  ],
  "summary": {
    "total": 2,
    "successful": 1,
    "failed": 1
  }
}
```

## Data Validation Rules

### Required Validations
1. **Title**: Non-empty string, max 255 characters
2. **Category**: Must be exactly "Blog", "Book", or "Product"
3. **GenreId**: Must exist in genre table
4. **CoverImage**: Required for all uploads, valid image format
5. **PdfFile**: Required for Book and Product categories

### Automatic Processing
1. **Slug Generation**: Auto-generated from title with uniqueness validation
2. **File Keys**: Unique S3 keys generated automatically
3. **Timestamps**: CreatedAt and UpdatedAt managed automatically
4. **Content Processing**: PDF text extraction if content not provided

### File Size Limits
- **Cover Images**: Maximum 10MB
- **PDF Files**: Maximum 50MB
- **Base64 Encoding**: Increases payload size by ~33%

## Rate Limits & Best Practices

### Rate Limiting
- **Single Upload**: 60 requests per minute per API token
- **Batch Upload**: 10 requests per minute per API token
- **Batch Size**: Maximum 20 documents per batch request

### Best Practices
1. **Genre Caching**: Cache genre list to avoid repeated API calls
2. **Error Retry**: Implement exponential backoff for temporary failures
3. **File Validation**: Validate files locally before upload
4. **Batch Optimization**: Use batch endpoint for multiple documents
5. **Token Security**: Store API tokens securely, never in client-side code

## Complete Integration Workflow

### Step 1: Setup
```javascript
// 1. Get API token from admin
const apiToken = 'api_xxxxxxxxxxxxxxxxxxxxxxxx';

// 2. Initialize API client
const api = new DigitalDossierAPI(apiToken);

// 3. Cache available genres
const genres = await api.getGenres();
```

### Step 2: Prepare Document Data
```javascript
// 4. Prepare document with required fields
const document = {
  title: 'Document Title',
  author: 'Author Name',
  category: 'Book', // Blog, Book, or Product
  genreId: 1, // From genres API
  coverImage: {
    data: fileToBase64('./cover.jpg'),
    filename: 'cover.jpg',
    mimeType: 'image/jpeg'
  },
  pdfFile: {
    data: fileToBase64('./document.pdf'),
    filename: 'document.pdf',
    mimeType: 'application/pdf'
  }
};
```

### Step 3: Upload
```javascript
// 5. Upload single document
const result = await api.uploadSingleDocument(document);

// Or upload multiple documents
const batchResult = await api.uploadBatchDocuments([doc1, doc2, doc3]);
```

### Step 4: Handle Response
```javascript
// 6. Process upload result
if (result.success) {
  console.log(`Successfully uploaded: ${result.data.title}`);
  console.log(`Access URL: ${api.baseUrl}/${result.data.category.toLowerCase()}s/${result.data.slug}`);
} else {
  console.error('Upload failed:', result.error.message);
}
```

## Security Considerations

### API Token Security
- Store tokens in secure environment variables
- Never commit tokens to version control
- Rotate tokens regularly (quarterly recommended)
- Use minimum required permissions per token

### File Upload Security
- Validate file types before encoding
- Check file sizes before upload
- Sanitize filenames for security
- Monitor upload patterns for abuse

### Network Security
- Always use HTTPS endpoints
- Implement request signing for additional security
- Use proper TLS certificate validation
- Monitor for suspicious API usage patterns

## Troubleshooting

### Common Issues

**1. Genre ID Not Found**
```
Error: Genre ID 99 does not exist
Solution: Call GET /api/genres to get valid genre IDs
```

**2. File Too Large** 
```
Error: File size exceeds maximum limit
Solution: Compress files or split into smaller chunks
```

**3. Invalid Base64 Data**
```
Error: Invalid base64 encoding
Solution: Ensure proper base64 encoding without headers
```

**4. Token Expired**
```
Error: API token has expired
Solution: Generate new token via admin dashboard
```

### Debug Mode

Enable debug logging in your external app to troubleshoot issues:

```javascript
// Add debug headers to requests
headers: {
  'Authorization': `Bearer ${apiToken}`,
  'X-Debug-Mode': 'true'
}
```

## Monitoring & Observability

### Success Metrics
- Track upload success rates
- Monitor file processing times
- Log API response times
- Monitor storage usage

### Error Tracking
- Log all API errors with context
- Track token usage patterns
- Monitor failed upload attempts
- Alert on rate limit violations

## Support

For integration support and troubleshooting:
1. Check this guide for common issues
2. Review API error responses for specific details
3. Contact system administrators for token-related issues
4. Monitor application logs for detailed error information

---

**Generated for Digital Dossier Platform**  
*Last updated: September 2025*