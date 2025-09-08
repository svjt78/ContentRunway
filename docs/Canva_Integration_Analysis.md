# Canva Integration Analysis & Implementation Plan

## Overview
This document outlines the research findings and implementation approach for integrating Canva with ContentRunway's publishing agent to automatically generate cover pages and formatted templates for blog content.

## Research Findings

### Canva Connect API Capabilities (2025)

#### Authentication
- **Method**: OAuth 2.0 with Authorization Code flow + PKCE (SHA-256)
- **Requirements**: 
  - Canva Pro account with MFA enabled
  - Developer integration created in Canva Developer Portal
  - Backend-only authentication (client secret cannot be exposed to browser)
- **Scopes Needed**: 
  - `design:content:read` - Read design content
  - `design:content:write` - Create and modify designs
  - `asset:read` - Access assets
  - `asset:write` - Upload assets

#### Design Creation & Manipulation
- **Create Designs**: POST `/v1/designs` with preset types or custom dimensions
- **Preset Types**: `doc`, `whiteboard`, `presentation`
- **Custom Dimensions**: 40-8000 pixels (width/height)
- **Content Modification**: Can insert assets and modify text elements
- **Rate Limits**: 20 requests per minute per user

#### Export Capabilities
- **Supported Formats**: PDF, PNG, JPG, GIF, PPTX, MP4
- **Export Process**: Asynchronous job system via `/v1/exports`
- **Job Status**: `in_progress`, `success`, `failed`
- **Download Links**: Valid for 24 hours
- **PNG Options**: Configurable dimensions, compression, transparent background (Pro feature)
- **PDF Options**: Multi-page support, quality settings

### Starred Templates Access Challenge

#### Current Limitation
- **No Direct API**: Canva Connect API does not provide endpoints for accessing starred/favorites folders
- **Available Endpoints**: Only general design listing (`/v1/designs`) and folder contents

#### Workaround Solutions
1. **Manual Template ID Collection**: Collect design IDs from starred folder manually
2. **Dedicated Template Folder**: Create "ContentRunway Templates" folder accessible via API
3. **Hybrid Approach**: Combine folder API with content-based template selection

### Current Publishing Infrastructure

#### Existing Components
- **Publishing Agent**: `/langgraph/contentrunway/agents/publishing.py`
- **Publishing API Tool**: `/langgraph/contentrunway/tools/publishing_api_tool.py`
- **Publishing Endpoint**: `/backend/app/api/endpoints/publishing.py`

#### Current Blog Integration
- Uses webhook-based publishing to personal blog
- Supports markdown/HTML content
- Includes metadata, tags, and categorization
- Rate limiting and error handling implemented

## Implementation Plan

### Phase 1: Canva Infrastructure Setup

#### 1.1 Developer Access Verification
- Test access to [www.canva.com/developers/](https://www.canva.com/developers/)
- Create private integration in Canva Developer Portal
- Configure OAuth redirect URLs for local development

#### 1.2 Authentication Service
**New File**: `/backend/app/services/canva_auth_service.py`
```python
class CanvaAuthService:
    - OAuth 2.0 + PKCE implementation
    - Token management (access/refresh)
    - Secure credential storage
    - Authentication status checking
```

#### 1.3 Core Canva Service
**New File**: `/backend/app/services/canva_service.py`
```python
class CanvaService:
    - API client with authentication
    - Design creation and modification
    - Export job management
    - File download handling
    - Error handling and retries
```

### Phase 2: Template Management Solution

#### 2.1 Template Selection Strategy
Given API limitations, implement hybrid approach:

**Option A**: Manual Template Mapping
```python
CONTENT_TEMPLATES = {
    'it_insurance': ['design_id_1', 'design_id_2'],
    'ai_research': ['design_id_3', 'design_id_4'], 
    'agentic_ai': ['design_id_5', 'design_id_6']
}
```

**Option B**: Folder-Based Templates
- Create "ContentRunway-Templates" folder in Canva
- Use folder API to list available templates
- Implement content-domain mapping logic

#### 2.2 Template Selection Logic
**New File**: `/backend/app/services/template_selector.py`
```python
class TemplateSelector:
    - Content domain analysis
    - Template matching algorithm
    - Fallback template handling
    - Template metadata caching
```

### Phase 3: Content Processing & Design Generation

#### 3.1 Content Analysis
- Extract key themes, domain, and visual requirements from content
- Determine appropriate template category
- Generate cover page text and metadata

#### 3.2 Design Creation Process
1. Select appropriate template based on content domain
2. Create new design from template
3. Remove all existing text elements
4. Insert new content-specific text
5. Preserve existing images and visual elements
6. Generate both cover (PNG) and full template (PDF)

#### 3.3 Content Insertion Strategy
**New File**: `/backend/app/services/content_inserter.py`
```python
class ContentInserter:
    - Text element identification and removal
    - Dynamic text insertion with formatting
    - Layout preservation
    - Image element preservation
    - Error handling for content overflow
```

### Phase 4: Export & Download Management

#### 4.1 Export Job Orchestration
- Create export jobs for PNG (cover) and PDF (template)
- Monitor job status with polling
- Handle export failures gracefully
- Implement timeout handling

#### 4.2 File Management
**New File**: `/backend/app/services/file_manager.py`
```python
class FileManager:
    - Download exported designs
    - Temporary file storage
    - File cleanup after upload
    - Format validation
    - File size optimization
```

### Phase 5: Publishing Integration

#### 5.1 Enhanced Publishing Agent
**Update**: `/langgraph/contentrunway/agents/publishing.py`
- Add Canva generation step before publishing
- Integrate with existing publishing workflow
- Handle Canva generation failures
- Pass generated assets to publishing tools

#### 5.2 Publishing API Enhancement
**Update**: `/langgraph/contentrunway/tools/publishing_api_tool.py`
- Extend blog upload to handle file attachments
- Add support for cover images and PDF attachments
- Update webhook payload structure
- Implement multipart file upload

#### 5.3 Publishing Endpoint Updates
**Update**: `/backend/app/api/endpoints/publishing.py`
- Add Canva generation orchestration
- File upload handling for generated assets
- Status tracking for design creation process
- Error handling and fallback to text-only publishing

### Phase 6: Configuration & Dependencies

#### 6.1 Environment Configuration
```bash
# Add to .env
CANVA_CLIENT_ID=your_client_id
CANVA_CLIENT_SECRET=your_client_secret
CANVA_REDIRECT_URI=http://localhost:8000/auth/canva/callback
CANVA_TEMPLATE_FOLDER_ID=folder_id_or_manual_mapping
```

#### 6.2 Dependencies
```python
# Add to requirements.txt
aiohttp>=3.9.0
aiofiles>=23.0.0
pillow>=10.0.0  # For image processing if needed
```

## Workflow Integration

### Complete Publishing Flow
1. **Content Analysis**: Determine domain and template requirements
2. **Template Selection**: Choose appropriate template from starred/configured set
3. **Design Creation**: Create new design from selected template
4. **Content Processing**: Remove existing text, insert new content
5. **Export Generation**: Create PNG cover and PDF template exports
6. **File Download**: Download generated assets to temporary storage
7. **Blog Publishing**: Upload content with generated cover and PDF attachments
8. **Cleanup**: Remove temporary files after successful upload

### Error Handling Strategy
- **Canva API Failures**: Fallback to text-only publishing
- **Export Timeouts**: Retry mechanism with exponential backoff
- **Download Failures**: Alternative export format attempts
- **Upload Failures**: Separate retry for blog upload with assets

## Technical Considerations

### Rate Limiting
- Canva API: 20 requests/minute for design operations
- Export API: 20 requests/minute for export jobs
- Implement queue system for bulk content processing

### Security
- Store Canva credentials securely (environment variables/secrets manager)
- Implement token refresh mechanism
- Validate all downloaded files before upload
- Sanitize content before insertion into designs

### Performance
- Parallel processing where possible
- Cache template metadata
- Optimize file download/upload operations
- Implement progress tracking for long operations

## Next Steps Required

1. **Canva Developer Access**: Verify access and create integration
2. **Blog API Specifications**: Receive detailed blog upload API documentation
3. **Template Strategy Decision**: Choose between manual template mapping or folder-based approach
4. **Authentication Setup**: Configure OAuth flow in development environment

## Questions for User

1. Can you access the Canva Developer Portal and create an integration?
2. What are the specific blog upload API endpoints and payload structure?
3. Would you prefer to manually specify template IDs or create a dedicated template folder?
4. Do you want the system to generate cover pages for all content or only specific types?

## Estimated Implementation Timeline

- **Phase 1-2**: 2-3 days (Setup + Template Management)
- **Phase 3-4**: 3-4 days (Content Processing + Export)
- **Phase 5-6**: 2-3 days (Publishing Integration + Configuration)
- **Total**: 7-10 days for complete integration

---

*Generated: 2025-09-02*
*Status: Analysis Complete - Awaiting Implementation Approval*