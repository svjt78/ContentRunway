# ContentRunway Publisher Agent - Complete Implementation Plan

## Executive Summary

This document outlines the complete implementation plan for enhancing the ContentRunway Publisher Agent to automatically upload content to digitaldossier.us platform. The enhancement includes automatic category detection, title generation, cover image selection with text removal, PDF generation, and API integration with comprehensive error handling.

## Initial Requirements Analysis

### Core Requirements
1. **PDF Generation**: Contents must be published as PDF format
2. **Cover Image as PNG**: Cover image must be provided as PNG format
3. **Automatic Category Detection**: Category must be determined automatically using OpenAI LLM
4. **Automatic Title Generation**: Title must be determined automatically using OpenAI LLM
5. **Local Cover Image Selection**: Cover images picked from `@docs/cover-image/` folder structure instead of Canva
6. **DigitalDossier Upload**: Content uploaded to digitaldossier.us using POST API
7. **Upload Strategy Implementation**: Follow artifacts in `@docs/blog-upload-strategy/` for detailed strategy
8. **Agent Architecture**: Create sub-agents and tools as appropriate

## API Analysis

### API Endpoints Identified

**Base URLs:**
- Test Environment: `http://localhost:3003`
- Production: `https://digitaldossier.us`

**Endpoints:**
1. **GET** `/api/genres` - Fetch available genres
2. **POST** `/api/upload/programmatic` - Single document upload
3. **POST** `/api/upload/programmatic/batch` - Batch document upload

### Authentication
- All requests require `Authorization: Bearer {api_token}` header
- API token must be obtained from admin dashboard

### Required Upload Fields

**Mandatory Fields:**
- `title` (string) - Document title
- `author` (string) - Author name (fixed: "Suvojit Dutta")
- `category` (string) - Must be "Blog", "Book", or "Product" (only Blog/Product for our use case)
- `genreId` (integer) - From genres API response
- `coverImage` (object, required) - Cover image file data

**Cover Image Object:**
```javascript
{
  data: "base64-encoded-image-data",
  filename: "cover.png", 
  mimeType: "image/png"
}
```

**PDF File Object:**
```javascript
{
  data: "base64-encoded-pdf-data",
  filename: "document.pdf",
  mimeType: "application/pdf"
}
```

**Optional Fields:**
- `summary` (string) - Brief description
- `content` (string) - Text content

## Q&A Session - Requirements Clarification

### 1. Publisher Agent Input Source
**Q:** Should the publisher receive from writing agent or editing agent?  
**A:** Analysis of `@langgraph/contentrunway/pipeline.py` shows the sequence includes editing and critique agents. Publisher can work on any content that passes quality gates without requiring human intervention. Human intervention only when required.

### 2. Cover Image Text Removal
**Q:** Should I implement automatic text removal or assume images are text-free?  
**A:** Need to implement image processing to automatically remove text from cover images while preserving graphics.

### 3. Genre Mapping Logic
**Q:** How should we map ContentRunway domains to digitaldossier.us genres?  
**A:** Use content analysis to pick the best matching genre. If no match found, create a new relevant genre and send for upload.

### 4. Pipeline State Integration
**Q:** Which files should I analyze for pipeline integration?  
**A:** `@langgraph/contentrunway/pipeline.py` - Must be 100% aligned with current agent orchestration system.

### 5. Dashboard Integration
**Q:** Is there existing logging/dashboard system?  
**A:** No existing log file system. Refer to `@frontend/src/components/` for dashboard and frontend systems.

### 6. Environment Configuration
**Q:** What should be the exact environment variable names?  
**A:** Use these exact names:
- `DIGITALDOSSIER_API_TOKEN`
- `DIGITALDOSSIER_BASE_URL` 
- `DIGITALDOSSIER_ADMIN_EMAIL`
- `DIGITALDOSSIER_ADMIN_PASSWORD`

### 7. Cover Image Usage Tracking
**Q:** Should usage tracking be persistent?  
**A:** For this phase, ignore repetition problem mitigation use case.

### 8. Additional Clarifications
- **Author Name**: Fixed as "Suvojit Dutta" for all content
- **Title Generation**: Generate 4 options and pick the best one
- **Category Classification**: Only classify between Blog and Product (no Book category)
  - Product: If article describes or talks about specific product/platform
  - Blog: All other content
- **Cover Image Selection**: Category-based (Blog folder for Blog, Product folder for Product)
- **Error Handling**: Maintain log file with timestamps, visible in dashboard, no retry logic
- **Base URL Configuration**: Test environment uses `http://localhost:3003`

## Pipeline Integration Analysis

### Current Pipeline Flow (from pipeline.py)
```
Research → Curation → SEO Strategy → Writing → Quality Gates → Editing → Critique → Formatting → Human Review → Publishing → Completion
```

### Publisher Agent Integration Points
- **Input**: Receives `ContentPipelineState` from pipeline (line 527-530 in pipeline.py)
- **State Access**: 
  - `state["channel_drafts"]` from formatting step
  - `state["draft"]` with complete content
  - `state` object with full pipeline context
- **State Updates**: Must update:
  - `state["publishing_results"]` with API response
  - `state["published_urls"]` with digitaldossier.us URLs
  - `state["progress_percentage"] = 95.0`
  - `state["step_history"].append("publishing_completed")`
- **Error Handling**: Use `state["error_message"]` and `state["status"] = "failed"`

## Technical Architecture

### Main Publisher Agent
**File**: `langgraph/contentrunway/agents/publisher.py`

**Workflow:**
1. Extract content from `state["draft"]` or `state["channel_drafts"]`
2. Orchestrate sub-agents for classification, title generation, image selection
3. Generate PDF from content
4. Process cover image (text removal)
5. Map content to appropriate genre
6. Upload to digitaldossier.us API
7. Update pipeline state with results
8. Log all operations with timestamps

### Sub-Agents Architecture

#### Category Classification Agent
**File**: `langgraph/contentrunway/agents/category_classifier_agent.py`
- Use OpenAI to analyze content structure and purpose
- Classify as Blog or Product based on content analysis
- Product: Content describing/discussing specific products/platforms
- Blog: General informational/educational content
- Return classification with confidence score

#### Title Generation Agent
**File**: `langgraph/contentrunway/agents/title_generator_agent.py`
- Generate 4 title variants using OpenAI
- Consider SEO, engagement, and domain relevance
- Score titles based on multiple criteria
- Return best title with reasoning

#### Cover Image Selection Agent
**File**: `langgraph/contentrunway/agents/cover_image_agent.py`
- Analyze content for relevant image themes
- Select from category-specific folders (blog/ or product/)
- Process selected image to remove text
- Return base64-encoded image with metadata

### Tools Implementation

#### DigitalDossier API Tool
**File**: `langgraph/contentrunway/tools/digitaldossier_api_tool.py`
- Handle all API communication with digitaldossier.us
- Environment-based configuration (test/production)
- Authentication management
- Genre fetching and mapping
- Document upload with comprehensive error handling
- Response processing and validation

#### PDF Generation Tool
**File**: `langgraph/contentrunway/tools/pdf_generator_tool.py`
- Convert markdown/HTML content to formatted PDF
- Include title page with author "Suvojit Dutta"
- Proper document formatting and structure
- Return base64-encoded PDF with metadata

#### Cover Image Processor Tool
**File**: `langgraph/contentrunway/tools/cover_image_processor_tool.py`
- File system operations for image selection
- Image processing for text removal using computer vision
- Base64 encoding with proper metadata
- Category-based folder navigation

#### Content Classification Tool
**File**: `langgraph/contentrunway/tools/content_classification_tool.py`
- OpenAI-based content analysis
- Domain expertise consideration (IT, Insurance, AI)
- Classification logic for Blog vs Product determination
- Confidence scoring and reasoning

#### Genre Mapping Tool
**File**: `langgraph/contentrunway/tools/genre_mapping_tool.py`
- Fetch available genres from digitaldossier.us API
- Content-based genre matching using semantic analysis
- Create new genre if no appropriate match found
- Genre caching for performance optimization

## Implementation Details

### File Structure
```
langgraph/contentrunway/
├── agents/
│   ├── publisher.py (complete rewrite)
│   ├── category_classifier_agent.py (new)
│   ├── title_generator_agent.py (new)
│   └── cover_image_agent.py (new)
├── tools/
│   ├── digitaldossier_api_tool.py (new)
│   ├── pdf_generator_tool.py (new)
│   ├── cover_image_processor_tool.py (new)
│   ├── content_classification_tool.py (new)
│   └── genre_mapping_tool.py (new)
└── utils/
    └── publisher_logger.py (new)
```

### Publisher Agent Implementation

**Key Components:**
```python
class PublisherAgent:
    def __init__(self):
        self.category_classifier = CategoryClassifierAgent()
        self.title_generator = TitleGeneratorAgent()
        self.cover_image_agent = CoverImageAgent()
        self.pdf_generator = PDFGeneratorTool()
        self.api_tool = DigitalDossierAPITool()
        self.logger = PublisherLogger()
    
    async def execute(self, channel_drafts, state):
        # 1. Extract content from state
        # 2. Generate category classification
        # 3. Generate optimized title
        # 4. Select and process cover image
        # 5. Generate PDF
        # 6. Map to appropriate genre
        # 7. Upload to digitaldossier.us
        # 8. Update pipeline state
        # 9. Log all operations
```

### Error Handling Strategy

**Logging System:**
```python
class PublisherLogger:
    def __init__(self):
        self.log_file = f"publisher_logs_{datetime.now().strftime('%Y%m%d')}.log"
    
    def log_error(self, operation, error, context=None):
        # Timestamp-based error logging
        # Dashboard-readable format
        # Context preservation for debugging
```

**Error Categories:**
1. API Communication Errors
2. File Processing Errors
3. Image Processing Errors
4. PDF Generation Errors
5. Genre Mapping Errors

**Error Response:**
- Log to timestamped file
- Update pipeline state with error details
- Provide actionable error messages
- No automatic retries (as specified)

### Cover Image Processing

**Implementation Approach:**
```python
class CoverImageProcessor:
    def select_image(self, category, content_analysis):
        # Select from appropriate folder (blog/ or product/)
        # Content-based relevance scoring
        # Return selected image path
    
    def remove_text(self, image_path):
        # Computer vision text detection
        # Inpainting to remove text areas
        # Preserve graphics and design elements
        # Return processed image
    
    def encode_image(self, processed_image):
        # Convert to PNG format
        # Base64 encoding
        # Generate metadata
```

**Text Removal Techniques:**
- OCR-based text detection
- Computer vision for text region identification
- Inpainting algorithms for text removal
- Quality preservation for graphics

## Environment Configuration

### Required Environment Variables
```env
# DigitalDossier API Configuration
DIGITALDOSSIER_API_TOKEN=your_api_token_here
DIGITALDOSSIER_BASE_URL=http://localhost:3003  # Test environment
DIGITALDOSSIER_ADMIN_EMAIL=your_admin_email
DIGITALDOSSIER_ADMIN_PASSWORD=your_admin_password

# OpenAI Configuration (if not already configured)
OPENAI_API_KEY=your_openai_key

# Logging Configuration
PUBLISHER_LOG_LEVEL=INFO
PUBLISHER_LOG_DIR=./logs/publisher/
```

### Configuration Management
```python
class PublisherConfig:
    def __init__(self):
        self.api_token = os.getenv('DIGITALDOSSIER_API_TOKEN')
        self.base_url = os.getenv('DIGITALDOSSIER_BASE_URL', 'https://digitaldossier.us')
        self.admin_email = os.getenv('DIGITALDOSSIER_ADMIN_EMAIL')
        self.admin_password = os.getenv('DIGITALDOSSIER_ADMIN_PASSWORD')
        
    def validate_config(self):
        # Validate all required environment variables
        # Throw configuration errors if missing
```

## Testing Strategy

### Unit Testing
```python
# Test Coverage Areas
- PDF generation from markdown
- Cover image text removal
- Category classification accuracy
- Title generation quality
- API communication
- Error handling scenarios
- State management updates
```

### Integration Testing
```python
# Test Scenarios
- End-to-end publishing workflow
- Pipeline state integration
- Error propagation
- Logging functionality
- Environment configuration switching
```

### Test Environment Setup
- Use `http://localhost:3003` for API testing
- Mock OpenAI responses for consistent testing
- Test image processing with sample cover images
- Validate PDF generation quality

## Dashboard Integration

### Log File Integration
**Log File Format:**
```json
{
  "timestamp": "2025-09-08T10:30:00Z",
  "level": "ERROR",
  "operation": "api_upload",
  "message": "Upload failed: Invalid genre ID",
  "context": {
    "title": "Sample Article",
    "category": "Blog",
    "genre_id": 999
  }
}
```

**Dashboard Display:**
- Real-time error log reading
- Pipeline progress tracking
- Publishing status monitoring
- Error categorization and filtering

### Frontend Component Updates
**Components to Update:**
- `frontend/src/components/dashboard/PipelineOverview.tsx`
- `frontend/src/components/dashboard/RecentRuns.tsx`

**New Components:**
- Publisher error log viewer
- Publishing status dashboard
- Cover image usage tracker (future enhancement)

## Deployment Considerations

### Production vs Test Environment
```python
class EnvironmentManager:
    def get_api_base_url(self):
        env = os.getenv('ENVIRONMENT', 'development')
        if env == 'production':
            return 'https://digitaldossier.us'
        else:
            return 'http://localhost:3003'
```

### Performance Optimization
- Genre caching to reduce API calls
- Image processing optimization
- Concurrent sub-agent execution
- PDF generation efficiency

### Monitoring and Observability
- Comprehensive logging for all operations
- Performance metrics tracking
- Success/failure rate monitoring
- API response time tracking

## Implementation Timeline

### Phase 1: Core Infrastructure (Week 1)
- Publisher agent framework
- Basic API integration tool
- PDF generation tool
- Environment configuration

### Phase 2: Intelligence Layer (Week 2)
- Category classification agent
- Title generation agent
- Cover image selection agent
- Content classification tool

### Phase 3: Image Processing (Week 3)
- Cover image processor tool
- Text removal implementation
- Image optimization
- Base64 encoding

### Phase 4: Integration & Testing (Week 4)
- Pipeline integration
- Error handling
- Logging system
- Comprehensive testing

### Phase 5: Dashboard & Monitoring (Week 5)
- Dashboard integration
- Log viewer implementation
- Performance monitoring
- Production deployment

## Success Criteria

### Functional Requirements
- ✅ Automatic PDF generation from content
- ✅ PNG cover image selection and processing
- ✅ Automatic category detection (Blog/Product)
- ✅ Automatic title generation (4 options, best selection)
- ✅ Local cover image selection with text removal
- ✅ DigitalDossier.us API integration
- ✅ Comprehensive error logging
- ✅ Pipeline state integration

### Quality Requirements
- 95%+ successful uploads for valid content
- <30 seconds total processing time
- 100% pipeline state compatibility
- Complete error traceability
- Dashboard error visibility

### Technical Requirements
- Zero breaking changes to existing pipeline
- Backward compatibility
- Environment-based configuration
- Comprehensive logging
- Scalable architecture

## Conclusion

This implementation plan provides a comprehensive roadmap for enhancing the ContentRunway Publisher Agent to automatically upload content to digitaldossier.us. The solution maintains 100% compatibility with the existing pipeline architecture while adding intelligent automation for category detection, title optimization, cover image processing, and API integration.

The modular architecture ensures maintainability and extensibility, while comprehensive error handling and logging provide operational visibility and debugging capabilities. The implementation follows ContentRunway's quality-first approach and integrates seamlessly with the existing agent orchestration system.

---

*Document Version: 1.0*  
*Last Updated: September 8, 2025*  
*Implementation Status: Ready for Development*