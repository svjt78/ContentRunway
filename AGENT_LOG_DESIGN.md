# Agent Log Tab Design Specification

## Overview

This document outlines the design for implementing an "Agent Log" tab in the ContentRunway pipeline interface. The feature will display real-time agent outputs for each pipeline run with refresh and delete functionality, maintaining complete isolation from existing functionality.

**Status: ✅ IMPLEMENTED** - This design has been fully implemented and is operational as of the latest deployment.

## Requirements

- Display agent logs for a specific pipeline run in a dedicated tab
- Real-time log updates with manual refresh capability
- Delete logs functionality for a specific pipeline run
- Maintain existing UI/UX patterns and styling
- Zero impact on existing codebase functionality
- Complete isolation of new features

## Architecture Overview

### Database Layer

#### New Table: `agent_logs`
```sql
CREATE TABLE IF NOT EXISTS agent_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_run_id UUID NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    agent_name VARCHAR(100) NOT NULL,
    stage VARCHAR(50) NOT NULL,
    operation VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    context JSONB,
    level VARCHAR(20) NOT NULL DEFAULT 'INFO', -- INFO, WARNING, ERROR, DEBUG
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_agent_logs_pipeline_run ON agent_logs(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_agent_logs_timestamp ON agent_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_agent_logs_level ON agent_logs(level);
CREATE INDEX IF NOT EXISTS idx_agent_logs_agent_name ON agent_logs(agent_name);
```

### Backend API Endpoints

#### Base URL: `/api/v1/agent-logs`

1. **GET /agent-logs/{pipeline_run_id}**
   - Fetch all agent logs for a specific pipeline run
   - Query parameters:
     - `limit`: Number of logs to return (default: 100)
     - `offset`: Pagination offset (default: 0)
     - `level`: Filter by log level (optional)
     - `agent_name`: Filter by agent name (optional)
   - Response: Array of agent log objects

2. **DELETE /agent-logs/{pipeline_run_id}**
   - Delete all agent logs for a specific pipeline run
   - Response: Success confirmation

#### Response Schema
```typescript
interface AgentLog {
  id: string;
  pipeline_run_id: string;
  agent_name: string;
  stage: string;
  operation: string;
  message: string;
  context?: Record<string, any>;
  level: 'INFO' | 'WARNING' | 'ERROR' | 'DEBUG';
  timestamp: string;
  created_at: string;
}
```

### Frontend Implementation

#### Tab Structure
Convert the existing pipeline detail page (`/pipelines/[id]/page.tsx`) to use a tabbed interface:

1. **Overview Tab** (existing content)
2. **Content Tab** (existing generated content section)
3. **Agent Log Tab** (new)

#### Agent Log Component (`AgentLog.tsx`)

```typescript
interface AgentLogProps {
  pipelineRunId: string;
}

interface AgentLogState {
  logs: AgentLog[];
  loading: boolean;
  error: string | null;
  autoRefresh: boolean;
  filters: {
    level?: string;
    agent_name?: string;
  };
}
```

#### Features:
- **Real-time Updates**: Auto-refresh every 5 seconds (toggleable)
- **Manual Refresh**: Button to manually refresh logs
- **Filtering**: Filter by log level and agent name
- **Delete Functionality**: Button to delete all logs for the pipeline
- **Chronological Display**: Latest logs at the top
- **Expandable Context**: Click to view detailed context data
- **Level-based Styling**: Color coding for different log levels

### UI/UX Design

#### Layout Structure
```
┌─────────────────────────────────────────────────────────────┐
│ Pipeline Run Header                                         │
├─────────────────────────────────────────────────────────────┤
│ [Overview] [Content] [Agent Log]                           │
├─────────────────────────────────────────────────────────────┤
│ Agent Log Tab Content:                                      │
│                                                             │
│ ┌─────────────────────┐ ┌──────────────┐ ┌───────────────┐ │
│ │ [🔄 Refresh] [Auto] │ │ Filter: All ▼│ │ [🗑️ Delete] │ │
│ └─────────────────────┘ └──────────────┘ └───────────────┘ │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 🟢 [2025-01-05 15:30:22] ResearchCoordinatorAgent      │ │
│ │    STAGE: research | OPERATION: execute_research        │ │
│ │    Starting research phase for IT Insurance domain     │ │
│ │    Context: {...}                                       │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 🟡 [2025-01-05 15:30:25] ContentCuratorAgent           │ │
│ │    STAGE: curation | OPERATION: analyze_topics         │ │
│ │    Analyzing 8 research topics for relevance           │ │
│ │    Context: {...}                                       │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 🔴 [2025-01-05 15:30:30] ContentWriterAgent            │ │
│ │    STAGE: writing | OPERATION: generate_content        │ │
│ │    Failed to generate content: API rate limit exceeded │ │
│ │    Context: {...}                                       │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### Color Coding
- 🟢 **INFO**: Green indicator
- 🟡 **WARNING**: Yellow indicator  
- 🔴 **ERROR**: Red indicator
- 🔵 **DEBUG**: Blue indicator

#### Interactive Elements
- **Expandable Logs**: Click log entry to expand context details
- **Auto-refresh Toggle**: Enable/disable automatic updates
- **Filter Dropdown**: Filter by log level and agent name
- **Delete Confirmation**: Modal dialog for delete confirmation

### Integration Points

#### Pipeline Tasks Integration
Add logging calls throughout the 10-stage pipeline in `pipeline_tasks.py`:

```python
def log_agent_activity(
    pipeline_run_id: str,
    agent_name: str,
    stage: str,
    operation: str,
    message: str,
    level: str = "INFO",
    context: dict = None
):
    """Log agent activity to database"""
    # Implementation details
```

#### Logging Points:
1. **Agent Start/End**: Log when each agent begins and completes
2. **Key Operations**: Log major operations within each agent
3. **Errors and Warnings**: Log all error conditions and warnings
4. **Quality Gate Results**: Log quality assessment outcomes
5. **State Transitions**: Log when pipeline moves between stages

#### Implementation Status:
**✅ Fully Instrumented Pipeline Stages:**
- Research Stage (ResearchCoordinatorAgent)
- Curation Stage (ContentCuratorAgent)
- SEO Strategy Stage (SEOStrategistAgent)
- Writing Stage (ContentWriterAgent)
- Quality Gates Stage (FactCheckGateAgent, DomainExpertiseGateAgent, StyleCriticGateAgent, ComplianceGateAgent)
- Editing Stage (ContentEditorAgent)
- Critique Stage (CritiqueAgent)
- Formatting Stage (ContentFormatterAgent)
- Human Review Stage (HumanReviewGateAgent)
- Publishing Stage (PublisherAgent)
- Resume Function (PipelineOrchestrator)

### Database Operations

#### New Functions in `sync_database.py`
```python
def create_agent_log(
    pipeline_run_id: str,
    agent_name: str,
    stage: str,
    operation: str,
    message: str,
    level: str = "INFO",
    context: dict = None
) -> str:
    """Create a new agent log entry"""

def get_agent_logs(
    pipeline_run_id: str,
    limit: int = 100,
    offset: int = 0,
    level: str = None,
    agent_name: str = None
) -> List[dict]:
    """Retrieve agent logs for a pipeline run"""

def delete_agent_logs(pipeline_run_id: str) -> int:
    """Delete all agent logs for a pipeline run"""
```

### API Implementation

#### New Endpoint File: `backend/app/api/endpoints/agent_logs.py`
```python
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from app.schemas.agent_logs import AgentLogResponse
from app.services.agent_log_service import AgentLogService

router = APIRouter()

@router.get("/{pipeline_run_id}")
async def get_agent_logs(
    pipeline_run_id: str,
    limit: int = 100,
    offset: int = 0,
    level: Optional[str] = None,
    agent_name: Optional[str] = None
) -> List[AgentLogResponse]:
    """Get agent logs for a pipeline run"""

@router.delete("/{pipeline_run_id}")
async def delete_agent_logs(pipeline_run_id: str):
    """Delete all agent logs for a pipeline run"""
```

### Frontend API Functions

#### New Functions in `frontend/src/lib/api.ts`
```typescript
export interface AgentLog {
  id: string;
  pipeline_run_id: string;
  agent_name: string;
  stage: string;
  operation: string;
  message: string;
  context?: Record<string, any>;
  level: 'INFO' | 'WARNING' | 'ERROR' | 'DEBUG';
  timestamp: string;
  created_at: string;
}

export const getAgentLogs = async (
  pipelineRunId: string,
  params?: {
    limit?: number;
    offset?: number;
    level?: string;
    agent_name?: string;
  }
): Promise<AgentLog[]> => {
  const response = await api.get(`/agent-logs/${pipelineRunId}`, { params });
  return response.data;
};

export const deleteAgentLogs = async (pipelineRunId: string): Promise<void> => {
  await api.delete(`/agent-logs/${pipelineRunId}`);
};
```

## Implementation Strategy

### Phase 1: Database and Backend ✅ COMPLETED
1. ✅ Create database migration file (`004_agent_logs.sql`)
2. ✅ Implement database operations in `sync_database.py`
3. ✅ Create API endpoints and schemas (`agent_logs.py`)
4. ✅ Add logging integration to pipeline tasks (all 10 stages + resume function)

### Phase 2: Frontend Implementation ✅ COMPLETED
1. ✅ Create `AgentLog` component (`AgentLog.tsx`)
2. ✅ Convert pipeline detail page to tabbed interface (`pipelines/[id]/page.tsx`)
3. ✅ Add API functions to `api.ts`
4. ✅ Integrate component with existing page

### Phase 3: Testing and Integration ✅ COMPLETED
1. ✅ Test all CRUD operations (database connections, log creation, retrieval)
2. ✅ Verify real-time updates (5-second auto-refresh working)
3. ✅ Test delete functionality (foreign key constraints enforced)
4. ✅ Ensure no impact on existing features (isolated implementation)

## Error Handling

### Backend
- Validate pipeline_run_id exists
- Handle database connection errors
- Return appropriate HTTP status codes
- Log errors for debugging

### Frontend
- Display loading states during API calls
- Show error messages for failed operations
- Graceful degradation if logs unavailable
- Confirmation dialogs for destructive actions

## Performance Considerations

### Database
- Indexed queries for fast retrieval
- Pagination for large log sets
- Cascade delete for cleanup
- Efficient bulk operations

### Frontend
- Virtualized scrolling for large log lists
- Debounced filtering
- Optimistic updates where appropriate
- Efficient re-rendering with React keys

## Security Considerations

- Validate pipeline_run_id ownership/access
- Sanitize log message content
- Rate limiting on API endpoints
- Secure delete operations with confirmation

## Future Enhancements

- Export logs to file formats (CSV, JSON)
- Advanced filtering and search capabilities
- Log retention policies
- Real-time notifications for critical errors
- Agent performance metrics and analytics

## Migration Path

This implementation maintains complete backward compatibility:
- No changes to existing database tables
- No modifications to existing API endpoints
- No alterations to existing frontend components
- Isolated new functionality with clear boundaries

The Agent Log feature can be developed, tested, and deployed independently without affecting current system operation.

## Implementation Notes

### Key Technical Decisions Made:
1. **JSONB Serialization**: Used `json.dumps()` instead of `str()` for proper JSON storage
2. **Foreign Key Constraints**: Enforced referential integrity to prevent orphaned logs
3. **Docker Migration**: Applied using `docker compose exec postgres psql` commands
4. **Logging Pattern**: Consistent start/complete/error logging across all pipeline stages
5. **Error Isolation**: Stage-level error handling prevents cascade failures
6. **Context Data**: Rich context objects provide debugging information for each log entry

### Critical Fixes Applied:
- ✅ Fixed JSONB serialization bug in `sync_database.py`
- ✅ Applied database migration to worker container
- ✅ Added comprehensive instrumentation to all 10 pipeline stages
- ✅ Implemented proper error handling and foreign key validation
- ✅ Verified end-to-end functionality from database to frontend

The Agent Log feature is now fully operational and provides comprehensive visibility into pipeline execution for debugging and monitoring purposes.