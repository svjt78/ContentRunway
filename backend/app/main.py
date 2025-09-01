from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import socketio

from app.core.config import settings
from app.api.routes import api_router
from app.db.database import engine
from app.models import Base

# Create FastAPI app
app = FastAPI(
    title="ContentRunway API",
    description="AI-powered content creation pipeline",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS,
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Socket.IO setup for real-time updates
sio = socketio.AsyncServer(
    cors_allowed_origins=settings.ALLOWED_ORIGINS,
    async_mode="asgi"
)

# Mount Socket.IO
socket_app = socketio.ASGIApp(sio, app)

@app.on_event("startup")
async def startup_event():
    """Initialize database and services on startup."""
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    print("ContentRunway API started successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await engine.dispose()
    print("ContentRunway API shut down.")

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    """Handle client connection."""
    print(f"Client {sid} connected")

@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    print(f"Client {sid} disconnected")

@sio.event
async def pipeline_status(sid, data):
    """Handle pipeline status updates."""
    # Emit status to connected clients
    await sio.emit("pipeline_update", data)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ContentRunway API"}

# Export the ASGI app
app = socket_app