"""
Celery worker configuration for ContentRunway pipeline execution.
"""

import os
from celery import Celery
from celery.signals import worker_ready, worker_shutdown
import logging

from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "contentrunway",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.tasks.pipeline_tasks",
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task routing
    task_routes={
        "app.tasks.pipeline_tasks.*": {"queue": "pipeline"},
    },
    
    # Task execution settings
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    
    # Result backend settings
    result_expires=3600,  # 1 hour
    result_backend_transport_options={
        "retry_on_timeout": True,
        "visibility_timeout": 3600,
    },
    
    # Retry settings
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    
    # Beat schedule (if needed for periodic tasks)
    beat_schedule={},
)


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handle worker startup."""
    logger.info("ContentRunway Celery worker started and ready")


@worker_shutdown.connect 
def worker_shutdown_handler(sender=None, **kwargs):
    """Handle worker shutdown."""
    logger.info("ContentRunway Celery worker shutting down")


# Make celery app available for CLI
app = celery_app

if __name__ == "__main__":
    celery_app.start()