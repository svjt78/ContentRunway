"""Publisher Logger - Comprehensive logging for DigitalDossier publishing operations."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


class PublisherLogger:
    """Logger for publisher operations with timestamped file logging."""
    
    def __init__(self):
        self.log_dir = Path(os.getenv('PUBLISHER_LOG_DIR', './logs/publisher/'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / f"publisher_logs_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Setup file logger
        self.logger = logging.getLogger('publisher')
        self.logger.setLevel(getattr(logging, os.getenv('PUBLISHER_LOG_LEVEL', 'INFO')))
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter for structured logging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
    
    def log_info(self, operation: str, message: str, context: Optional[Dict[str, Any]] = None):
        """Log information message."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "operation": operation,
            "message": message,
            "context": context or {}
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_error(self, operation: str, error: str, context: Optional[Dict[str, Any]] = None):
        """Log error message with context."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "ERROR",
            "operation": operation,
            "message": error,
            "context": context or {}
        }
        
        self.logger.error(json.dumps(log_entry))
    
    def log_warning(self, operation: str, message: str, context: Optional[Dict[str, Any]] = None):
        """Log warning message."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "WARNING",
            "operation": operation,
            "message": message,
            "context": context or {}
        }
        
        self.logger.warning(json.dumps(log_entry))
    
    def log_operation_start(self, operation: str, context: Optional[Dict[str, Any]] = None):
        """Log operation start."""
        self.log_info(
            operation=operation,
            message=f"Starting {operation}",
            context=context
        )
    
    def log_operation_success(self, operation: str, result: Dict[str, Any], context: Optional[Dict[str, Any]] = None):
        """Log successful operation completion."""
        self.log_info(
            operation=operation,
            message=f"Successfully completed {operation}",
            context={**(context or {}), "result": result}
        )
    
    def log_operation_failure(self, operation: str, error: str, context: Optional[Dict[str, Any]] = None):
        """Log operation failure."""
        self.log_error(
            operation=operation,
            error=f"Failed to complete {operation}: {error}",
            context=context
        )
    
    def get_log_file_path(self) -> str:
        """Get current log file path."""
        return str(self.log_file)
    
    def get_recent_logs(self, lines: int = 100) -> list:
        """Get recent log entries."""
        logs = []
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                # Read last n lines
                file_lines = f.readlines()
                recent_lines = file_lines[-lines:] if len(file_lines) > lines else file_lines
                
                for line in recent_lines:
                    try:
                        # Extract JSON from log line (after timestamp and level info)
                        json_part = line.split(' - ', 3)[-1].strip()
                        log_entry = json.loads(json_part)
                        logs.append(log_entry)
                    except (json.JSONDecodeError, IndexError):
                        continue
        
        except FileNotFoundError:
            pass
        
        return logs