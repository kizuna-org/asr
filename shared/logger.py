#!/usr/bin/env python3
"""
Structured Logger for CI/CD System
Provides consistent JSON logging across all components
"""

import json
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    FATAL = "FATAL"

class Component(Enum):
    APP = "app"
    BUILD_SUBSCRIBER = "build_subscriber"
    APP_SUBSCRIBER = "app_subscriber"
    GITHUB_ACTIONS = "github_actions"

class StructuredLogger:
    def __init__(self, component: Component, job_id: Optional[str] = None):
        self.component = component
        self.job_id = job_id
    
    def _create_log_entry(
        self,
        level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
        performance: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a structured log entry"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "level": level.value,
            "component": self.component.value,
            "message": message
        }
        
        if self.job_id:
            log_entry["job_id"] = self.job_id
        
        if context:
            log_entry["context"] = context
        
        if error:
            log_entry["error"] = error
        
        if performance:
            log_entry["performance"] = performance
        
        if tags:
            log_entry["tags"] = tags
        
        return log_entry
    
    def _log(self, log_entry: Dict[str, Any]) -> str:
        """Output the log entry and return as string"""
        log_json = json.dumps(log_entry, separators=(',', ':'))
        print(log_json)
        return log_json + '\n'
    
    def debug(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Log debug message"""
        log_entry = self._create_log_entry(LogLevel.DEBUG, message, context=context, tags=tags)
        return self._log(log_entry)
    
    def info(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        performance: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Log info message"""
        log_entry = self._create_log_entry(
            LogLevel.INFO, message, context=context, performance=performance, tags=tags
        )
        return self._log(log_entry)
    
    def warn(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Log warning message"""
        log_entry = self._create_log_entry(LogLevel.WARN, message, context=context, tags=tags)
        return self._log(log_entry)
    
    def error(
        self,
        message: str,
        error_details: Optional[str] = None,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Log error message"""
        error_info = {}
        
        if exception:
            error_info.update({
                "type": type(exception).__name__,
                "details": str(exception),
                "stack_trace": traceback.format_exc()
            })
        elif error_details:
            error_info["details"] = error_details
        
        log_entry = self._create_log_entry(
            LogLevel.ERROR, message, context=context, error=error_info, tags=tags
        )
        return self._log(log_entry)
    
    def fatal(
        self,
        message: str,
        error_details: Optional[str] = None,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Log fatal message"""
        error_info = {}
        
        if exception:
            error_info.update({
                "type": type(exception).__name__,
                "details": str(exception),
                "stack_trace": traceback.format_exc()
            })
        elif error_details:
            error_info["details"] = error_details
        
        log_entry = self._create_log_entry(
            LogLevel.FATAL, message, context=context, error=error_info, tags=tags
        )
        return self._log(log_entry)
    
    def operation_start(
        self,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Log operation start"""
        op_context = {"operation": operation, "step": "start"}
        if context:
            op_context.update(context)
        
        return self.info(f"Starting operation: {operation}", context=op_context, tags=tags)
    
    def operation_progress(
        self,
        operation: str,
        step: str,
        current: Optional[int] = None,
        total: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Log operation progress"""
        op_context = {"operation": operation, "step": step}
        
        if current is not None and total is not None:
            percentage = (current / total) * 100
            op_context["progress"] = {
                "current": current,
                "total": total,
                "percentage": round(percentage, 2)
            }
        
        if context:
            op_context.update(context)
        
        message = f"Operation {operation}: {step}"
        if current is not None and total is not None:
            message += f" ({current}/{total})"
        
        return self.info(message, context=op_context, tags=tags)
    
    def operation_complete(
        self,
        operation: str,
        duration_ms: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Log operation completion"""
        op_context = {"operation": operation, "step": "complete"}
        if context:
            op_context.update(context)
        
        performance = {}
        if duration_ms is not None:
            performance["duration_ms"] = duration_ms
        
        return self.info(
            f"Operation completed: {operation}",
            context=op_context,
            performance=performance if performance else None,
            tags=tags
        )
    
    def operation_failed(
        self,
        operation: str,
        error_details: Optional[str] = None,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Log operation failure"""
        op_context = {"operation": operation, "step": "failed"}
        if context:
            op_context.update(context)
        
        return self.error(
            f"Operation failed: {operation}",
            error_details=error_details,
            exception=exception,
            context=op_context,
            tags=tags
        )