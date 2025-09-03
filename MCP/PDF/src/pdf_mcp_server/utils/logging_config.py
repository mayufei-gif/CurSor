"""Logging configuration for PDF-MCP server.

This module provides centralized logging configuration with support for
structured logging, file rotation, and different log levels.

中文说明：集中式日志配置模块，支持结构化(JSON)日志、文件轮转、
控制台彩色输出以及多级日志等级控制，便于生产/测试环境统一管理日志。
"""

import logging  # 日志系统主模块
import logging.handlers  # 含文件轮转等处理器
import sys  # 标准输出/错误输出等控制台相关
import json  # JSON 序列化（用于结构化日志）
from pathlib import Path  # 路径处理
from typing import Optional, Dict, Any  # 类型注解
from datetime import datetime  # 时间格式化


class JSONFormatter(logging.Formatter):  # 自定义 JSON 格式化器（结构化日志）
    """Custom JSON formatter for structured logging.

    中文：将日志记录转为 JSON 字符串，便于日志聚合/检索。
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON formatted log string
        """
        # Create log entry dictionary
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add process and thread info
        if hasattr(record, 'process'):
            log_entry["process_id"] = record.process
        if hasattr(record, 'thread'):
            log_entry["thread_id"] = record.thread
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'exc_info',
                'exc_text', 'stack_info'
            }:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):  # 控制台彩色输出格式化器
    """Colored formatter for console output.

    中文：按日志级别添加 ANSI 颜色（在支持的终端展示）。
    """
    
    # Color codes
    COLORS = {  # ANSI 颜色映射
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors.
        
        Args:
            record: Log record to format
            
        Returns:
            Colored log string
        """
        # Get color for log level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Format message
        message = record.getMessage()
        
        # Create formatted string
        formatted = f"{color}[{timestamp}] {record.levelname:8} {record.name:20} {message}{reset}"
        
        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    json_format: bool = False,
    console_output: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    logger_name: Optional[str] = None
) -> logging.Logger:
    """Setup logging configuration.

    中文：初始化并配置日志系统（控制台/文件、JSON/文本、轮转等）。
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name
        log_dir: Optional log directory
        json_format: Whether to use JSON formatting
        console_output: Whether to output to console
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        logger_name: Optional logger name
        
    Returns:
        Configured logger
    """
    # Get logger
    logger = logging.getLogger(logger_name or "pdf_mcp_server")
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Create formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    colored_formatter = ColoredFormatter()
    
    # Setup console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        # Use colored formatter for console if not JSON
        if json_format:
            console_handler.setFormatter(formatter)
        else:
            console_handler.setFormatter(colored_formatter)
        
        logger.addHandler(console_handler)
    
    # Setup file handler
    if log_file or log_dir:
        if log_dir:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            log_file_path = log_dir_path / (log_file or "pdf_mcp_server.log")
        else:
            log_file_path = Path(log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def setup_uvicorn_logging(log_level: str = "INFO", json_format: bool = False) -> Dict[str, Any]:
    """Setup logging configuration for Uvicorn.
    
    Args:
        log_level: Logging level
        json_format: Whether to use JSON formatting
        
    Returns:
        Uvicorn logging configuration
    """
    # Define formatters
    formatters = {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "access": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(client_addr)s - \"%(request_line)s\" %(status_code)s",
        },
    }
    
    if json_format:
        formatters["default"]["()"] = "pdf_mcp_server.utils.logging_config.JSONFormatter"
        formatters["access"]["()"] = "pdf_mcp_server.utils.logging_config.JSONFormatter"
    
    # Define handlers
    handlers = {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    }
    
    # Define loggers
    loggers = {
        "uvicorn": {
            "handlers": ["default"],
            "level": log_level,
            "propagate": False,
        },
        "uvicorn.error": {
            "handlers": ["default"],
            "level": log_level,
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["access"],
            "level": log_level,
            "propagate": False,
        },
    }
    
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": loggers,
    }


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_function_call(func):
    """Decorator to log function calls.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    
    return wrapper


def log_processing_step(step_name: str, logger: Optional[logging.Logger] = None):
    """Decorator to log processing steps.
    
    Args:
        step_name: Name of the processing step
        logger: Optional logger to use
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            log = logger or get_logger(func.__module__)
            log.info(f"Starting {step_name}")
            
            try:
                result = func(*args, **kwargs)
                log.info(f"Completed {step_name}")
                return result
            except Exception as e:
                log.error(f"Failed {step_name}: {e}")
                raise
        
        return wrapper
    return decorator


class LogContext:
    """Context manager for adding context to log messages."""
    
    def __init__(self, logger: logging.Logger, **context):
        """Initialize log context.
        
        Args:
            logger: Logger to add context to
            **context: Context key-value pairs
        """
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        """Enter context manager."""
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        logging.setLogRecordFactory(self.old_factory)


def create_file_logger(
    name: str,
    log_file: str,
    log_level: str = "INFO",
    json_format: bool = False,
    max_file_size: int = 10 * 1024 * 1024,
    backup_count: int = 5
) -> logging.Logger:
    """Create a file-only logger.
    
    Args:
        name: Logger name
        log_file: Log file path
        log_level: Logging level
        json_format: Whether to use JSON formatting
        max_file_size: Maximum file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured file logger
    """
    logger = logging.getLogger(name)
    logger.handlers.clear()
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Create formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Create file handler
    log_file_path = Path(log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.propagate = False
    
    return logger
