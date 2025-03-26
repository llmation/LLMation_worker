"""
日志服务模式定义

该模块定义了应用程序日志系统的模式和接口。
所有日志相关的类型和验证都基于此模式。
"""

from enum import Enum
from typing import Dict, Any, Optional


class LogLevel(str, Enum):
    """日志级别枚举"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """日志格式枚举"""

    SIMPLE = "SIMPLE"  # 简单文本格式
    JSON = "JSON"  # JSON结构化格式


class LoggerConfig:
    """日志配置模式"""

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        format_type: LogFormat = LogFormat.JSON,
        file_path: Optional[str] = None,
        rotation: Optional[str] = None,
        retention: Optional[str] = None,
        extra_handlers: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化日志配置

        Args:
            name: 日志记录器名称
            level: 日志级别
            format_type: 日志格式类型
            file_path: 日志文件路径（如果需要文件输出）
            rotation: 日志轮转设置，例如"1 day", "10 MB"
            retention: 日志保留设置，例如"10 days", "5 files"
            extra_handlers: 额外的日志处理器配置
        """
        self.name = name
        self.level = level
        self.format_type = format_type
        self.file_path = file_path
        self.rotation = rotation
        self.retention = retention
        self.extra_handlers = extra_handlers or {}


class LoggerInterface:
    """日志接口定义"""

    def debug(self, message: str, **kwargs: Any) -> None:
        """记录调试级别日志"""
        pass

    def info(self, message: str, **kwargs: Any) -> None:
        """记录信息级别日志"""
        pass

    def warning(self, message: str, **kwargs: Any) -> None:
        """记录警告级别日志"""
        pass

    def error(self, message: str, **kwargs: Any) -> None:
        """记录错误级别日志"""
        pass

    def critical(self, message: str, **kwargs: Any) -> None:
        """记录严重错误级别日志"""
        pass

    def log(self, level: LogLevel, message: str, **kwargs: Any) -> None:
        """记录指定级别日志"""
        pass
