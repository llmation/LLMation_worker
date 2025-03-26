"""
日志服务实现

该模块实现了基于模式定义的日志系统。
提供了结构化日志输出和多种格式选项。
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from loguru import logger
from pythonjsonlogger import jsonlogger
from flask import Flask, has_request_context, request

from app.utils.logger_schema import LogLevel, LogFormat, LoggerConfig, LoggerInterface


class JSONLogFormatter(jsonlogger.JsonFormatter):
    """自定义JSON日志格式化器"""

    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ) -> None:
        """添加自定义字段到日志记录"""
        super().add_fields(log_record, record, message_dict)

        # 添加标准字段
        log_record["timestamp"] = datetime.utcnow().isoformat()
        log_record["level"] = record.levelname
        log_record["module"] = record.module

        # 添加请求相关信息（如果在Flask请求上下文中）
        if has_request_context():
            log_record["request_id"] = getattr(request, "id", None)
            log_record["remote_addr"] = request.remote_addr
            log_record["method"] = request.method
            log_record["path"] = request.path

        # 处理额外参数
        for key, value in message_dict.items():
            if key not in log_record:
                log_record[key] = value


class AppLogger(LoggerInterface):
    """应用日志记录器实现"""

    def __init__(self, config: LoggerConfig):
        """
        初始化日志记录器

        Args:
            config: 日志配置对象
        """
        self.config = config
        self.name = config.name
        self._logger = self._configure_logger(config)

    def _configure_logger(self, config: LoggerConfig) -> logger:
        """配置loguru日志记录器"""
        # 删除默认处理器
        logger.remove()

        # 根据配置设置日志格式
        if config.format_type == LogFormat.JSON:
            # 使用简单的字符串格式，避免自定义函数出错
            log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}"
            logger.add(
                sys.stderr,
                format=log_format,
                level=config.level.value,
                serialize=True,  # 使用loguru内置的JSON序列化
                backtrace=True,
                diagnose=True,
                catch=True,
            )

            if config.file_path:
                file_path = Path(config.file_path)
                file_path.parent.mkdir(parents=True, exist_ok=True)

                logger.add(
                    str(file_path),
                    format=log_format,
                    level=config.level.value,
                    rotation=config.rotation,
                    retention=config.retention,
                    serialize=True,  # 使用loguru内置的JSON序列化
                    backtrace=True,
                    diagnose=True,
                    catch=True,
                )
        else:
            # 使用标准格式
            log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

            logger.add(
                sys.stderr,
                format=log_format,
                level=config.level.value,
                serialize=False,
                backtrace=True,
                diagnose=True,
                catch=True,
            )

            if config.file_path:
                file_path = Path(config.file_path)
                file_path.parent.mkdir(parents=True, exist_ok=True)

                logger.add(
                    str(file_path),
                    format=log_format,
                    level=config.level.value,
                    rotation=config.rotation,
                    retention=config.retention,
                    serialize=False,
                    backtrace=True,
                    diagnose=True,
                    catch=True,
                )

        # 添加额外处理器
        for handler_name, handler_config in (config.extra_handlers or {}).items():
            if callable(handler_config.get("sink")):
                logger.add(**handler_config)

        return logger.bind(name=config.name)

    def debug(self, message: str, **kwargs: Any) -> None:
        """记录调试级别日志"""
        self._logger.bind(**kwargs).debug(message)

    def info(self, message: str, **kwargs: Any) -> None:
        """记录信息级别日志"""
        self._logger.bind(**kwargs).info(message)

    def warning(self, message: str, **kwargs: Any) -> None:
        """记录警告级别日志"""
        self._logger.bind(**kwargs).warning(message)

    def error(self, message: str, **kwargs: Any) -> None:
        """记录错误级别日志"""
        self._logger.bind(**kwargs).error(message)

    def critical(self, message: str, **kwargs: Any) -> None:
        """记录严重错误级别日志"""
        self._logger.bind(**kwargs).critical(message)

    def log(self, level: LogLevel, message: str, **kwargs: Any) -> None:
        """记录指定级别日志"""
        self._logger.bind(**kwargs).log(level.value, message)


class FlaskAppLoggerAdapter:
    """Flask应用日志适配器"""

    def __init__(self, app: Flask, config: Optional[LoggerConfig] = None):
        """
        初始化Flask应用的日志适配器

        Args:
            app: Flask应用实例
            config: 日志配置对象，如果为None则使用默认配置
        """
        # 如果未提供配置，创建默认配置
        if config is None:
            log_dir = os.path.join(app.instance_path, "logs")
            log_file = os.path.join(log_dir, "app.log")

            config = LoggerConfig(
                name=app.name,
                level=LogLevel.INFO if not app.debug else LogLevel.DEBUG,
                format_type=LogFormat.JSON,
                file_path=log_file,
                rotation="10 MB",
                retention="30 days",
            )

        # 初始化应用日志记录器
        self.logger = AppLogger(config)

        # 替换Flask的默认日志记录器
        app.logger_name = config.name

        # 添加请求ID中间件
        @app.before_request
        def add_request_id():
            """为每个请求添加唯一标识符"""
            request.id = (
                datetime.now().strftime("%Y%m%d%H%M%S") + "-" + str(os.urandom(4).hex())
            )

        # 记录请求开始和结束
        @app.before_request
        def log_request_start():
            """记录请求开始"""
            request._start_time = datetime.now()
            try:
                self.logger.info(
                    f"开始处理请求: {request.method} {request.path}",
                    request_id=getattr(request, "id", None),
                    remote_addr=request.remote_addr,
                    method=request.method,
                    path=request.path,
                    user_agent=request.user_agent.string
                    if request.user_agent
                    else None,
                )
            except Exception as e:
                print(f"请求日志记录失败: {str(e)}")

        @app.after_request
        def log_request_end(response):
            """记录请求结束"""
            try:
                duration = datetime.now() - request._start_time
                status_code = response.status_code

                log_method = self.logger.info
                if status_code >= 500:
                    log_method = self.logger.error
                elif status_code >= 400:
                    log_method = self.logger.warning

                log_method(
                    f"完成处理请求: {request.method} {request.path} {status_code}",
                    request_id=getattr(request, "id", None),
                    method=request.method,
                    path=request.path,
                    status_code=status_code,
                    duration_ms=duration.total_seconds() * 1000,
                )
            except Exception as e:
                print(f"响应日志记录失败: {str(e)}")

            return response

        # 记录未捕获的异常
        @app.errorhandler(Exception)
        def log_exception(error):
            """记录未捕获的异常"""
            try:
                error_traceback = traceback.format_exc()
                self.logger.error(
                    f"未捕获异常: {str(error)}",
                    request_id=getattr(request, "id", None)
                    if has_request_context()
                    else None,
                    exception_type=error.__class__.__name__,
                    exception=str(error),
                    traceback=error_traceback,
                )
            except Exception as log_err:
                # 如果日志记录本身失败，使用标准库记录
                print(f"日志记录失败: {str(log_err)}")
                print(f"原始错误: {str(error)}")
                print(traceback.format_exc())

            # 不要再调用app.handle_http_exception，直接返回错误响应
            if hasattr(error, "code"):
                return app.make_response((str(error), error.code))
            return None


def init_app_logger(app: Flask, config: Optional[LoggerConfig] = None) -> AppLogger:
    """
    初始化应用日志系统

    Args:
        app: Flask应用实例
        config: 日志配置对象，如果为None则使用默认配置

    Returns:
        配置好的日志记录器实例
    """
    adapter = FlaskAppLoggerAdapter(app, config)
    return adapter.logger


# 创建一个默认的日志记录器实例，可用于非Flask上下文
default_logger = AppLogger(
    LoggerConfig(name="app", level=LogLevel.INFO, format_type=LogFormat.SIMPLE)
)
