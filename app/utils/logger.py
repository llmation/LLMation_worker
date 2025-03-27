"""
日志服务实现

该模块实现了基于模式定义的日志系统。
提供了结构化日志输出和多种格式选项。
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from loguru import logger  # type: ignore
from pythonjsonlogger import jsonlogger  # type: ignore
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


# 添加高可读性格式处理器
class ReadableLogFormatter:
    """生成高可读性日志格式的格式化器"""

    @staticmethod
    def format(record):
        """格式化日志记录为更具可读性的格式（终端版本，带颜色）"""
        # 基础属性格式化
        timestamp = record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level_color = {
            "DEBUG": "\033[36m",  # 青色
            "INFO": "\033[32m",  # 绿色
            "WARNING": "\033[33m",  # 黄色
            "ERROR": "\033[31m",  # 红色
            "CRITICAL": "\033[35m",  # 紫色
        }.get(record["level"].name, "\033[0m")
        reset = "\033[0m"

        # 构建基本日志行 - 使用更紧凑的格式
        log_line = f"{level_color}{timestamp} | {record['level'].name} | "

        # 添加上下文信息
        if "name" in record["extra"]:
            log_line += f"{record['extra']['name']} | "

        # 添加文件和行号 - 不再使用固定宽度
        file_name = (
            record["file"].name
            if hasattr(record["file"], "name")
            else str(record["file"])
        )
        line_number = record["line"] if "line" in record else 0
        log_line += f"{file_name}:{line_number} | "

        # 添加消息
        log_line += f"{record['message']}{reset}"

        # 添加结构化数据（如果存在）
        # 安全处理额外数据，确保不会尝试访问不存在的键
        extra_data = {}
        for k, v in record["extra"].items():
            if k != "name" and v is not None:
                extra_data[k] = v

        if extra_data:
            try:
                import json

                log_line += f"\n{level_color}额外数据: {json.dumps(extra_data, ensure_ascii=False, indent=2)}{reset}"
            except Exception:
                # 如果JSON序列化失败，使用简单格式
                log_line += f"\n{level_color}额外数据: {str(extra_data)}{reset}"

        # 添加异常信息（如果存在）
        if record["exception"]:
            log_line += f"\n{level_color}异常信息: {record['exception']}{reset}"

        # 确保每条日志以换行符结束
        log_line += "\n"

        return log_line

    @staticmethod
    def format_file(record):
        """格式化日志记录为更具可读性的格式（文件版本，无颜色）"""
        # 基础属性格式化
        timestamp = record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # 构建基本日志行 - 使用更紧凑的格式
        log_line = f"{timestamp} | {record['level'].name} | "

        # 添加上下文信息
        if "name" in record["extra"]:
            log_line += f"{record['extra']['name']} | "

        # 添加文件和行号 - 不再使用固定宽度
        file_name = (
            record["file"].name
            if hasattr(record["file"], "name")
            else str(record["file"])
        )
        line_number = record["line"] if "line" in record else 0
        log_line += f"{file_name}:{line_number} | "

        # 添加消息
        log_line += f"{record['message']}"

        # 添加结构化数据（如果存在）
        # 安全处理额外数据，确保不会尝试访问不存在的键
        extra_data = {}
        for k, v in record["extra"].items():
            if k != "name" and v is not None:
                extra_data[k] = v

        if extra_data:
            try:
                import json

                log_line += f"\n额外数据: {json.dumps(extra_data, ensure_ascii=False, indent=2)}"
            except Exception:
                # 如果JSON序列化失败，使用简单格式
                log_line += f"\n额外数据: {str(extra_data)}"

        # 添加异常信息（如果存在）
        if record["exception"]:
            log_line += f"\n异常信息: {record['exception']}"

        # 确保每条日志以换行符结束
        log_line += "\n"

        return log_line


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
            log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}\n"
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
        elif config.format_type == LogFormat.HYBRID:
            # HYBRID模式: 终端使用可读性格式，文件同时保存JSON和可读性格式

            # 终端使用可读性格式（带颜色）
            logger.add(
                sys.stderr,
                format=ReadableLogFormatter.format,
                level=config.level.value,
                backtrace=True,
                diagnose=True,
                catch=True,
            )

            if config.file_path:
                file_path = Path(config.file_path)
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # 保存JSON格式日志
                json_file_path = file_path.with_suffix(".json.log")
                # 确保JSON格式日志也能正确换行
                log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}\n"
                logger.add(
                    str(json_file_path),
                    format=log_format,
                    level=config.level.value,
                    rotation=config.rotation,
                    retention=config.retention,
                    serialize=True,  # 使用JSON序列化
                    backtrace=True,
                    diagnose=True,
                    catch=True,
                )

                # 同时保存可读性格式日志（无颜色版本）
                readable_file_path = file_path.with_suffix(".readable.log")
                logger.add(
                    str(readable_file_path),
                    format=ReadableLogFormatter.format_file,  # 使用无颜色版本
                    level=config.level.value,
                    rotation=config.rotation,
                    retention=config.retention,
                    backtrace=True,
                    diagnose=True,
                    catch=True,
                )
        elif config.format_type == LogFormat.READABLE:
            # 使用自定义的高可读性格式
            # 终端使用带颜色版本
            logger.add(
                sys.stderr,
                format=ReadableLogFormatter.format,
                level=config.level.value,
                backtrace=True,
                diagnose=True,
                catch=True,
            )

            if config.file_path:
                file_path = Path(config.file_path)
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # 文件使用无颜色版本
                logger.add(
                    str(file_path),
                    format=ReadableLogFormatter.format_file,  # 使用无颜色版本
                    level=config.level.value,
                    rotation=config.rotation,
                    retention=config.retention,
                    backtrace=True,
                    diagnose=True,
                    catch=True,
                )
        else:
            # 使用标准格式
            log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n"

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
                format_type=LogFormat.HYBRID,  # 使用HYBRID模式
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
                # 安全获取请求ID，确保它不为空
                request_id = getattr(request, "id", None)
                if not request_id:
                    request_id = "无ID-" + datetime.now().strftime("%Y%m%d%H%M%S")

                self.logger.info(
                    f"开始处理请求: {request.method} {request.path}",
                    request_id=request_id,
                    remote_addr=request.remote_addr,
                    method=request.method,
                    path=request.path,
                    user_agent=request.user_agent.string
                    if request.user_agent
                    else None,
                )
            except Exception as e:
                print(f"请求日志记录失败: {str(e)}")
                # 尝试使用更简单的方式记录
                try:
                    print(f"备用日志: 开始处理请求 {request.method} {request.path}")
                except Exception:
                    pass

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

                # 安全获取请求ID，确保它不为空
                request_id = getattr(request, "id", None)
                if not request_id:
                    request_id = "无ID-" + datetime.now().strftime("%Y%m%d%H%M%S")

                # 使用命名参数而不是嵌套的字典，避免格式化问题
                log_method(
                    f"完成处理请求: {request.method} {request.path} {status_code}",
                    request_id=request_id,
                    method=request.method,
                    path=request.path,
                    status_code=status_code,
                    duration_ms=duration.total_seconds() * 1000,
                )
            except Exception as e:
                print(f"响应日志记录失败: {str(e)}")
                # 尝试使用更简单的方式记录
                try:
                    print(
                        f"备用日志: 完成处理请求 {request.method} {request.path} {response.status_code}"
                    )
                except Exception:
                    pass

            return response

        # 记录未捕获的异常
        @app.errorhandler(Exception)
        def log_exception(error):
            """记录未捕获的异常"""
            try:
                error_traceback = traceback.format_exc()

                # 安全获取请求ID
                request_id = None
                if has_request_context():
                    request_id = getattr(request, "id", None)
                    if not request_id:
                        request_id = "无ID-" + datetime.now().strftime("%Y%m%d%H%M%S")

                self.logger.error(
                    f"未捕获异常: {str(error)}",
                    request_id=request_id,
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
    LoggerConfig(name="app", level=LogLevel.INFO, format_type=LogFormat.HYBRID)
)

# 使用示例:
# from app.utils.logger import default_logger
#
# # 基本日志
# default_logger.info("这是一条信息日志")
# default_logger.warning("这是一条警告日志")
# default_logger.error("这是一条错误日志")
#
# # 带有额外字段的结构化日志
# default_logger.info("用户登录成功", user_id="12345", ip="192.168.1.1", action="login")
#
# # 自定义配置的日志记录器
# custom_logger = AppLogger(
#     LoggerConfig(
#         name="api",
#         level=LogLevel.DEBUG,
#         format_type=LogFormat.READABLE,
#         file_path="logs/api.log",
#         rotation="10 MB",
#         retention="30 days"
#     )
# )
#
# # 带上下文的错误日志
# try:
#     # 某些可能出错的代码
#     result = 1 / 0
# except Exception as e:
#     custom_logger.error(
#         f"操作失败: {str(e)}",
#         operation="division",
#         inputs={"dividend": 1, "divisor": 0},
#         traceback=traceback.format_exc()
#     )


def make_readable_logger(original_logger: AppLogger) -> AppLogger:
    """
    将现有的日志记录器转换为高可读性格式，而不修改原始配置的其他属性

    Args:
        original_logger: 原始日志记录器实例

    Returns:
        配置了高可读性格式的新日志记录器实例
    """
    # 克隆原始配置但设置格式类型为READABLE
    readable_config = LoggerConfig(
        name=original_logger.name,
        level=original_logger.config.level,
        format_type=LogFormat.READABLE,
        file_path=original_logger.config.file_path,
        rotation=original_logger.config.rotation,
        retention=original_logger.config.retention,
        extra_handlers=original_logger.config.extra_handlers,
    )

    # 创建并返回新的日志记录器
    return AppLogger(readable_config)


def enhance_flask_app_logging(app: Flask) -> None:
    """
    增强Flask应用的日志可读性，而不修改现有日志结构

    Args:
        app: Flask应用实例
    """
    # 检查应用是否已经有日志记录器
    if hasattr(app, "logger") and app.logger:
        # 保存原始配置
        original_name = app.logger_name

        # 创建一个新的高可读性格式的日志配置
        log_dir = os.path.join(app.instance_path, "logs")
        log_file = os.path.join(log_dir, "app.log")

        config = LoggerConfig(
            name=original_name,
            level=LogLevel.DEBUG if app.debug else LogLevel.INFO,
            format_type=LogFormat.READABLE,
            file_path=log_file,
            rotation="10 MB",
            retention="30 days",
        )

        # 初始化应用日志记录器
        adapter = FlaskAppLoggerAdapter(app, config)

        # 返回配置好的日志记录器实例
        return adapter.logger

    # 如果应用没有日志记录器，创建一个新的
    return init_app_logger(app)


def make_hybrid_logger(original_logger: AppLogger) -> AppLogger:
    """
    将现有的日志记录器转换为混合格式（终端可读性，文件双格式保存）

    Args:
        original_logger: 原始日志记录器实例

    Returns:
        配置了混合格式的新日志记录器实例
    """
    # 克隆原始配置但设置格式类型为HYBRID
    hybrid_config = LoggerConfig(
        name=original_logger.name,
        level=original_logger.config.level,
        format_type=LogFormat.HYBRID,
        file_path=original_logger.config.file_path,
        rotation=original_logger.config.rotation,
        retention=original_logger.config.retention,
        extra_handlers=original_logger.config.extra_handlers,
    )

    # 创建并返回新的日志记录器
    return AppLogger(hybrid_config)


def fix_log_file_formatting(log_file_path: str) -> None:
    """
    修复现有的日志文件格式，添加正确的换行并清理颜色控制代码

    Args:
        log_file_path: 日志文件路径
    """
    import re

    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 查找所有日志记录的开始（时间戳格式）
        # 格式如: 2025-03-27 00:29:23.374
        timestamp_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})"

        # 在每个时间戳前添加换行符（第一个除外）
        formatted_content = re.sub(r"(?<!^)" + timestamp_pattern, r"\n\1", content)

        # 清理ANSI颜色控制代码，如 \033[32m 和 \033[0m
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        formatted_content = ansi_escape.sub("", formatted_content)

        # 确保文件以换行符结束
        if not formatted_content.endswith("\n"):
            formatted_content += "\n"

        # 写回文件
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(formatted_content)

        print(f"日志文件 {log_file_path} 已成功格式化。")

    except Exception as e:
        print(f"格式化日志文件时出错: {str(e)}")


def clean_all_log_files(log_dir: str) -> None:
    """
    整理指定目录中的所有日志文件

    Args:
        log_dir: 日志目录路径
    """
    import os
    import glob

    # 查找所有日志文件
    log_patterns = ["*.log", "*.json.log", "*.readable.log"]
    log_files = []

    for pattern in log_patterns:
        log_files.extend(glob.glob(os.path.join(log_dir, pattern)))

    if not log_files:
        print(f"在 {log_dir} 中未找到日志文件。")
        return

    # 修复每个日志文件
    for log_file in log_files:
        print(f"正在整理日志文件: {log_file}")
        fix_log_file_formatting(log_file)

    print(f"共整理了 {len(log_files)} 个日志文件。")
