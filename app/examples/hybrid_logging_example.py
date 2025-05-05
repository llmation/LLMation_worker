"""
HYBRID日志模式示例

本示例展示如何使用HYBRID日志模式，同时在终端显示可读性日志格式，
并将两种格式（JSON和可读性）的日志保存到文件中。
"""

import os
import time
import traceback
from pathlib import Path

from flask import Flask

from app.utils.logger import (
    AppLogger,
    LoggerConfig,
    LogLevel,
    LogFormat,
    default_logger,
    make_hybrid_logger,
    init_app_logger,
)


def demo_hybrid_logging():
    """演示HYBRID日志模式的基本使用"""
    print("\n=== HYBRID日志模式基本使用 ===")

    # 使用默认日志记录器（现在是HYBRID模式）
    default_logger.debug("调试信息将会输出到终端（可读性格式）和两种日志文件")
    default_logger.info("这条信息在终端以可读性格式显示，同时保存在两个日志文件中")
    default_logger.warning("警告消息，终端以颜色标记，两个日志文件分别保存")

    # 带有结构化数据的日志
    complex_data = {
        "user": {"id": "user123", "name": "张三", "roles": ["admin", "editor"]},
        "context": {
            "session_id": "sess_12345",
            "ip": "192.168.1.100",
            "browser": "Chrome/98.0.4758.102",
            "timestamp": time.time(),
        },
    }

    default_logger.info(
        "带有复杂结构化数据的日志",
        data=complex_data,
        request_path="/api/users",
        method="GET",
    )

    # 模拟错误日志
    try:
        # 故意制造错误
        result = 1 / 0
    except Exception as e:
        default_logger.error(
            f"发生错误: {str(e)}",
            error_type=type(e).__name__,
            traceback=traceback.format_exc(),
            context={"operation": "division", "inputs": {"dividend": 1, "divisor": 0}},
        )


def demo_custom_hybrid_logger():
    """演示创建自定义HYBRID日志记录器"""
    print("\n=== 自定义HYBRID日志记录器 ===")

    # 创建临时日志目录
    log_dir = Path("./temp_logs")
    log_dir.mkdir(exist_ok=True)

    # 创建自定义日志记录器
    custom_logger = AppLogger(
        LoggerConfig(
            name="custom_hybrid",
            level=LogLevel.DEBUG,
            format_type=LogFormat.HYBRID,
            file_path=str(log_dir / "custom.log"),
            rotation="5 MB",
            retention="2 days",
        )
    )

    # 使用自定义日志记录器
    custom_logger.info("使用自定义HYBRID日志记录器")
    custom_logger.debug(
        "可以查看temp_logs目录中的文件", log_dir=str(log_dir.absolute())
    )

    print(f"\n日志文件已保存在: {log_dir.absolute()}")
    print("将生成两个日志文件:")
    print(f"1. {log_dir / 'custom.json.log'} - JSON格式")
    print(f"2. {log_dir / 'custom.readable.log'} - 可读性格式")


def demo_convert_existing_logger():
    """演示将现有日志记录器转换为HYBRID模式"""
    print("\n=== 转换现有日志记录器 ===")

    # 创建一个使用JSON格式的日志记录器
    json_logger = AppLogger(
        LoggerConfig(
            name="old_json_logger", level=LogLevel.INFO, format_type=LogFormat.JSON
        )
    )

    # 使用原始日志记录器
    json_logger.info("这是原始的JSON格式日志", source="original")

    # 转换为HYBRID模式
    hybrid_logger = make_hybrid_logger(json_logger)

    # 使用转换后的日志记录器
    hybrid_logger.info("这是转换为HYBRID模式后的日志", source="converted")


def demo_flask_hybrid_logging():
    """演示在Flask应用中使用HYBRID日志模式"""
    print("\n=== Flask应用HYBRID日志模式 ===")

    # 创建一个Flask测试应用
    app = Flask("hybrid_demo")
    app.config["TESTING"] = True

    # 创建临时日志目录
    log_dir = Path("./temp_logs/flask")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 配置应用日志
    logger_config = LoggerConfig(
        name="flask_hybrid",
        level=LogLevel.DEBUG,
        format_type=LogFormat.HYBRID,
        file_path=str(log_dir / "flask_app.log"),
        rotation="5 MB",
        retention="2 days",
    )

    # 初始化应用日志
    app_logger = init_app_logger(app, logger_config)

    # 记录一些应用日志
    app_logger.info("Flask应用已初始化", app_name=app.name)
    app_logger.debug(
        "应用配置信息",
        config={k: v for k, v in app.config.items() if k in ["TESTING", "DEBUG"]},
    )

    print(f"\nFlask应用日志文件已保存在: {log_dir.absolute()}")


if __name__ == "__main__":
    """运行所有HYBRID日志模式示例"""
    print("\nHYBRID日志模式示例")
    print("=" * 50)

    demo_hybrid_logging()
    demo_custom_hybrid_logger()
    demo_convert_existing_logger()
    demo_flask_hybrid_logging()

    print("\n示例完成")
    print("=" * 50)
    print("\n提示：查看temp_logs目录可以找到生成的日志文件")
