"""
日志可读性增强示例

本模块展示了如何使用增强的高可读性日志格式，
同时不改变现有的日志结构和接口。
"""

import time
import traceback
from typing import Dict, Any

from flask import Flask

from app.utils.logger import (
    AppLogger,
    LoggerConfig,
    LogLevel,
    LogFormat,
    default_logger,
    make_readable_logger,
    enhance_flask_app_logging,
)


def demo_basic_logging():
    """演示基本日志输出"""
    print("\n=== 基本日志输出 ===")

    # 使用默认日志记录器（现在是READABLE格式）
    default_logger.debug("这是一条调试信息")
    default_logger.info("这是一条普通信息")
    default_logger.warning("这是一条警告信息")
    default_logger.error("这是一条错误信息")
    default_logger.critical("这是一条严重错误信息")


def demo_structured_logging():
    """演示结构化日志输出"""
    print("\n=== 结构化日志输出 ===")

    # 带有额外字段的日志
    user_data = {
        "id": "user123",
        "name": "张三",
        "role": "admin",
        "login_time": time.time(),
    }

    default_logger.info(
        "用户登录成功",
        user=user_data,
        ip="192.168.1.100",
        session_id="sess_12345",
        auth_method="password",
    )

    # 记录异常信息
    try:
        # 模拟异常
        result = 1 / 0
    except Exception as e:
        default_logger.error(
            f"操作失败: {str(e)}",
            operation="division",
            inputs={"dividend": 1, "divisor": 0},
            exception_type=type(e).__name__,
            traceback=traceback.format_exc(),
        )


def demo_convert_existing_logger():
    """演示转换现有日志记录器"""
    print("\n=== 转换现有日志记录器 ===")

    # 创建一个使用JSON格式的日志记录器
    json_logger = AppLogger(
        LoggerConfig(
            name="json_logger", level=LogLevel.INFO, format_type=LogFormat.JSON
        )
    )

    # 输出一些JSON格式的日志
    json_logger.info("这是JSON格式的日志", field1="值1", field2="值2")

    # 将JSON日志记录器转换为可读性格式
    readable_logger = make_readable_logger(json_logger)

    # 输出相同内容但使用可读性格式
    readable_logger.info("这是可读性格式的日志", field1="值1", field2="值2")


def demo_flask_app_logging():
    """演示Flask应用日志可读性增强"""
    print("\n=== Flask应用日志可读性增强 ===")

    # 创建一个Flask应用
    app = Flask("demo_app")

    # 增强应用的日志可读性
    logger = enhance_flask_app_logging(app)

    # 模拟一些应用日志
    logger.info("Flask应用启动")
    logger.info("路由注册完成", routes=["/api/v1/users", "/api/v1/products"])

    # 模拟请求处理日志
    logger.info(
        "处理API请求",
        method="GET",
        path="/api/v1/users",
        query_params={"limit": 10, "offset": 0},
        user_id="admin123",
    )


if __name__ == "__main__":
    """运行所有示例"""
    print("\n日志可读性增强示例")
    print("=" * 50)

    demo_basic_logging()
    demo_structured_logging()
    demo_convert_existing_logger()
    demo_flask_app_logging()

    print("\n示例完成")
    print("=" * 50)
