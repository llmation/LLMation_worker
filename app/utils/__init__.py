"""应用工具模块"""

from app.utils.logger import (
    default_logger,
    AppLogger,
    LoggerConfig,
    LogLevel,
    LogFormat,
    init_app_logger,
    make_readable_logger,
    make_hybrid_logger,
    fix_log_file_formatting,
    clean_all_log_files,
)

# 导出常用实例和类型供简便使用
logger = default_logger
