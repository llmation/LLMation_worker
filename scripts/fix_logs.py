#!/usr/bin/env python
"""
日志文件格式修复工具

该脚本用于修复已经生成的混乱日志文件，添加正确的换行符。
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.utils import clean_all_log_files, fix_log_file_formatting


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="修复日志文件格式")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", help="要修复的日志文件路径")
    group.add_argument("--dir", "-d", help="要修复的日志目录路径")
    group.add_argument(
        "--auto", "-a", action="store_true", help="自动查找项目中的日志文件进行修复"
    )

    return parser.parse_args()


def find_instance_logs():
    """查找项目中的instance/logs目录"""
    instance_dir = project_root / "instance"
    if instance_dir.exists():
        logs_dir = instance_dir / "logs"
        if logs_dir.exists() and logs_dir.is_dir():
            return str(logs_dir)
    return None


def main():
    """主函数"""
    args = parse_args()

    if args.file:
        # 修复单个日志文件
        if not os.path.isfile(args.file):
            print(f"错误：文件 {args.file} 不存在")
            return 1

        print(f"正在修复日志文件: {args.file}")
        fix_log_file_formatting(args.file)

    elif args.dir:
        # 修复目录中的所有日志文件
        if not os.path.isdir(args.dir):
            print(f"错误：目录 {args.dir} 不存在")
            return 1

        print(f"正在修复目录中的所有日志文件: {args.dir}")
        clean_all_log_files(args.dir)

    elif args.auto:
        # 自动查找并修复项目中的日志文件
        logs_dir = find_instance_logs()
        if not logs_dir:
            print("错误：未能自动找到项目的日志目录，请手动指定日志文件或目录")
            return 1

        print(f"已自动找到日志目录: {logs_dir}")
        clean_all_log_files(logs_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
