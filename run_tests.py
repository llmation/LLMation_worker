#!/usr/bin/env python
"""
项目测试运行器 - 从项目根目录运行
"""

import os
import sys
import subprocess


def run_command(command):
    """运行系统命令并打印输出"""
    print(f"执行命令: {command}")
    # 设置encoding='utf-8'来处理非ASCII字符
    try:
        # 使用直接打印输出的方式，避免编码问题
        result = subprocess.run(
            command, shell=True, check=False, encoding="utf-8", errors="replace"
        )
        return result.returncode
    except Exception as e:
        print(f"命令执行出错: {str(e)}")
        return 1


if __name__ == "__main__":
    # 确保我们在项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 确保python路径包含当前目录
    sys.path.insert(0, script_dir)

    # 确保项目代码可以被导入
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path[0]}")

    if len(sys.argv) < 2:
        print("请指定要运行的测试类型: unit, api, integration, e2e, all")
        sys.exit(1)

    test_type = sys.argv[1].lower()

    # 设置环境变量指定编码
    os.environ["PYTHONIOENCODING"] = "utf-8"

    if test_type == "unit":
        print("运行单元测试...")
        exit_code = run_command(
            "python -m pytest tests/test_models.py tests/test_validators.py -v"
        )
    elif test_type == "api":
        print("运行API测试...")
        exit_code = run_command("python -m pytest tests/test_api_routes.py -v")
    elif test_type == "integration":
        print("运行集成测试...")
        exit_code = run_command(
            "python -m pytest tests/test_integration.py::TestMockIntegration -v"
        )
    elif test_type == "e2e":
        print("运行端到端测试...")
        os.environ["RUN_E2E_TESTS"] = "1"
        exit_code = run_command(
            "python -m pytest tests/test_integration.py::TestRealIntegration -v"
        )
    elif test_type == "all":
        print("运行所有测试...")
        exit_code = run_command("python -m pytest tests/ -v")
    else:
        print(f"未知的测试类型: {test_type}")
        print("请指定: unit, api, integration, e2e, all")
        sys.exit(1)

    # 返回命令的退出代码
    sys.exit(exit_code)
