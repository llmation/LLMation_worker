"""
测试入口文件 - 用于运行各种测试套件
"""

import os
import sys
import pytest

# 获取当前文件所在目录的父目录(项目根目录)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 获取测试目录的绝对路径
TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def run_unit_tests():
    """运行单元测试"""
    # 切换到项目根目录
    os.chdir(ROOT_DIR)
    # 使用测试文件的完整路径
    test_files = [
        os.path.join(TEST_DIR, "test_models.py"),
        os.path.join(TEST_DIR, "test_validators.py"),
    ]
    pytest.main(["-xvs"] + test_files)


def run_api_tests():
    """运行API测试"""
    os.chdir(ROOT_DIR)
    test_file = os.path.join(TEST_DIR, "test_api_routes.py")
    pytest.main(["-xvs", test_file])


def run_integration_tests():
    """运行集成测试"""
    os.chdir(ROOT_DIR)
    test_module = "tests.test_integration::TestMockIntegration"
    pytest.main(["-xvs", test_module])


def run_e2e_tests():
    """运行端到端测试"""
    os.chdir(ROOT_DIR)
    os.environ["RUN_E2E_TESTS"] = "1"
    test_module = "tests.test_integration::TestRealIntegration"
    pytest.main(["-xvs", test_module])


def run_all_tests():
    """运行所有测试（不包括真实API测试）"""
    os.chdir(ROOT_DIR)
    # 使用模块名而不是文件路径
    pytest.main(
        [
            "-xvs",
            "tests.test_models",
            "tests.test_validators",
            "tests.test_api_routes",
            "tests.test_integration::TestMockIntegration",
        ]
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python run_tests.py [unit|api|integration|e2e|all]")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "unit":
        run_unit_tests()
    elif command == "api":
        run_api_tests()
    elif command == "integration":
        run_integration_tests()
    elif command == "e2e":
        run_e2e_tests()
    elif command == "all":
        run_all_tests()
    else:
        print(f"未知命令: {command}")
        print("用法: python run_tests.py [unit|api|integration|e2e|all]")
        sys.exit(1)
