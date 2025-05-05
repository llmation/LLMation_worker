"""
Author: ethereal ethereal49@outlook.com
Date: 2025-03-26 21:11:54
LastEditors: ethereal ethereal49@outlook.com
LastEditTime: 2025-03-26 21:12:38
FilePath: \LLMation_worker\tests\test_validators.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

"""
测试验证器功能
"""

import pytest
from app.utils.validators import validate_llm_backend_request


class TestValidators:
    """测试验证器功能"""

    def test_validate_llm_backend_request_valid(
        self, test_data_minimal, test_data_complete
    ):
        """测试有效的LLMBackendRequest验证"""
        # 测试最小必要数据
        assert validate_llm_backend_request(test_data_minimal) is True

        # 测试完整数据
        assert validate_llm_backend_request(test_data_complete) is True

    def test_validate_llm_backend_request_invalid(self, invalid_test_cases):
        """测试无效的LLMBackendRequest验证"""
        for test_case in invalid_test_cases:
            assert validate_llm_backend_request(test_case["data"]) is False

    def test_validate_llm_backend_request_edge_cases(self):
        """测试LLMBackendRequest验证的边界情况"""
        # None值
        assert validate_llm_backend_request(None) is False

        # 非字典值
        assert validate_llm_backend_request("不是字典") is False
        assert validate_llm_backend_request([]) is False
        assert validate_llm_backend_request(123) is False

        # 缺少必要字段
        assert validate_llm_backend_request({}) is False
        assert validate_llm_backend_request({"enginePrompt": "测试"}) is False
        assert validate_llm_backend_request({"conversation": []}) is False

        # enginePrompt不是字符串
        assert (
            validate_llm_backend_request(
                {
                    "enginePrompt": 123,
                    "conversation": [{"type": "user", "content": "测试"}],
                }
            )
            is False
        )

        # conversation不是列表
        assert (
            validate_llm_backend_request(
                {"enginePrompt": "测试", "conversation": "不是列表"}
            )
            is False
        )

        # conversation中的消息不是字典
        assert (
            validate_llm_backend_request(
                {"enginePrompt": "测试", "conversation": ["不是字典"]}
            )
            is False
        )

        # conversation中没有用户消息
        assert (
            validate_llm_backend_request(
                {
                    "enginePrompt": "测试",
                    "conversation": [{"type": "system", "content": "系统消息"}],
                }
            )
            is False
        )

        # conversation中没有content的用户消息
        assert (
            validate_llm_backend_request(
                {"enginePrompt": "测试", "conversation": [{"type": "user"}]}
            )
            is False
        )
