import pytest
from app.models.chat_model import ChatModel
import os
import yaml

def test_process_json_input():
    """测试JSON输入处理功能"""
    # 创建测试用的ChatModel实例
    chat_model = ChatModel()

    # 测试数据
    test_data = {
        "question": "测试问题",
        "documents": [
            {
                "page_content": "测试文档内容",
                "metadata": {"source": "test.txt"}
            }
        ]
    }

    # 处理JSON输入
    result = chat_model.process_json_input(test_data)

    # 验证结果
    assert "问题: 测试问题" in result
    assert "相关文档:" in result
    assert "文档 1:" in result
    assert "```yaml" in result
    assert "page_content: 测试文档内容" in result
    assert "source: test.txt" in result