"""
集成测试 - 测试完整的系统流程
"""

import pytest
import json
import os
from unittest.mock import patch, MagicMock


class TestIntegration:
    """系统集成测试"""

    @pytest.mark.skipif(
        not os.environ.get("RUN_INTEGRATION_TESTS"),
        reason="需要设置RUN_INTEGRATION_TESTS环境变量才能运行集成测试",
    )
    def test_workflow_completions_integration(self, client, test_data_minimal):
        """工作流完成端点的集成测试"""
        # 确保有API密钥
        assert os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "DASHSCOPE_API_KEY"
        ), "需要设置OPENAI_API_KEY或DASHSCOPE_API_KEY环境变量"

        # 发送请求
        response = client.post(
            "/api/workflow/completions",
            json=test_data_minimal,
            content_type="application/json",
        )

        # 验证响应
        assert response.status_code == 200
        assert response.mimetype == "text/event-stream"

        # 尝试读取前几个响应片段
        data = b""
        for chunk in response.response:
            data += chunk
            # 读取一小部分数据后停止
            if len(data) > 100:
                break

        # 验证响应包含JSON数据
        assert b"{" in data
        assert b"content" in data

    @pytest.mark.skipif(
        not os.environ.get("RUN_E2E_TESTS"),
        reason="需要设置RUN_E2E_TESTS环境变量才能运行端到端测试",
    )
    def test_full_conversation_flow(self, client):
        """测试完整的对话流程"""
        # 确保有API密钥
        assert os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "DASHSCOPE_API_KEY"
        ), "需要设置OPENAI_API_KEY或DASHSCOPE_API_KEY环境变量"

        # 准备对话数据
        conversation_data = {
            "enginePrompt": "你是一个友好的AI助手。",
            "conversation": [{"type": "user", "content": "你好，介绍一下自己。"}],
        }

        # 发送第一个请求
        response1 = client.post(
            "/api/workflow/completions",
            json=conversation_data,
            content_type="application/json",
        )

        # 验证响应
        assert response1.status_code == 200

        # 准备第二轮对话数据
        conversation_data["conversation"].append(
            {"type": "assistant", "content": "你好！我是AI助手，很高兴为你服务。"}
        )
        conversation_data["conversation"].append(
            {"type": "user", "content": "谢谢，你能做什么?"}
        )

        # 发送第二个请求
        response2 = client.post(
            "/api/workflow/completions",
            json=conversation_data,
            content_type="application/json",
        )

        # 验证响应
        assert response2.status_code == 200

    @pytest.mark.skipif(
        not os.environ.get("RUN_E2E_TESTS"),
        reason="需要设置RUN_E2E_TESTS环境变量才能运行端到端测试",
    )
    def test_rag_functionality(self, client):
        """测试RAG功能"""
        # 确保有API密钥
        assert os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "DASHSCOPE_API_KEY"
        ), "需要设置OPENAI_API_KEY或DASHSCOPE_API_KEY环境变量"

        # 准备带有参考文档的数据
        rag_data = {
            "enginePrompt": "你是一个能够使用参考文档回答问题的AI助手。",
            "reference": [
                {
                    "type": "document",
                    "key": "测试文档",
                    "value": "人工智能(AI)是计算机科学的一个分支，旨在开发能够执行通常需要人类智能的任务的系统。",
                }
            ],
            "conversation": [
                {"type": "user", "content": "根据参考文档，什么是人工智能？"}
            ],
        }

        # 发送请求
        response = client.post(
            "/api/workflow/completions", json=rag_data, content_type="application/json"
        )

        # 验证响应
        assert response.status_code == 200

        # 尝试读取部分响应
        data = b""
        for chunk in response.response:
            data += chunk
            if len(data) > 150:
                break

        # 验证响应中包含相关内容
        data_str = data.decode("utf-8", errors="ignore")
        assert "人工智能" in data_str or "AI" in data_str


class TestMockIntegration:
    """使用模拟对象的集成测试"""

    @patch("app.models.chat_model.ChatOpenAI")
    @patch("app.models.chat_model.DashScopeEmbeddings")
    @patch("app.models.chat_model.InMemoryVectorStore")
    @patch("app.models.chat_model.ChatModel._initialize_vectorstore")
    @patch("app.routes.chat_routes.chat_model")
    def test_workflow_mock_integration(
        self,
        mock_chat_model,
        mock_init_vs,
        mock_vectorstore,
        mock_dashscope,
        mock_openai,
        client,
        test_data_complete,
    ):
        """使用模拟对象的工作流集成测试"""
        # 配置模拟对象
        mock_chat_model.process_json_input.return_value = (
            "系统提示",
            "用户输入",
            [{"type": "user", "content": "用户输入"}],
        )
        mock_chat_model.stream_chat_with_rag.return_value = iter(
            ["这是", "模拟的", "响应"]
        )

        # 强制设置模拟API密钥环境变量
        os.environ["OPENAI_API_KEY"] = "sk-mock-key-for-testing"
        os.environ["DASHSCOPE_API_KEY"] = "mock-dashscope-key-for-testing"

        # 发送请求
        response = client.post(
            "/api/workflow/completions",
            json=test_data_complete,
            content_type="application/json",
        )

        # 验证响应
        assert response.status_code == 200

        # 读取响应数据
        data = b""
        for chunk in response.response:
            data += chunk

        # 解码响应内容并打印
        data_str = data.decode("utf-8", errors="replace")
        print(f"响应内容: {data_str[:200]}")

        # 在JSON解析之前检查是否为空
        if not data_str.strip():
            assert False, "接收到空响应"

        # 检查是否包含内容
        found = False
        # 1. 检查直接包含文本
        for expected in ["这是", "模拟的", "响应"]:
            if expected in data_str:
                found = True
                break

        # 2. 如果没有直接找到，尝试解析JSON
        if not found:
            try:
                chunks = data_str.strip().split("\n\n")
                for chunk in chunks:
                    if chunk:
                        parsed = json.loads(chunk)
                        content = parsed.get("content", "")
                        if any(
                            expected in content
                            for expected in ["这是", "模拟的", "响应"]
                        ):
                            found = True
                            break
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")

        assert found, "响应中未找到期望的内容"

        # 验证模拟函数调用
        mock_chat_model.process_json_input.assert_called_once_with(test_data_complete)
        mock_chat_model.stream_chat_with_rag.assert_called_once()


# 以下测试需要真实的API密钥，默认跳过
@pytest.mark.skipif(
    not os.environ.get("RUN_E2E_TESTS"),
    reason="需要设置RUN_E2E_TESTS环境变量和有效的API密钥才能运行端到端测试",
)
class TestRealIntegration:
    """使用真实API的集成测试"""

    def test_workflow_completions_integration(self, client, test_data_minimal):
        """工作流完成端点的集成测试"""
        # 确保有API密钥
        assert os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "DASHSCOPE_API_KEY"
        ), "需要设置OPENAI_API_KEY或DASHSCOPE_API_KEY环境变量"

        # 发送请求
        response = client.post(
            "/api/workflow/completions",
            json=test_data_minimal,
            content_type="application/json",
        )

        # 验证响应
        assert response.status_code == 200
        assert response.mimetype == "text/event-stream"

        # 尝试读取前几个响应片段
        data = b""
        for chunk in response.response:
            data += chunk
            # 读取一小部分数据后停止
            if len(data) > 100:
                break

        # 验证响应包含JSON数据
        assert b"{" in data
        assert b"content" in data

    def test_full_conversation_flow(self, client):
        """测试完整的对话流程"""
        # 确保有API密钥
        assert os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "DASHSCOPE_API_KEY"
        ), "需要设置OPENAI_API_KEY或DASHSCOPE_API_KEY环境变量"

        # 准备对话数据
        conversation_data = {
            "enginePrompt": "你是一个友好的AI助手。",
            "conversation": [{"type": "user", "content": "你好，介绍一下自己。"}],
        }

        # 发送第一个请求
        response1 = client.post(
            "/api/workflow/completions",
            json=conversation_data,
            content_type="application/json",
        )

        # 验证响应
        assert response1.status_code == 200

        # 准备第二轮对话数据
        conversation_data["conversation"].append(
            {"type": "assistant", "content": "你好！我是AI助手，很高兴为你服务。"}
        )
        conversation_data["conversation"].append(
            {"type": "user", "content": "谢谢，你能做什么?"}
        )

        # 发送第二个请求
        response2 = client.post(
            "/api/workflow/completions",
            json=conversation_data,
            content_type="application/json",
        )

        # 验证响应
        assert response2.status_code == 200
