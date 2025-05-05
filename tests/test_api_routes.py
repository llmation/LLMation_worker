"""
测试API路由和端点

本模块测试API路由和端点功能：
1. 工作流完成端点
2. RAG测试页面
3. 错误处理和边缘情况
4. 聊天模型初始化
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from flask import Response
from app.routes.chat_routes import chat_bp


class TestChatRoutes:
    """测试聊天相关的API路由"""

    @patch("app.routes.chat_routes.chat_model")
    def test_workflow_completions_endpoint(
        self, mock_chat_model, client, test_data_minimal
    ):
        """
        测试工作流完成端点 - 基础测试

        验证:
        - 端点正确处理JSON请求
        - 返回正确的状态码和MIME类型
        - 调用适当的处理方法

        模拟:
        - 聊天模型及其处理方法
        """
        # 配置模拟对象
        mock_chat_model.process_json_input.return_value = ("提示", "用户输入", [])
        mock_chat_model.stream_chat_with_rag.return_value = iter(["测试", "响应"])

        # 发送请求
        response = client.post(
            "/api/workflow/completions",
            json=test_data_minimal,
            content_type="application/json",
        )

        # 验证响应
        assert response.status_code == 200
        assert response.mimetype == "text/event-stream"

        # 验证调用
        mock_chat_model.process_json_input.assert_called_once_with(test_data_minimal)
        mock_chat_model.stream_chat_with_rag.assert_called_once()

    def test_workflow_completions_invalid_json(self, client, invalid_test_cases):
        """
        测试工作流完成端点 - 无效JSON输入

        验证:
        - 端点正确处理无效JSON输入
        - 返回适当的错误状态码(400或500)
        - 响应包含错误信息

        测试:
        - 多种无效输入情况，使用invalid_test_cases夹具
        """
        for test_case in invalid_test_cases:
            # 发送请求
            response = client.post(
                "/api/workflow/completions",
                json=test_case["data"],
                content_type="application/json",
            )

            # 验证响应 - 接受400或500状态码
            assert response.status_code in [400, 500]
            response_data = json.loads(response.data)
            assert "error" in response_data

    def test_workflow_completions_no_json(self, client):
        """
        测试工作流完成端点 - 无JSON输入

        验证:
        - 端点正确处理没有JSON的请求
        - 返回适当的错误状态码(400或500)
        - 响应包含错误信息
        """
        # 发送请求
        response = client.post("/api/workflow/completions")

        # 验证响应 - 接受400或500状态码
        assert response.status_code in [400, 500]
        response_data = json.loads(response.data)
        assert "error" in response_data

    def test_rag_test_page(self, client):
        """
        测试RAG测试页面端点

        验证:
        - 端点正确返回HTML页面
        - 返回200状态码
        - 响应包含HTML文档结构
        """
        # 发送请求
        response = client.get("/api/rag-test")

        # 验证响应
        assert response.status_code == 200
        assert b"<!DOCTYPE html>" in response.data


@patch("app.routes.chat_routes.ChatModel")
def test_initialize_chat_model(mock_chat_model_class, client, app):
    """
    测试请求前初始化聊天模型

    验证:
    - 请求前初始化钩子正确初始化聊天模型
    - 全局chat_model变量被正确设置

    模拟:
    - ChatModel类
    - 在正确的应用上下文中执行测试
    """
    from app.routes.chat_routes import initialize_chat_model, chat_model

    # 设置原始的全局变量状态
    import app.routes.chat_routes

    app.routes.chat_routes.chat_model = None

    # 模拟ChatModel的实例化
    mock_instance = MagicMock()
    mock_chat_model_class.return_value = mock_instance

    # 使用正确的应用上下文 - 这里应该使用客户端的应用
    with client.application.app_context():
        initialize_chat_model()

    # 验证ChatModel被初始化
    mock_chat_model_class.assert_called_once()
    assert app.routes.chat_routes.chat_model is mock_instance


@patch("app.routes.chat_routes.chat_model")
def test_workflow_completions_exception(mock_chat_model, client, test_data_minimal):
    """
    测试工作流完成端点 - 异常处理

    验证:
    - 端点正确处理处理过程中抛出的异常
    - 返回适当的错误状态码(500)
    - 响应包含详细的错误信息

    模拟:
    - 聊天模型抛出异常的情况
    """
    # 配置模拟对象抛出异常
    mock_chat_model.process_json_input.side_effect = Exception("测试异常")

    # 发送请求
    response = client.post(
        "/api/workflow/completions",
        json=test_data_minimal,
        content_type="application/json",
    )

    # 验证响应
    assert response.status_code == 500
    response_data = json.loads(response.data)
    assert "error" in response_data
    assert "测试异常" in response_data["error"]
