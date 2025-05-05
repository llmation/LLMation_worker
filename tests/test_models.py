"""
测试数据模型和schema

本模块测试ChatModel类的核心功能：
1. 模型初始化和配置
2. JSON输入处理
3. YAML序列化
4. 流式RAG聊天
"""

import pytest
from app.models.chat_model import ChatModel
import yaml
from unittest.mock import patch, MagicMock, ANY


class TestChatModel:
    """测试ChatModel类的功能"""

    @patch("langchain_openai.ChatOpenAI")
    @patch("langchain_community.embeddings.DashScopeEmbeddings")
    def test_init(self, mock_dashscope, mock_openai):
        """
        测试ChatModel实例化

        验证:
        - ChatModel实例创建成功
        - API密钥正确设置
        - 聊天模型正确初始化

        模拟:
        - ChatOpenAI和DashScopeEmbeddings类
        - 禁用向量存储初始化
        """
        # 配置模拟对象行为
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance

        mock_dashscope_instance = MagicMock()
        mock_dashscope.return_value = mock_dashscope_instance

        # 禁用初始化向量存储的方法
        with patch.object(ChatModel, "_initialize_vectorstore", return_value=None):
            model = ChatModel(openai_api_key="test-key", dashscope_api_key="test-key")
            assert model is not None
            assert model.openai_api_key == "test-key"
            assert model.dashscope_api_key == "test-key"
            assert model.chat is not None

    @patch("langchain_openai.ChatOpenAI")
    @patch("langchain_community.embeddings.DashScopeEmbeddings")
    @patch("langchain_community.vectorstores.InMemoryVectorStore")
    def test_process_json_input(self, mock_vectorstore, mock_dashscope, mock_openai):
        """
        测试JSON输入处理功能

        验证:
        - 最小数据处理正确
        - 完整数据处理正确
        - 嵌入提示、文档引用和对话历史

        模拟:
        - 所有外部依赖：聊天模型、嵌入模型和向量存储
        """
        # 配置模拟对象
        mock_openai.return_value = MagicMock()
        mock_dashscope.return_value = MagicMock()
        mock_vectorstore.return_value = MagicMock()

        # 禁用初始化向量存储的方法
        with patch.object(ChatModel, "_initialize_vectorstore", return_value=None):
            # 创建测试用的ChatModel实例
            chat_model = ChatModel(
                openai_api_key="test-key", dashscope_api_key="test-key"
            )

            # 测试最小数据
            minimal_data = {
                "enginePrompt": "测试提示",
                "conversation": [{"type": "user", "content": "测试内容"}],
            }

            engine_prompt, user_input, conversations = chat_model.process_json_input(
                minimal_data
            )

            # 验证返回值
            assert engine_prompt.startswith("测试提示")
            assert user_input == "测试内容"
            assert len(conversations) == 1
            assert conversations[0]["type"] == "user"
            assert conversations[0]["content"] == "测试内容"

            # 测试完整数据
            complete_data = {
                "enginePrompt": "完整提示",
                "active": {"doc1": {"id": "doc1", "name": "测试文档"}},
                "reference": [
                    {"type": "document", "key": "ref1", "value": "参考文档内容"}
                ],
                "referenceNodes": [{"id": "node1", "name": "参考节点"}],
                "conversation": [{"type": "user", "content": "用户内容"}],
            }

            engine_prompt, user_input, conversations = chat_model.process_json_input(
                complete_data
            )

            # 验证返回值
            assert engine_prompt.startswith("完整提示")
            assert "## 活动文档" in engine_prompt
            assert "## 引用文档" in engine_prompt
            assert "## 引用节点" in engine_prompt
            assert "doc1" in engine_prompt
            assert "参考文档内容" in engine_prompt
            assert "node1" in engine_prompt
            assert user_input == "用户内容"
            assert len(conversations) == 1

    @patch("langchain_openai.ChatOpenAI")
    @patch("langchain_community.embeddings.DashScopeEmbeddings")
    @patch("langchain_community.vectorstores.InMemoryVectorStore")
    def test_yaml_serialization(self, mock_vectorstore, mock_dashscope, mock_openai):
        """
        测试YAML序列化功能

        验证:
        - 嵌套对象正确序列化为YAML
        - YAML格式正确嵌入到提示中

        模拟:
        - 外部依赖同process_json_input测试
        """
        # 配置模拟对象
        mock_openai.return_value = MagicMock()
        mock_dashscope.return_value = MagicMock()
        mock_vectorstore.return_value = MagicMock()

        # 禁用初始化向量存储的方法
        with patch.object(ChatModel, "_initialize_vectorstore", return_value=None):
            # 创建测试用的ChatModel实例
            chat_model = ChatModel(
                openai_api_key="test-key", dashscope_api_key="test-key"
            )

            # 测试数据
            test_data = {
                "id": "doc1",
                "name": "测试文档",
                "nested": {"key": "value", "list": [1, 2, 3]},
            }

            # 通过处理JSON输入来间接测试YAML序列化
            json_input = {
                "enginePrompt": "测试",
                "active": {"doc1": test_data},
                "conversation": [{"type": "user", "content": "测试"}],
            }

            engine_prompt, _, _ = chat_model.process_json_input(json_input)

            # 验证YAML序列化结果
            assert "```yaml" in engine_prompt
            assert "id: doc1" in engine_prompt
            assert "name: 测试文档" in engine_prompt
            assert "key: value" in engine_prompt

    @patch("langchain_openai.ChatOpenAI")
    @patch("langchain_community.embeddings.DashScopeEmbeddings")
    @patch("langchain_community.vectorstores.InMemoryVectorStore")
    @patch("langchain_core.messages.SystemMessage")
    @patch("langchain_core.messages.HumanMessage")
    @patch("langchain_core.messages.AIMessage")
    def test_stream_chat_with_rag_mocked(
        self,
        mock_ai_message,
        mock_human_message,
        mock_system_message,
        mock_vectorstore,
        mock_dashscope,
        mock_openai,
    ):
        """
        使用模拟对象测试流式RAG增强聊天功能

        验证:
        - 流式聊天功能正常工作
        - 返回预期的响应内容
        - 调用了正确的方法

        模拟:
        - 消息类 (SystemMessage, HumanMessage, AIMessage)
        - 嵌入模型与向量存储
        - 聊天模型的流式响应

        注意:
        这个测试使用公共API模拟，避免模拟内部实现细节，确保测试稳定性
        """
        # 设置模拟对象
        mock_chat = MagicMock()
        mock_openai.return_value = mock_chat

        # 模拟消息类
        mock_system_message.return_value = MagicMock()
        mock_human_message.return_value = MagicMock()
        mock_ai_message.return_value = MagicMock()

        # 模拟DashScope的公共接口方法
        mock_dashscope_instance = MagicMock()
        mock_dashscope.return_value = mock_dashscope_instance
        mock_dashscope_instance.embed_documents.return_value = [[0.1, 0.2, 0.3] * 10]
        mock_dashscope_instance.embed_query.return_value = [0.1, 0.2, 0.3] * 10

        # 模拟向量存储
        mock_vectorstore_instance = MagicMock()
        mock_vectorstore.return_value = mock_vectorstore_instance
        mock_vectorstore_instance.similarity_search.return_value = []

        # 创建模拟的流式响应
        mock_response = MagicMock()
        mock_response.content = "模拟的响应"
        mock_chat.stream.return_value = [mock_response]

        # 禁用向量存储初始化
        with patch.object(ChatModel, "_initialize_vectorstore", return_value=None):
            # 创建测试用的ChatModel实例
            chat_model = ChatModel(
                openai_api_key="test-key", dashscope_api_key="test-key"
            )

            # 直接设置属性，确保使用模拟对象
            chat_model.vectorstore = mock_vectorstore_instance
            chat_model.embeddings = mock_dashscope_instance
            chat_model.chat = mock_chat

            # 测试流式聊天功能
            print("开始测试流式聊天...")
            results = list(
                chat_model.stream_chat_with_rag(
                    "系统提示", "用户输入", [{"type": "user", "content": "用户输入"}]
                )
            )

            # 验证结果
            print(f"流式聊天结果: {results}")
            assert len(results) > 0
            assert "模拟的响应" in results[0]  # 检查返回的内容是否包含期望的响应

            # 验证是否调用了stream方法
            mock_chat.stream.assert_called()
