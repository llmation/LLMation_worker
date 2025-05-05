"""
Pytest配置文件 - 提供测试夹具和共享资源

本配置文件包含所有测试所需的共享夹具和全局设置，包括：
1. 全局模拟配置，确保测试不会发起真实网络请求
2. Flask应用实例和测试客户端
3. 模拟API密钥
4. 测试数据模板
"""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock
from flask import Flask
from dotenv import load_dotenv
from langchain_openai.chat_models.base import ChatOpenAI

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 加载环境变量
load_dotenv()

# 导入应用工厂函数和ChatModel
from app import create_app  # noqa: E402
from app.models.chat_model import ChatModel  # noqa: E402


# 全局禁用所有真实网络请求
@pytest.fixture(autouse=True)
def no_real_requests():
    """
    全局禁用所有可能的网络请求

    这个夹具会自动应用于所有测试，它通过模拟以下组件来防止真实网络请求：
    1. 所有requests模块的HTTP方法(GET, POST等)
    2. LangChain的DashScopeEmbeddings嵌入方法(embed_documents, embed_query)
    3. LangChain的ChatOpenAI模型方法(_generate, _stream)
    4. ChatModel的向量存储初始化方法

    注意：模拟策略是直接模拟公共API而不是内部实现，这样可以保持测试的稳定性
    """
    with (
        patch("requests.get"),
        patch("requests.post"),
        patch("requests.put"),
        patch("requests.delete"),
        patch("requests.patch"),
        # 模拟DashScope嵌入 - 在LangChain接口层面模拟，而不是底层实现
        patch(
            "langchain_community.embeddings.DashScopeEmbeddings.embed_documents",
            return_value=[[0.1, 0.2, 0.3, 0.4, 0.5] * 20 for _ in range(10)],
        ),
        patch(
            "langchain_community.embeddings.DashScopeEmbeddings.embed_query",
            return_value=[0.1, 0.2, 0.3, 0.4, 0.5] * 20,
        ),
        # 模拟OpenAI API调用 - 返回固定的模拟响应
        patch("langchain_openai.chat_models.base.ChatOpenAI._generate"),
        patch(
            "langchain_openai.chat_models.base.ChatOpenAI._stream",
            return_value=[MagicMock(content="模拟的响应")],
        ),
        # 模拟向量存储初始化 - 避免文件系统操作
        patch.object(ChatModel, "_initialize_vectorstore", return_value=None),
    ):
        yield


@pytest.fixture
def app():
    """
    创建Flask应用实例

    返回一个配置为测试模式的Flask应用，包含测试用的模拟API密钥
    """
    app = create_app(
        test_config={
            "TESTING": True,
            "DEBUG": False,
            "OPENAI_API_KEY": "sk-mock-key-for-testing",
            "DASHSCOPE_API_KEY": "mock-dashscope-key-for-testing",
        }
    )
    yield app


@pytest.fixture
def client(app):
    """
    创建测试客户端

    基于app夹具创建Flask测试客户端，用于发送模拟HTTP请求
    """
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_openai_api_key():
    """
    创建一个模拟的OpenAI API密钥

    临时设置环境变量，测试结束后还原原始状态
    """
    original = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "sk-mock-key-for-testing"
    yield
    if original:
        os.environ["OPENAI_API_KEY"] = original
    else:
        del os.environ["OPENAI_API_KEY"]


@pytest.fixture
def mock_dashscope_api_key():
    """
    创建一个模拟的灵积API密钥

    临时设置环境变量，测试结束后还原原始状态
    """
    original = os.environ.get("DASHSCOPE_API_KEY")
    os.environ["DASHSCOPE_API_KEY"] = "mock-dashscope-key-for-testing"
    yield
    if original:
        os.environ["DASHSCOPE_API_KEY"] = original
    else:
        del os.environ["DASHSCOPE_API_KEY"]


@pytest.fixture
def test_data_minimal():
    """
    最小必要请求数据

    提供包含必要字段的最小请求数据，用于简单的端点测试
    """
    return {
        "enginePrompt": "你是一个AI助手",
        "conversation": [{"type": "user", "content": "你好，这是一个测试"}],
    }


@pytest.fixture
def test_data_complete():
    """
    包含所有字段的完整请求数据

    提供包含所有可选字段的完整请求数据，用于全面测试处理逻辑
    """
    return {
        "enginePrompt": "你是一个专业的AI工作流助手，请根据提供的文档和上下文回答问题。",
        "active": {
            "doc1": {
                "id": "doc1",
                "name": "主文档",
                "description": "这是测试文档",
                "engine": "openai",
                "effects": ["effect1"],
                "inputs": {"input1": {"type": "string", "value": "测试输入"}},
                "nodes": [{"id": "node1", "type": "text", "content": "节点内容"}],
                "outputs": {"output1": {"type": "string", "value": "测试输出"}},
            }
        },
        "reference": [
            {"type": "document", "key": "ref-doc-1", "value": "这是参考文档内容。"}
        ],
        "referenceNodes": [
            {
                "id": "refNode1",
                "name": "参考节点",
                "description": "参考节点描述",
                "engine": "openai",
                "effects": [],
                "inputs": {},
                "nodes": [],
                "outputs": {},
            }
        ],
        "conversation": [
            {"type": "system", "content": "系统初始化消息"},
            {"type": "assistant", "content": "你好！有什么可以帮助你的？"},
            {"type": "user", "content": "请分析文档并总结关键信息"},
        ],
    }


@pytest.fixture
def invalid_test_cases():
    """
    无效请求数据测试用例

    提供一组无效的请求数据，用于测试错误处理和验证逻辑
    """
    return [
        {
            "name": "缺少enginePrompt",
            "data": {"conversation": [{"type": "user", "content": "测试"}]},
        },
        {"name": "缺少conversation", "data": {"enginePrompt": "测试提示"}},
        {
            "name": "conversation不是列表",
            "data": {"enginePrompt": "测试提示", "conversation": "不是列表"},
        },
        {
            "name": "没有用户消息",
            "data": {
                "enginePrompt": "测试提示",
                "conversation": [{"type": "system", "content": "系统消息"}],
            },
        },
    ]
