# LLMation Worker 测试文档

本文档详细说明 LLMation Worker 项目的测试框架、策略和最佳实践。

## 测试结构

```text
tests/
├── conftest.py         # 测试配置和共享夹具
├── test_api_routes.py  # API路由测试
├── test_integration.py # 集成和端到端测试
├── test_models.py      # 数据模型测试
└── test_validators.py  # 验证器测试
```

## 测试类型

项目包含以下类型的测试:

1. **单元测试**
   - 测试各组件的独立功能
   - 对外部依赖进行完全模拟
   - 位于 `test_models.py` 和 `test_validators.py`

2. **API测试**
   - 测试API端点和路由
   - 使用Flask测试客户端
   - 位于 `test_api_routes.py`

3. **集成测试**
   - 测试多个组件的协同工作
   - 包含模拟集成测试和真实E2E测试
   - 位于 `test_integration.py`

## 测试夹具

所有共享夹具在 `conftest.py` 中定义:

- **全局模拟夹具**: `no_real_requests` - 自动应用于所有测试
- **应用夹具**: `app`, `client` - 创建Flask应用和测试客户端
- **模拟API密钥**: `mock_openai_api_key`, `mock_dashscope_api_key`
- **测试数据**: `test_data_minimal`, `test_data_complete`, `invalid_test_cases`

## 模拟策略

项目采用以下模拟策略，确保测试的稳定性和可靠性:

### 1. 接口级模拟

模拟在公共API级别进行，而不是内部实现细节，例如:

```python
# 好的做法 - 模拟公共接口
patch("langchain_community.embeddings.DashScopeEmbeddings.embed_documents", ...)

# 避免 - 模拟内部实现
patch("dashscope.embeddings.client.call", ...)
```

### 2. 全局模拟配置

使用自动应用的夹具禁用真实网络请求:

```python
@pytest.fixture(autouse=True)
def no_real_requests():
    with (
        patch("requests.get"),
        patch("requests.post"),
        ...
    ):
        yield
```

### 3. 直接属性注入

对于复杂对象，直接设置实例属性，而不是尝试拦截初始化时的调用:

```python
# 创建测试实例
chat_model = ChatModel(...)

# 直接设置属性，确保使用模拟对象
chat_model.vectorstore = mock_vectorstore_instance
chat_model.embeddings = mock_dashscope_instance
chat_model.chat = mock_chat
```

## 测试最佳实践

### 详细测试注释

每个测试函数应包含详细注释，说明:

- 验证什么功能
- 模拟哪些组件
- 测试的核心断言

例如:

```python
"""
测试流式RAG增强聊天功能

验证:
- 流式聊天功能正常工作
- 返回预期的响应内容
- 调用了正确的方法

模拟:
- 消息类 (SystemMessage, HumanMessage, AIMessage)
- 嵌入模型与向量存储
- 聊天模型的流式响应
"""
```

### 避免模拟内部实现

测试应该关注组件的行为和接口，而不是内部实现细节:

- **使用公共方法**: 模拟组件的公共方法，而不是私有方法
- **避免内部依赖**: 不要模拟实现特定的内部依赖
- **基于行为测试**: 验证组件的行为和输出，而不是实现细节

### 运行测试

从项目根目录运行测试:

```bash
# 运行所有测试
python run_tests.py all

# 运行特定类型的测试
python run_tests.py unit
python run_tests.py api
python run_tests.py integration

# 运行端到端测试(需要真实API密钥)
RUN_E2E_TESTS=1 python run_tests.py all
```

## 常见问题

### 测试失败与API密钥

如果测试因为API密钥问题失败，可能是:

1. 模拟策略不完整，有真实网络请求泄漏
2. E2E测试运行但没有提供有效API密钥

解决方法:

- 检查模拟配置
- 确认使用接口级模拟
- 如果需要E2E测试，设置有效的API密钥环境变量

### 模拟混淆

如果测试由于模拟混淆而失败:

1. 确认只在适当的测试范围内模拟
2. 避免全局模拟与局部模拟冲突
3. 使用 `patch.object` 精确控制模拟范围

### 测试隔离

确保测试相互独立:

1. 不要在测试之间共享可变状态
2. 使用夹具进行正确的设置和清理
3. 避免测试顺序依赖
