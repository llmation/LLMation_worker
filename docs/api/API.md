# LLMation Worker API API 文档


        ## LLMation Worker - 智能对话与RAG服务

        这是一个基于FastAPI和LangChain构建的智能对话后端服务，提供以下核心功能：

        ### 🚀 主要特性
        - **流式对话**: 支持实时流式响应的智能对话
        - **RAG增强**: 基于文档检索的增强生成
        - **多模型支持**: 支持OpenAI和阿里云DashScope模型
        - **文档处理**: 智能文档解析和向量化存储
        - **实时日志**: 完整的请求追踪和日志记录

        ### 📋 API端点
        - `POST /api/workflow/completions` - 流式对话完成
        - `GET /api/rag-test` - RAG功能测试界面
        - `GET /docs` - API文档 (当前页面)
        - `GET /redoc` - 替代文档界面

        ### 🔧 技术栈
        - **框架**: FastAPI + Uvicorn
        - **AI**: LangChain + OpenAI/DashScope
        - **向量存储**: FAISS
        - **日志**: Loguru
        - **验证**: Pydantic

        ### 📖 使用说明
        1. 配置API密钥 (OpenAI_API_KEY, DASHSCOPE_API_KEY)
        2. 启动服务: `python app.py`
        3. 访问测试页面: `/api/rag-test`
        4. 调用API进行对话

        ---
        **版本**: 1.0.0 | **协议**: Apache 2.0
        

**版本**: 1.0.0

## 服务器

- **开发环境**: `http://localhost:12000`
- **生产环境**: `https://work-1-npxugdcnhgrookmw.prod-runtime.all-hands.dev`

## API 端点

### /api/workflow/completions

#### POST

**摘要**: 流式对话完成

**描述**:
处理LLMBackendRequest JSON输入并提供流式聊天响应。
    
    ### 功能特性
    - 支持实时流式响应
    - RAG文档检索增强
    - 多轮对话上下文
    - 智能错误处理
    
    ### 请求格式
    ```json
    {
        "enginePrompt": "系统提示词",
        "conversation": [
            {"type": "user", "content": "用户消息"},
            {"type": "assistant", "content": "助手回复"}
        ],
        "active": {
            "doc1": {"title": "文档标题", "content": "文档内容"}
        },
        "reference": [
            {"type": "document", "key": "ref1", "value": "参考内容"}
        ]
    }
    ```
    
    ### 响应格式
    流式JSON响应，每行包含：
    ```json
    {"content": "生成的文本片段"}
    ```
    
    错误时返回：
    ```json
    {"error": "错误描述"}
    ```

**标签**: 对话, RAG, 流式响应

**请求体**:
- Content-Type: `application/json`

**响应**:
- **200**: 流式JSON响应，包含生成的文本内容
- **422**: Validation Error

---

### /api/rag-test

#### GET

**摘要**: RAG测试页面

**描述**:
返回RAG功能测试的交互式网页界面。
    
    ### 功能特性
    - 可视化JSON输入编辑器
    - 实时流式响应显示
    - 多种示例模板
    - 错误信息展示
    
    ### 使用方法
    1. 在左侧编辑JSON请求
    2. 点击"开始流式输出"按钮
    3. 在右侧查看实时响应
    4. 使用示例按钮快速加载模板
    
    ### 示例模板
    - **基本示例**: 简单对话测试
    - **对话示例**: 多轮对话测试
    - **文档示例**: 带文档的RAG测试

**标签**: 测试, 界面, RAG

**响应**:
- **200**: Successful Response

---

### /

#### GET

**摘要**: Root

**响应**:
- **200**: Successful Response

---

