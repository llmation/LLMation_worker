# LLMation Worker

基于 Flask 和 LangChain 的大语言模型处理 Worker，用于处理 LLMation 系统的对话请求。

## 项目结构

```text
LLMation_worker/
├── app/                    # 应用主目录
│   ├── models/             # 数据模型
│   ├── routes/             # API路由
│   ├── services/           # 服务层
│   ├── templates/          # HTML模板
│   ├── utils/              # 工具函数
│   └── __init__.py         # 应用初始化
├── docs/                   # 文档目录
├── tests/                  # 测试目录
├── venv/                   # 虚拟环境
├── app.py                  # 应用入口
├── requirements.txt        # 依赖列表
├── LICENSE                 # 许可证
└── README.md               # 项目说明
```

## 功能特性

- 基于 OPENAI_API 和 DASHSCOPE API 的大语言模型对话
- 支持 RAG（检索增强生成）文档处理
- 支持流式响应输出
- 支持 JSON 输入转换为自然语言+YAML 提示词
- 文档序列化为 YAML 格式
- 工作流程处理功能

## 安装与运行

### 环境准备

1. 确保已安装 Python 3.8+
2. 创建并激活虚拟环境：
   - Windows: `.\.venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
3. 安装依赖：

```bash
pip install -r requirements.txt
```

### 配置

创建`.env`文件，设置必要的环境变量：

```env
OPENAI_API_KEY
DASHSCOPE_API_KEY
SECRET_KEY
```

### 运行

```bash
python app.py
```

服务将在`http://localhost:5000`启动。

## API 接口

### 1. 工作流程处理接口（流式）

- **URL**: `/api/workflow/completions`
- **方法**: POST
- **描述**: 处理工作流 JSON 请求并使用流式响应返回大语言模型生成内容
- **请求体**:

  ```json
  {
    "enginePrompt": "引擎提示内容...",
    "active": {
      "doc1": {
        "id": "doc1",
        "name": "示例文档",
        "description": "这是一个示例",
        "engine": "model-name",
        "effects": [],
        "inputs": {},
        "nodes": [],
        "outputs": {}
      }
    },
    "reference": [
      {
        "type": "document",
        "key": "doc-ref-1",
        "value": "参考文档内容"
      }
    ],
    "referenceNodes": [
      {
        "id": "ref1",
        "name": "参考节点",
        "description": "参考节点描述",
        "engine": "model-name",
        "effects": [],
        "inputs": {},
        "nodes": [],
        "outputs": {}
      }
    ],
    "conversation": [
      {
        "type": "user",
        "content": "用户输入"
      }
    ]
  }
  ```

- **响应**: Server-Sent Events (SSE)

  ```json
  { "content": "模型生成的内容片段..." }
  ```

### 2. RAG 测试页面

- **URL**: `/api/rag-test`
- **方法**: GET
- **描述**: 返回 RAG 测试用的 HTML 页面
- **响应**: HTML 页面

## 使用的主要依赖

- Flask: Web 应用框架
- LangChain: 大语言模型应用开发框架
- LangChain-OpenAI: OpenAI 模型集成
- Python-dotenv: 环境变量管理
- Pydantic: 数据验证
- PyYAML: YAML 数据处理
- DashScope: 灵积模型 API 集成

## 测试

项目使用 Pytest 进行单元测试和集成测试。测试包括：

### 测试类型

- **单元测试**: 测试各个组件的独立功能
- **API 测试**: 测试 API 端点
- **集成测试**: 测试多个组件协同工作

### 运行测试

使用以下命令运行测试：

```bash
# 从项目根目录运行
python run_tests.py unit    # 运行单元测试
python run_tests.py api     # 运行API测试
python run_tests.py integration  # 运行集成测试
python run_tests.py all     # 运行所有测试
```

### 测试策略

项目采用以下测试策略：

1. **完全模拟外部依赖**：所有外部 API 调用（OpenAI、DashScope）都被模拟，确保测试不依赖于真实 API 密钥
2. **接口级模拟**：模拟在公共 API 级别进行，而不是内部实现细节，提高测试稳定性
3. **全局模拟配置**：在`conftest.py`中提供全局模拟配置，简化测试设置
4. **隔离测试**：通过模拟确保测试相互隔离，不影响其他测试

### 端到端测试

端到端测试默认被跳过，如果需要运行，设置环境变量`RUN_E2E_TESTS=1`，并确保提供了有效的 API 密钥。

```bash
# 运行端到端测试(需要真实API密钥)
RUN_E2E_TESTS=1 python run_tests.py all
```

## 许可证

本项目采用 MIT 许可证。详情请查看 LICENSE 文件。
