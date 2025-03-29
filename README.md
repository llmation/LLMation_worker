# LLMation Worker

基于 Flask 和 LangChain 的大语言模型处理 Worker，用于处理 LLMation 系统的对话请求。
TODOlist 在 TODO.md

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
├── instance/               # 实例特定文件(日志等)
├── scripts/                # 维护脚本
├── tests/                  # 测试目录
├── .venv/                  # 虚拟环境
├── app.py                  # 应用入口
├── pyproject.toml          # 项目定义和依赖
├── requirements.txt        # 依赖列表
├── uv.lock                 # UV锁定文件
├── pytest.ini              # 测试配置
├── run_tests.py            # 测试运行脚本
├── LICENSE                 # Apache 2.0许可证
└── README.md               # 项目说明
```

## 功能特性

- 基于 OPENAI_API 和 DASHSCOPE API 的大语言模型对话
- 支持 RAG（检索增强生成）文档处理
- 支持流式响应输出
- 支持 JSON 输入转换为自然语言+YAML 提示词
- 文档序列化为 YAML 格式
- 工作流程处理功能
- 高级日志记录系统（JSON/结构化/可读性格式）

## 安装与运行

### 环境准备

1. 确保已安装 Python 3.11+
2. 使用 UV 进行环境管理和依赖安装（推荐）：

```bash
# 安装 UV 工具
pip install uv

# 创建虚拟环境
uv venv

# 激活虚拟环境
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 使用 uv 同步依赖
uv pip sync requirements.txt
```

或者使用传统的 pip 方式：

```bash
# 创建并激活虚拟环境
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 配置

创建`.env`文件，设置必要的环境变量：

```env
OPENAI_API_KEY=your_openai_api_key
DASHSCOPE_API_KEY=your_dashscope_api_key
SECRET_KEY=your_secret_key
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

## 日志管理

项目集成了高级日志系统，支持多种格式：

- JSON 格式：用于机器处理
- 可读性格式：方便开发人员阅读
- 混合模式：终端可读性 + 文件双格式

日志文件位于 `instance/logs/` 目录中。

## 使用的主要依赖

- Flask 3.1.0: Web 应用框架
- LangChain 0.3.18: 大语言模型应用开发框架
- Langchain-OpenAI 0.3.4: OpenAI 模型集成
- Loguru 0.7.2: 高级日志系统
- Python-dotenv 1.0.1: 环境变量管理
- Pydantic 2.10.6: 数据验证
- PyYAML 6.0.2: YAML 数据处理
- DashScope: 灵积模型 API

## 测试

运行测试：

```bash
# 使用pytest直接运行测试
pytest

# 或使用自定义测试脚本
python run_tests.py
```

## 许可证

本项目采用 Apache License 2.0 许可证。详情请查看 LICENSE 文件。
