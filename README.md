# LLMation Worker

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

🚀 **LLMation Worker** 是一个基于 FastAPI 和 LangChain 构建的智能对话后端服务，提供流式响应和RAG（检索增强生成）功能。

## ✨ 主要特性

- 🔄 **流式对话**: 支持实时流式响应的智能对话
- 📚 **RAG增强**: 基于文档检索的增强生成
- 🤖 **多模型支持**: 支持OpenAI和阿里云DashScope模型
- 📄 **文档处理**: 智能文档解析和向量化存储
- 📊 **实时日志**: 完整的请求追踪和日志记录
- 🌐 **现代架构**: FastAPI + 异步支持 + 自动文档生成

## 🚀 快速开始

### 环境要求

- Python 3.8+
- pip 或 poetry

### 安装依赖

```bash
# 使用 pip
pip install -r requirements.txt

# 或使用 poetry
poetry install
```

### 配置环境变量

```bash
# 可选：OpenAI API密钥
export OPENAI_API_KEY="your-openai-api-key"

# 可选：阿里云DashScope API密钥
export DASHSCOPE_API_KEY="your-dashscope-api-key"
```

### 启动服务

```bash
python app.py
```

服务将在 `http://localhost:12000` 启动。

## 📋 API 文档

### 自动生成的文档

- **Swagger UI**: http://localhost:12000/docs
- **ReDoc**: http://localhost:12000/redoc
- **RAG测试页面**: http://localhost:12000/api/rag-test

### 主要端点

#### POST /api/workflow/completions

流式对话完成端点，支持RAG增强。

**请求示例**:
```json
{
    "enginePrompt": "你是一个友好的AI助手",
    "conversation": [
        {"type": "user", "content": "你好"}
    ]
}
```

**响应**: 流式JSON，每行包含生成的文本片段。

#### GET /api/rag-test

返回RAG功能测试的交互式网页界面。

## 🔧 技术栈

- **Web框架**: [FastAPI](https://fastapi.tiangolo.com/) - 现代、快速的Python Web框架
- **AI框架**: [LangChain](https://langchain.com/) - LLM应用开发框架
- **向量存储**: [FAISS](https://github.com/facebookresearch/faiss) - 高效相似性搜索
- **日志系统**: [Loguru](https://github.com/Delgan/loguru) - 现代Python日志库
- **数据验证**: [Pydantic](https://pydantic.dev/) - 数据验证和设置管理
- **ASGI服务器**: [Uvicorn](https://www.uvicorn.org/) - 高性能ASGI服务器

## 📁 项目结构

```
LLMation_worker/
├── app/                    # 应用主目录
│   ├── __init__.py        # FastAPI应用工厂
│   ├── models/            # 数据模型
│   ├── routes/            # API路由
│   ├── utils/             # 工具函数
│   └── templates/         # 模板文件
├── docs/                  # 文档目录
├── tests/                 # 测试文件
├── scripts/               # 脚本文件
├── app.py                 # 应用入口
├── requirements.txt       # 依赖列表
└── README.md             # 项目说明
```

## 🧪 测试

```bash
# 运行所有测试
python -m pytest

# 运行特定测试
python -m pytest tests/test_api_routes.py

# 生成覆盖率报告
python -m pytest --cov=app
```

## 📚 文档生成

```bash
# 生成所有文档
python scripts/generate_docs.py

# 生成的文档位置：
# - docs/api/openapi.json    # OpenAPI JSON规范
# - docs/api/openapi.yaml    # OpenAPI YAML规范
# - docs/api/API.md          # Markdown API文档
# - docs/api/index.html      # HTML静态文档
```

## 🔧 配置

### 环境变量

| 变量名 | 描述 | 默认值 | 必需 |
|--------|------|--------|------|
| `OPENAI_API_KEY` | OpenAI API密钥 | - | 否 |
| `DASHSCOPE_API_KEY` | 阿里云DashScope API密钥 | - | 否 |
| `LOG_LEVEL` | 日志级别 | `INFO` | 否 |

### 日志配置

日志文件位于 `instance/logs/app.log`，支持：
- 自动轮转（10MB）
- 保留30天
- 结构化JSON格式
- 控制台彩色输出

## 🚀 部署

### Docker 部署

```bash
# 构建镜像
docker build -t llmation-worker .

# 运行容器
docker run -p 12000:12000 -e OPENAI_API_KEY=your-key llmation-worker
```

### 生产部署

```bash
# 使用 Gunicorn + Uvicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:12000
```

## 🤝 贡献

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 Apache 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

## 🔗 相关链接

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [LangChain 文档](https://langchain.com/)
- [OpenAI API](https://openai.com/api/)
- [阿里云DashScope](https://dashscope.aliyun.com/)

---

**Made with ❤️ by LLMation Team**
