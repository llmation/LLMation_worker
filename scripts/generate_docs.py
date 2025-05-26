#!/usr/bin/env python3
"""
LLMation Worker API 文档生成脚本

此脚本用于生成项目的各种文档格式：
- OpenAPI JSON/YAML 规范
- Markdown API 文档
- HTML 静态文档
- 开发者指南
"""

import json
import yaml
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app import create_app


def generate_openapi_spec():
    """生成 OpenAPI 规范文件"""
    print("🔄 生成 OpenAPI 规范...")
    
    app = create_app()
    openapi_schema = app.openapi()
    
    # 创建文档目录
    docs_dir = project_root / "docs" / "api"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成 JSON 格式
    json_path = docs_dir / "openapi.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2, ensure_ascii=False)
    print(f"✅ OpenAPI JSON: {json_path}")
    
    # 生成 YAML 格式
    yaml_path = docs_dir / "openapi.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(openapi_schema, f, default_flow_style=False, allow_unicode=True)
    print(f"✅ OpenAPI YAML: {yaml_path}")
    
    return openapi_schema


def generate_markdown_docs(openapi_schema):
    """生成 Markdown API 文档"""
    print("🔄 生成 Markdown 文档...")
    
    docs_dir = project_root / "docs" / "api"
    md_path = docs_dir / "API.md"
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"""# {openapi_schema['info']['title']} API 文档

{openapi_schema['info']['description']}

**版本**: {openapi_schema['info']['version']}

## 服务器

""")
        
        for server in openapi_schema.get('servers', []):
            f.write(f"- **{server['description']}**: `{server['url']}`\n")
        
        f.write("\n## API 端点\n\n")
        
        # 生成端点文档
        for path, methods in openapi_schema['paths'].items():
            f.write(f"### {path}\n\n")
            
            for method, details in methods.items():
                f.write(f"#### {method.upper()}\n\n")
                f.write(f"**摘要**: {details.get('summary', 'N/A')}\n\n")
                
                if 'description' in details:
                    f.write(f"**描述**:\n{details['description']}\n\n")
                
                if 'tags' in details:
                    f.write(f"**标签**: {', '.join(details['tags'])}\n\n")
                
                # 请求体
                if 'requestBody' in details:
                    f.write("**请求体**:\n")
                    content = details['requestBody']['content']
                    for content_type, schema in content.items():
                        f.write(f"- Content-Type: `{content_type}`\n")
                    f.write("\n")
                
                # 响应
                if 'responses' in details:
                    f.write("**响应**:\n")
                    for status, response in details['responses'].items():
                        f.write(f"- **{status}**: {response.get('description', 'N/A')}\n")
                    f.write("\n")
                
                f.write("---\n\n")
    
    print(f"✅ Markdown 文档: {md_path}")


def generate_html_docs():
    """生成 HTML 静态文档"""
    print("🔄 生成 HTML 文档...")
    
    docs_dir = project_root / "docs" / "api"
    html_path = docs_dir / "index.html"
    
    html_content = """<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLMation Worker API 文档</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .card {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }
        .endpoint {
            border-left: 4px solid #007bff;
            padding-left: 1rem;
            margin: 1rem 0;
        }
        .method {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-weight: bold;
            font-size: 0.8rem;
            margin-right: 0.5rem;
        }
        .method.post { background-color: #28a745; color: white; }
        .method.get { background-color: #007bff; color: white; }
        .method.put { background-color: #ffc107; color: black; }
        .method.delete { background-color: #dc3545; color: white; }
        .code {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
        }
        .nav {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .nav a {
            color: #007bff;
            text-decoration: none;
            margin-right: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            transition: background-color 0.3s;
        }
        .nav a:hover {
            background-color: #e9ecef;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 LLMation Worker API</h1>
        <p>智能对话与RAG服务 - 完整API文档</p>
    </div>

    <div class="nav">
        <a href="#overview">概览</a>
        <a href="#endpoints">API端点</a>
        <a href="#examples">示例</a>
        <a href="#testing">测试</a>
        <a href="http://localhost:12000/docs" target="_blank">交互式文档</a>
        <a href="http://localhost:12000/redoc" target="_blank">ReDoc文档</a>
    </div>

    <div class="card" id="overview">
        <h2>📋 概览</h2>
        <p>LLMation Worker 是一个基于 FastAPI 和 LangChain 构建的智能对话后端服务。</p>
        
        <h3>🚀 主要特性</h3>
        <ul>
            <li><strong>流式对话</strong>: 支持实时流式响应的智能对话</li>
            <li><strong>RAG增强</strong>: 基于文档检索的增强生成</li>
            <li><strong>多模型支持</strong>: 支持OpenAI和阿里云DashScope模型</li>
            <li><strong>文档处理</strong>: 智能文档解析和向量化存储</li>
            <li><strong>实时日志</strong>: 完整的请求追踪和日志记录</li>
        </ul>

        <h3>🔧 技术栈</h3>
        <ul>
            <li><strong>框架</strong>: FastAPI + Uvicorn</li>
            <li><strong>AI</strong>: LangChain + OpenAI/DashScope</li>
            <li><strong>向量存储</strong>: FAISS</li>
            <li><strong>日志</strong>: Loguru</li>
            <li><strong>验证</strong>: Pydantic</li>
        </ul>
    </div>

    <div class="card" id="endpoints">
        <h2>📋 API 端点</h2>
        
        <div class="endpoint">
            <h3><span class="method post">POST</span>/api/workflow/completions</h3>
            <p><strong>功能</strong>: 流式对话完成</p>
            <p><strong>描述</strong>: 处理LLMBackendRequest JSON输入并提供流式聊天响应</p>
            
            <h4>请求示例</h4>
            <div class="code">
{
    "enginePrompt": "你是一个友好的AI助手",
    "conversation": [
        {"type": "user", "content": "你好"}
    ]
}
            </div>
            
            <h4>响应示例</h4>
            <div class="code">
{"content": "你好！我是你的AI助手，很高兴为你服务。"}
            </div>
        </div>

        <div class="endpoint">
            <h3><span class="method get">GET</span>/api/rag-test</h3>
            <p><strong>功能</strong>: RAG测试页面</p>
            <p><strong>描述</strong>: 返回RAG功能测试的交互式网页界面</p>
        </div>

        <div class="endpoint">
            <h3><span class="method get">GET</span>/</h3>
            <p><strong>功能</strong>: 根路径</p>
            <p><strong>描述</strong>: 重定向到API文档页面</p>
        </div>
    </div>

    <div class="card" id="examples">
        <h2>💡 使用示例</h2>
        
        <h3>基本对话</h3>
        <div class="code">
curl -X POST "http://localhost:12000/api/workflow/completions" \\
  -H "Content-Type: application/json" \\
  -d '{
    "enginePrompt": "你是一个友好的AI助手",
    "conversation": [
      {"type": "user", "content": "你好"}
    ]
  }'
        </div>

        <h3>带文档的RAG对话</h3>
        <div class="code">
curl -X POST "http://localhost:12000/api/workflow/completions" \\
  -H "Content-Type: application/json" \\
  -d '{
    "enginePrompt": "你是一个专业的文档助手",
    "conversation": [
      {"type": "user", "content": "请总结文档内容"}
    ],
    "active": {
      "doc1": {
        "title": "产品说明",
        "content": "这是一个AI产品的详细说明..."
      }
    }
  }'
        </div>
    </div>

    <div class="card" id="testing">
        <h2>🧪 测试工具</h2>
        <p>我们提供了多种测试和文档工具：</p>
        <ul>
            <li><a href="http://localhost:12000/docs" target="_blank">Swagger UI</a> - 交互式API文档</li>
            <li><a href="http://localhost:12000/redoc" target="_blank">ReDoc</a> - 美观的API文档</li>
            <li><a href="http://localhost:12000/api/rag-test" target="_blank">RAG测试页面</a> - 可视化测试界面</li>
        </ul>
    </div>

    <div class="card">
        <h2>📞 联系信息</h2>
        <ul>
            <li><strong>项目地址</strong>: <a href="https://github.com/llmation/LLMation_worker" target="_blank">GitHub</a></li>
            <li><strong>协议</strong>: Apache 2.0</li>
            <li><strong>版本</strong>: 1.0.0</li>
        </ul>
    </div>
</body>
</html>"""
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ HTML 文档: {html_path}")


def generate_readme_docs():
    """更新 README.md 文档"""
    print("🔄 更新 README.md...")
    
    readme_path = project_root / "README.md"
    
    readme_content = """# LLMation Worker

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
"""
    
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"✅ README 文档: {readme_path}")


def main():
    """主函数"""
    print("🚀 开始生成 LLMation Worker API 文档...\n")
    
    try:
        # 生成 OpenAPI 规范
        openapi_schema = generate_openapi_spec()
        
        # 生成 Markdown 文档
        generate_markdown_docs(openapi_schema)
        
        # 生成 HTML 文档
        generate_html_docs()
        
        # 更新 README
        generate_readme_docs()
        
        print("\n✅ 所有文档生成完成！")
        print("\n📚 生成的文档：")
        print("  - docs/api/openapi.json    # OpenAPI JSON规范")
        print("  - docs/api/openapi.yaml    # OpenAPI YAML规范")
        print("  - docs/api/API.md          # Markdown API文档")
        print("  - docs/api/index.html      # HTML静态文档")
        print("  - README.md                # 项目说明文档")
        
        print("\n🌐 在线文档：")
        print("  - http://localhost:12000/docs      # Swagger UI")
        print("  - http://localhost:12000/redoc     # ReDoc")
        print("  - http://localhost:12000/api/rag-test  # RAG测试页面")
        
    except Exception as e:
        print(f"❌ 文档生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()