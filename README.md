# LLMation Worker

基于Flask和LangChain的大语言模型处理Worker，用于处理LLMation系统的对话请求。
TODOlist 在 TODO.md

## 项目结构

```text
LLMation_worker/
├── app/                    # 应用主目录
│   ├── models/             # 数据模型
│   ├── routes/             # API路由
│   ├── services/           # 服务层
│   ├── utils/              # 工具函数
│   └── __init__.py         # 应用初始化
├── docs/                   # 文档目录
├── tests/                  # 测试目录
├── venv/                   # 虚拟环境
├── app.py                  # 应用入口
├── requirements.txt        # 依赖列表
└── README.md               # 项目说明
```

## 功能特性

- 基于OPENAI_API的大语言模型对话
- 支持RAG（检索增强生成）
- 支持流式输出
- 支持JSON输入转换为自然语言+YAML提示词
- 文档序列化为YAML格式

## 安装与运行

### 环境准备

1. 确保已安装Python 3.8+
2. 创建并激活虚拟环境：`.\.venv\Scripts\activate source venv/bin/activate`
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

## API接口

### 1. 聊天接口

- **URL**: `/api/chat`
- **方法**: POST
- **请求体**:

  ```json
  {
    "question": "你好，请介绍一下LLMation"
  }
  ```

- **响应**:

  ```json
  {
    "response": "LLMation是一个基于大模型的自动化流程引擎..."
  }
  ```

### 2. 流式聊天接口

- **URL**: `/api/chat/stream`
- **方法**: POST
- **请求体**:

  ```json
  {
    "question": "你好，请介绍一下LLMation"
  }
  ```

- **响应**: Server-Sent Events (SSE)

### 3. 清空聊天历史

- **URL**: `/api/chat/clear`
- **方法**: POST
- **响应**:

  ```json
  {
    "status": "success",
    "message": "聊天历史已清空"
  }
  ```

### 4. JSON处理接口

- **URL**: `/api/process`
- **方法**: POST
- **请求体**:

  ```json
  {
    "enginePrompt": "现在你在编写一个应用，你需要...",
    "active": {
      "doc1": {
        "id": "doc1",
        "name": "示例文档",
        "description": "这是一个示例",
        "engine": "dolphin",
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
        "value": "参考文档1内容"
      },
      {
        "type": "url",
        "key": "https://example.com",
        "value": "参考网址解析完的内容"
      }
    ],
    "referenceNodes": [
      {
        "id": "ref1",
        "name": "参考节点",
        "description": "参考节点描述",
        "engine": "dolphin",
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

- **响应**:

  ```json
  {
    "status": "success",
    "prompt": "引擎提示: 现在你在编写一个应用，你需要...\n\n## 活动文档\n\n文档 ID: doc1\n```yaml\nid: doc1\nname: 示例文档\ndescription: 这是一个示例\nengine: dolphin\neffects: []\ninputs: {}\nnodes: []\noutputs: {}\n```\n\n..."
  }
  ```

## 测试

运行测试：

```bash
pytest
```
