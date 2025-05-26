import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.utils.logger import init_app_logger, LoggerConfig, LogLevel, LogFormat
from app.models.chat_model import ChatModel

# 加载环境变量
load_dotenv()

# 全局变量存储聊天模型实例
chat_model_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global chat_model_instance
    
    # 启动时初始化
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")
    
    chat_model_instance = ChatModel(
        openai_api_key=openai_api_key,
        dashscope_api_key=dashscope_api_key,
    )
    
    yield
    
    # 关闭时清理
    chat_model_instance = None


def create_app(test_config=None):
    """创建并配置FastAPI应用"""
    
    # 确保实例文件夹存在
    instance_path = os.path.join(os.getcwd(), "instance")
    try:
        os.makedirs(instance_path, exist_ok=True)
    except OSError:
        pass

    # 确保日志目录存在
    log_dir = os.path.join(instance_path, "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError:
        pass

    # 配置日志系统
    logger_config = LoggerConfig(
        name="LLMation_worker",
        level=LogLevel.DEBUG,
        format_type=LogFormat.HYBRID,
        file_path=os.path.join(log_dir, "app.log"),
        rotation="10 MB",
        retention="30 days",
    )
    init_app_logger(None, logger_config)

    # 创建FastAPI应用
    app = FastAPI(
        title="LLMation Worker API",
        description="""
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
        """,
        version="1.0.0",
        lifespan=lifespan,
        contact={
            "name": "LLMation Team",
            "url": "https://github.com/llmation/LLMation_worker",
        },
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        },
        servers=[
            {
                "url": "http://localhost:12000",
                "description": "开发环境"
            },
            {
                "url": "https://work-1-npxugdcnhgrookmw.prod-runtime.all-hands.dev",
                "description": "生产环境"
            }
        ]
    )

    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    from app.routes.chat_routes import router as chat_router
    app.include_router(chat_router, prefix="/api")

    # 静态文件服务
    if os.path.exists("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # 文档静态文件服务
    docs_path = os.path.join(os.getcwd(), "docs")
    if os.path.exists(docs_path):
        app.mount("/docs-static", StaticFiles(directory=docs_path), name="docs-static")

    # 根路径重定向
    @app.get("/")
    async def root():
        return RedirectResponse(url="/api/rag-test")

    return app


def get_chat_model() -> ChatModel:
    """获取聊天模型实例"""
    global chat_model_instance
    if chat_model_instance is None:
        raise RuntimeError("Chat model not initialized")
    return chat_model_instance
