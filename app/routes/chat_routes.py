import json
import traceback
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

from app import get_chat_model
from app.utils.validators import validate_llm_backend_request
from app.utils import logger

# 创建路由器
router = APIRouter()


class LLMBackendRequest(BaseModel):
    """LLM后端请求模型"""
    enginePrompt: str
    conversation: list


@router.post(
    "/workflow/completions",
    summary="流式对话完成",
    description="""
处理LLMBackendRequest JSON输入并提供流式聊天响应。

**功能特性:**
- 支持实时流式响应
- RAG文档检索增强  
- 多轮对话上下文
- 智能错误处理

**请求示例:** 包含enginePrompt、conversation、active文档和reference参考
**响应格式:** 流式JSON，每行包含content字段或error字段
    """,
    response_description="流式JSON响应，包含生成的文本内容",
    tags=["对话"]
)
async def process_json_stream(request: Request, data: Dict[str, Any]):
    """处理LLMBackendRequest JSON输入并使用流式输出提供聊天内容"""
    try:
        client_host = request.client.host if request.client else "unknown"
        content_type = request.headers.get("content-type", "unknown")
        content_length = request.headers.get("content-length", "unknown")
        
        logger.info(
            "接收到聊天请求",
            remote_addr=client_host,
            content_type=content_type,
            content_length=content_length,
        )

        if not data:
            logger.warning("请求不包含有效的JSON数据")
            raise HTTPException(status_code=400, detail="无效的JSON输入")

        # 验证LLMBackendRequest格式
        if not validate_llm_backend_request(data):
            logger.warning(
                "无效的LLMBackendRequest格式",
                keys_provided=list(data.keys() if isinstance(data, dict) else []),
            )
            raise HTTPException(status_code=400, detail="无效的LLMBackendRequest格式")

        # 获取聊天模型实例
        chat_model = get_chat_model()

        # 处理JSON输入转换为提示词
        logger.info("开始处理JSON输入")
        engine_prompt, user_input, conversations = chat_model.process_json_input(data)
        logger.info(
            "JSON输入处理完成",
            prompt_length=len(engine_prompt),
            input_length=len(user_input),
            conversation_turns=len(conversations),
        )

        async def generate():
            try:
                logger.info("开始生成流式响应")
                for content in chat_model.stream_chat_with_rag(
                    engine_prompt, user_input, conversations
                ):
                    yield f"{json.dumps({'content': content})}\n\n"
                logger.info("流式响应生成完成")
            except Exception as e:
                error_traceback = traceback.format_exc()
                logger.error(
                    "流式JSON处理生成错误",
                    error=str(e),
                    error_type=type(e).__name__,
                    traceback=error_traceback,
                )
                yield f"{json.dumps({'error': str(e)})}\n\n"

        logger.info("返回流式响应")
        return StreamingResponse(generate(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(
            "流式JSON处理请求错误",
            error=str(e),
            error_type=type(e).__name__,
            traceback=error_traceback,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/rag-test",
    summary="RAG测试页面",
    description="""
返回RAG功能测试的交互式网页界面。

**功能特性:**
- 可视化JSON输入编辑器
- 实时流式响应显示  
- 多种示例模板
- 错误信息展示

**使用方法:** 编辑JSON请求 → 点击开始流式输出 → 查看实时响应
**示例模板:** 基本示例、对话示例、文档示例
    """,
    response_class=HTMLResponse,
    tags=["测试界面"]
)
async def rag_test_page(request: Request):
    """返回RAG测试页面"""
    client_host = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    logger.info(
        "访问RAG测试页面",
        remote_addr=client_host,
        user_agent=user_agent,
    )
    
    # 读取HTML模板文件
    try:
        with open("app/templates/rag_test.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="RAG测试页面未找到")
