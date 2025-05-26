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


@router.post("/workflow/completions")
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


@router.get("/rag-test", response_class=HTMLResponse)
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
