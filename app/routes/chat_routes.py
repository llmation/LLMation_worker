import json
import traceback

from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    render_template,
    request,
    stream_with_context,
)

from app.models.chat_model import ChatModel
from app.utils.validators import validate_llm_backend_request
from app.utils import logger

# 创建蓝图
chat_bp = Blueprint("chat", __name__, url_prefix="/api")

# 创建聊天模型实例
chat_model = None


@chat_bp.before_app_request
def initialize_chat_model():
    """在请求之前初始化聊天模型（如果尚未初始化）"""
    global chat_model
    if chat_model is None:
        logger.info("首次请求，初始化ChatModel实例")
        chat_model = ChatModel(
            openai_api_key=current_app.config.get("OPENAI_API_KEY"),
            dashscope_api_key=current_app.config.get("DASHSCOPE_API_KEY"),
        )


@chat_bp.route("/workflow/completions", methods=["POST"])
def process_json_stream():
    """处理LLMBackendRequest JSON输入并使用流式输出提供聊天内容"""
    try:
        logger.info(
            "接收到聊天请求",
            remote_addr=request.remote_addr,
            content_type=request.content_type,
            content_length=request.content_length,
        )

        data = request.json
        if not data:
            logger.warning("请求不包含有效的JSON数据")
            return jsonify({"error": "无效的JSON输入"}), 400

        # 验证LLMBackendRequest格式
        if not validate_llm_backend_request(data):
            logger.warning(
                "无效的LLMBackendRequest格式",
                keys_provided=list(data.keys() if isinstance(data, dict) else []),
            )
            return jsonify({"error": "无效的LLMBackendRequest格式"}), 400

        # 处理JSON输入转换为提示词
        logger.info("开始处理JSON输入")
        engine_prompt, user_input, conversations = chat_model.process_json_input(data)
        logger.info(
            "JSON输入处理完成",
            prompt_length=len(engine_prompt),
            input_length=len(user_input),
            conversation_turns=len(conversations),
        )

        def generate():
            try:
                logger.info("开始生成流式响应")
                for content in chat_model.stream_chat_with_rag(
                    engine_prompt, user_input, conversations
                ):
                    yield f"{json.dumps({'content': content})}\n\n"
                logger.info("流式响应生成完成")
                yield ""
            except Exception as e:
                error_traceback = traceback.format_exc()
                logger.error(
                    "流式JSON处理生成错误",
                    error=str(e),
                    error_type=type(e).__name__,
                    traceback=error_traceback,
                )
                yield f"{json.dumps({'error': str(e)})}\n\n"
                yield ""

        logger.info("返回流式响应")
        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(
            "流式JSON处理请求错误",
            error=str(e),
            error_type=type(e).__name__,
            traceback=error_traceback,
        )
        return jsonify({"error": str(e)}), 500


@chat_bp.route("/rag-test", methods=["GET"])
def rag_test_page():
    """返回RAG测试页面"""
    logger.info(
        "访问RAG测试页面",
        remote_addr=request.remote_addr,
        user_agent=request.user_agent.string,
    )
    return render_template("rag_test.html")
