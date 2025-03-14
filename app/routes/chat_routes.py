from flask import Blueprint, request, jsonify, current_app, Response, stream_with_context
from app.models.chat_model import ChatModel
from app.utils.validators import validate_chat_input
# from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import json
from typing import Dict, Any

# 创建蓝图
chat_bp = Blueprint('chat', __name__, url_prefix='/api')

# 创建聊天模型实例
chat_model = None

@chat_bp.before_app_request
def initialize_chat_model():
    """在请求之前初始化聊天模型（如果尚未初始化）"""
    global chat_model
    if chat_model is None:
        chat_model = ChatModel(
            openai_api_key=current_app.config.get('OPENAI_API_KEY'),
            dashscope_api_key=current_app.config.get('DASHSCOPE_API_KEY')
        )

@chat_bp.route('/chat', methods=['POST'])
def chat():
    """处理聊天请求"""
    # 验证输入
    data: Dict[str, Any] = request.get_json()
    if not validate_chat_input(data):
        return jsonify({"error": "无效的输入格式"}), 400

    try:
        # 获取用户问题
        question = data.get('question', '')

        # 调用聊天模型
        response = chat_model.chat_with_user(question)

        return jsonify({"response": response})

    except Exception as e:
        current_app.logger.error(f"聊天请求处理错误: {str(e)}")
        return jsonify({"error": str(e)}), 500

@chat_bp.route('/chat/stream', methods=['POST'])
def chat_stream():
    """处理流式聊天请求"""
    # 验证输入
    data: Dict[str, Any] = request.get_json()
    if not validate_chat_input(data):
        return jsonify({"error": "无效的输入格式"}), 400

    try:
        # 获取用户问题
        question = data.get('question', '')

        def generate():
            try:
                for content in chat_model.stream_chat_with_user(question):
                    yield f"data: {json.dumps({'content': content})}\n\n"
                yield f"data: [DONE]\n\n"
            except Exception as e:
                current_app.logger.error(f"流式聊天生成错误: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield f"data: [DONE]\n\n"

        return Response(stream_with_context(generate()), mimetype='text/event-stream')

    except Exception as e:
        current_app.logger.error(f"流式聊天请求处理错误: {str(e)}")
        return jsonify({"error": str(e)}), 500

@chat_bp.route('/chat/clear', methods=['POST'])
def clear_history():
    """清空聊天历史"""
    try:
        chat_model.clear_history()
        return jsonify({"status": "success", "message": "聊天历史已清空"})
    except Exception as e:
        current_app.logger.error(f"清空聊天历史错误: {str(e)}")
        return jsonify({"error": str(e)}), 500

@chat_bp.route('/process', methods=['POST'])
def process_json():
    """处理LLMBackendRequest JSON输入并转换为自然语言+YAML的提示词"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "无效的JSON输入"}), 400

        # 验证LLMBackendRequest格式
        if "enginePrompt" not in data:
            return jsonify({"error": "缺少必要字段: enginePrompt"}), 400

        # 处理JSON输入
        prompt = chat_model.process_json_input(data)

        return jsonify({
            "status": "success",
            "prompt": prompt
        })

    except Exception as e:
        current_app.logger.error(f"处理JSON输入错误: {str(e)}")
        return jsonify({"error": str(e)}), 500