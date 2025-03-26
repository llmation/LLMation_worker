from typing import Dict, Any, Optional

from app.utils import logger


def validate_llm_backend_request(data: Optional[Dict[str, Any]]) -> bool:
    """
    验证 LLMBackendRequest 输入数据格式

    Args:
        data: 输入数据字典

    Returns:
        验证结果，True表示有效，False表示无效
    """
    if not data or not isinstance(data, dict):
        logger.warning("验证失败：输入数据为空或非字典类型")
        return False

    # 检查是否包含enginePrompt字段
    if "enginePrompt" not in data or not isinstance(data["enginePrompt"], str):
        logger.warning("验证失败：缺少enginePrompt字段或类型不是字符串")
        return False

    # 检查conversation字段是否存在且为列表
    if "conversation" not in data or not isinstance(data["conversation"], list):
        logger.warning("验证失败：缺少conversation字段或类型不是列表")
        return False

    # 检查对话列表是否包含至少一个用户消息
    user_message_exists = False
    for message in data["conversation"]:
        if not isinstance(message, dict):
            logger.warning("验证失败：对话列表中包含非字典类型的消息")
            return False
        if message.get("type") == "user" and "content" in message:
            user_message_exists = True
            break

    if not user_message_exists:
        logger.warning("验证失败：对话列表中不包含有效的用户消息")
        return False

    logger.debug(
        "验证成功：LLMBackendRequest格式有效",
        conversation_length=len(data["conversation"]),
    )
    return True
