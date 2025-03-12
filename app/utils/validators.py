from typing import Dict, Any, Optional

def validate_chat_input(data: Optional[Dict[str, Any]]) -> bool:
    """
    验证聊天输入数据

    Args:
        data: 输入数据字典

    Returns:
        验证结果，True表示有效，False表示无效
    """
    if not data or not isinstance(data, dict):
        return False

    # 检查是否包含question字段
    if 'question' not in data or not isinstance(data['question'], str):
        return False

    # 检查question是否为空
    if not data['question'].strip():
        return False

    return True