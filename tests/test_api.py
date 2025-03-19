import requests
import json
import time

BASE_URL = "http://localhost:5000/api"

def test_chat():
    print("\n=== 测试普通聊天 ===")
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"question": "什么是人工智能？"}
    )
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")

def test_chat_stream():
    print("\n=== 测试流式聊天 ===")
    response = requests.post(
        f"{BASE_URL}/chat/stream",
        json={"question": "请简要介绍机器学习"},
        stream=True
    )
    print(f"状态码: {response.status_code}")

    full_response = ""
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith("data: ") and line != "data: [DONE]":
                data = json.loads(line[6:])
                content = data.get("content", "")
                full_response += content
                print(content, end="", flush=True)
            elif line == "data: [DONE]":
                print("\n[完成]")

    print("\n完整响应:", full_response)

def test_clear_history():
    print("\n=== 测试清空历史 ===")
    response = requests.post(f"{BASE_URL}/chat/clear")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")

def test_conversation_flow():
    print("\n=== 测试对话流程 ===")

    # 清空历史
    requests.post(f"{BASE_URL}/chat/clear")

    # 第一个问题
    print("问题1: 什么是人工智能？")
    response1 = requests.post(
        f"{BASE_URL}/chat",
        json={"question": "什么是人工智能？"}
    )
    print(f"回答1: {response1.json().get('response')}")

    # 第二个问题（基于上下文）
    time.sleep(1)
    print("\n问题2: 它有哪些应用领域？")
    response2 = requests.post(
        f"{BASE_URL}/chat",
        json={"question": "它有哪些应用领域？"}
    )
    print(f"回答2: {response2.json().get('response')}")

if __name__ == "__main__":
    test_chat()
    test_clear_history()
    test_conversation_flow()
    # 注释掉流式聊天测试，因为它可能会阻塞控制台输出
    # test_chat_stream()