from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict, Any, Optional
import os
import yaml
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

class ChatModel:
    """聊天模型类，处理与大语言模型的交互"""

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 dashscope_api_key: Optional[str] = None,
                 docs_dir: str = "docs"):
        """
        初始化聊天模型

        Args:
            openai_api_key: OpenAI API密钥
            dashscope_api_key: DashScope API密钥
            docs_dir: 文档目录路径
        """
        # 使用传入的API密钥或环境变量
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.dashscope_api_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY")

        # 初始化OpenAI聊天模型
        self.chat = ChatOpenAI(
            api_key=self.openai_api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )

        # 使用DashScope Embedding
        self.embeddings = DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=self.dashscope_api_key
        )

        # 使用In-Memory向量存储
        self.vectorstore = InMemoryVectorStore(self.embeddings)

        # 初始化向量数据库
        self._initialize_vectorstore(docs_dir)

        # 设置RAG提示模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个友好的AI助手。使用以下上下文信息来回答用户的问题。如果问题无法从上下文中得到答案，请说明并基于你的知识回答。
            请按照以下格式返回答案:
            ```json
             - "answer": "对问题的直接回答",
             - "reason": "解释为什么给出这个答案,包括使用了哪些上下文信息或知识"
            ```
            上下文:
            {context}"""),
            ("human", "{question}")
        ])

        # 初始化对话历史
        self.messages = [
            SystemMessage(content="你是一个友好的AI助手,可以帮助用户解答各种问题。")
        ]

    def _initialize_vectorstore(self, docs_dir: str) -> None:
        """
        初始化向量数据库

        Args:
            docs_dir: 文档目录路径
        """
        try:
            # 加载文档
            documents = []
            if os.path.exists(docs_dir):
                for filename in os.listdir(docs_dir):
                    if filename.endswith('.txt'):
                        file_path = os.path.join(docs_dir, filename)
                        loader = TextLoader(file_path)
                        documents.extend(loader.load())

            if not documents:
                print("警告: 未找到文档或文档目录不存在")
                return

            # 分割文档
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)

            # 将文档添加到向量数据库
            self.vectorstore.add_documents(texts)
            print(f"成功加载 {len(texts)} 个文档片段到向量数据库")

        except Exception as e:
            print(f"初始化向量数据库时出错: {str(e)}")
            raise e

    def chat_with_user(self, user_input: str) -> str:
        """
        与用户进行对话，支持RAG

        Args:
            user_input: 用户输入的问题

        Returns:
            AI的回复
        """
        try:
            # 从向量数据库检索相关文档
            relevant_docs = []
            if self.vectorstore:
                relevant_docs = self.vectorstore.similarity_search(user_input, k=3)
                context = "\n".join([doc.page_content for doc in relevant_docs])
            else:
                context = "无可用文档上下文"

            # 构建带有上下文的消息
            prompt_value = self.prompt.format(context=context, question=user_input)
            messages = self.messages + [HumanMessage(content=prompt_value)]

            # 使用流式输出模式获取回复
            response_stream = self.chat.stream(messages)

            full_response = ""
            for chunk in response_stream:
                if chunk.content:
                    full_response += chunk.content

            # 将用户输入和AI回复添加到对话历史
            self.messages.append(HumanMessage(content=user_input))
            self.messages.append(AIMessage(content=full_response))

            return full_response

        except Exception as e:
            print(f"聊天过程中出错: {str(e)}")
            raise e

    def clear_history(self) -> None:
        """清空对话历史"""
        self.messages = [
            SystemMessage(content="你是一个友好的AI助手,可以帮助用户解答各种问题。")
        ]

    def create_chain(self):
        """
        创建可供Flask使用的chain

        Returns:
            可运行的LangChain链
        """
        def format_messages(input_dict: Dict[str, Any]):
            """格式化消息"""
            question = input_dict.get("question", "")

            # 检索相关文档
            relevant_docs = []
            if self.vectorstore:
                relevant_docs = self.vectorstore.similarity_search(question, k=3)

            # 将Document对象序列化为YAML
            context_docs = []
            for doc in relevant_docs:
                doc_yaml = yaml.dump({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })
                context_docs.append(doc_yaml)

            context = "\n".join(context_docs) if context_docs else "无可用文档上下文"

            # 使用prompt模板创建消息
            return self.prompt.format_messages(
                context=context,
                question=question
            )

        # 构建支持流式输出的chain
        chain = (
            RunnablePassthrough()
            | format_messages
            | self.chat
        )

        return chain

    def process_json_input(self, json_input: Dict[str, Any]) -> str:
        """
        处理LLMBackendRequest JSON输入并转换为自然语言+YAML的提示词

        Args:
            json_input: LLMBackendRequest格式的JSON输入数据

        Returns:
            处理后的提示词
        """
        # 提取引擎提示
        engine_prompt = json_input.get("enginePrompt", "")

        # 提取活动文档
        active_docs = json_input.get("active", {})

        # 提取引用文档
        references = json_input.get("reference", [])

        # 提取引用节点
        reference_nodes = json_input.get("referenceNodes", [])

        # 提取对话历史
        conversations = json_input.get("conversation", [])

        # 构建提示词
        prompt = f"""引擎提示: {engine_prompt}\n\n"""

        # 添加活动文档
        if active_docs:
            prompt += "## 活动文档\n\n"
            for doc_id, doc in active_docs.items():
                doc_yaml = yaml.dump(doc)
                prompt += f"文档 ID: {doc_id}\n```yaml\n{doc_yaml}\n```\n\n"

        # 添加引用文档
        if references:
            prompt += "## 引用文档\n\n"
            for i, ref in enumerate(references):
                ref_type = ref.get("type", "")
                ref_key = ref.get("key", "")
                ref_value = ref.get("value", "")

                prompt += f"引用 {i+1} (类型: {ref_type}):\n"
                if ref_type == "document":
                    prompt += f"键: {ref_key}\n```yaml\n{ref_value}\n```\n\n"
                else:
                    prompt += f"URL: {ref_key}\n内容: {ref_value}\n\n"

        # 添加引用节点
        if reference_nodes:
            prompt += "## 引用节点\n\n"
            for i, node in enumerate(reference_nodes):
                node_yaml = yaml.dump(node)
                prompt += f"节点 {i+1}:\n```yaml\n{node_yaml}\n```\n\n"

        # 添加对话历史
        if conversations:
            prompt += "## 对话历史\n\n"
            for i, conv in enumerate(conversations):
                conv_type = conv.get("type", "")
                conv_content = conv.get("content", "")
                conv_name = conv.get("name", "")

                if conv_name:
                    prompt += f"{conv_type} ({conv_name}): {conv_content}\n\n"
                else:
                    prompt += f"{conv_type}: {conv_content}\n\n"

        return prompt