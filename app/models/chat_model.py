import os
from typing import Any, Dict, Generator, List, Optional, Tuple

import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.utils import logger


class ChatModel:
    """聊天模型类，使用 RAG 处理与大语言模型的交互"""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        dashscope_api_key: Optional[str] = None,
        docs_dir: str = "docs",
    ):
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

        logger.info("开始初始化ChatModel", docs_dir=docs_dir)

        # 初始化OpenAI聊天模型
        try:
            self.chat = ChatOpenAI(
                api_key=self.openai_api_key,
                model_name="gemini-2.0-flash",
                temperature=0.7,
                max_tokens=1000,
                base_url="https://api.bailili.top/v1",  # 自定义API端点
            )
            logger.info("成功初始化聊天模型", model="gemini-2.0-pro-exp")
        except Exception as e:
            logger.error("初始化聊天模型失败", error=str(e))
            raise e

        # 初始化嵌入模型和向量存储
        try:
            if not self.dashscope_api_key:
                logger.warning("DashScope API密钥未设置，将无法使用向量检索功能")
                self.embeddings = None
                self.vectorstore = None
            else:
                # 使用DashScope Embedding
                self.embeddings = DashScopeEmbeddings(
                    model="text-embedding-v1", dashscope_api_key=self.dashscope_api_key
                )

                # 使用In-Memory向量存储
                self.vectorstore = InMemoryVectorStore(self.embeddings)
                logger.info("成功初始化嵌入模型和向量存储", model="text-embedding-v1")
        except Exception as e:
            logger.error("初始化嵌入模型或向量存储失败", error=str(e))
            self.embeddings = None
            self.vectorstore = None

        # 初始化向量数据库
        if self.vectorstore:
            self._initialize_vectorstore(docs_dir)
        else:
            logger.warning("由于嵌入模型或向量存储初始化失败，跳过文档加载")

    def _initialize_vectorstore(self, docs_dir: str) -> None:
        """
        初始化向量数据库

        Args:
            docs_dir: 文档目录路径
        """
        try:
            # 加载文档
            documents = []
            load_errors = []

            if os.path.exists(docs_dir):
                logger.info("开始加载文档", docs_dir=docs_dir)
                for filename in os.listdir(docs_dir):
                    file_path = os.path.join(docs_dir, filename)
                    if filename.endswith(".txt"):
                        try:
                            loader = TextLoader(file_path, encoding="utf-8")
                            doc_chunks = loader.load()
                            documents.extend(doc_chunks)
                            logger.info("成功加载文档", filename=filename)
                        except Exception as e:
                            logger.error(
                                "加载文档失败", filename=filename, error=str(e)
                            )
                            load_errors.append(f"{filename}: {str(e)}")
                            continue

            if not documents:
                error_msg = "未找到文档或文档目录不存在" + (
                    f"\n错误详情: {load_errors}" if load_errors else ""
                )
                logger.warning(error_msg, errors=load_errors)
                return

            # 分割文档
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                texts = text_splitter.split_documents(documents)
                logger.info(
                    "成功分割文档",
                    chunks_count=len(texts),
                    chunk_size=1000,
                    chunk_overlap=200,
                )
            except Exception as e:
                logger.error("分割文档失败", error=str(e))
                raise e

            # 将文档添加到向量数据库
            try:
                if self.vectorstore and self.embeddings:
                    self.vectorstore.add_documents(texts)
                    logger.info("成功加载文档片段到向量数据库", chunks_count=len(texts))
                else:
                    logger.error("向量存储或嵌入模型未初始化")
            except Exception as e:
                logger.error("向量数据库添加文档失败", error=str(e))
                raise e

        except Exception as e:
            logger.error("初始化向量数据库失败", error=str(e))
            raise e

    def stream_chat_with_rag(
        self, engine_prompt: str, user_input: str, conversations: List[Dict[str, Any]]
    ) -> Generator[str, None, None]:
        """
        使用 RAG 进行流式聊天

        Args:
            engine_prompt: 系统提示词
            user_input: 用户输入
            conversations: 对话历史

        Yields:
            AI的回复内容块
        """
        try:
            logger.info(
                "开始处理聊天请求",
                prompt_length=len(engine_prompt),
                input_length=len(user_input),
                conversation_turns=len(conversations),
            )

            # 从向量数据库检索相关文档
            context = "无可用文档上下文"

            if self.vectorstore and self.embeddings:
                try:
                    logger.debug("开始检索相关文档", query=user_input[:50])
                    relevant_docs = self.vectorstore.similarity_search(user_input, k=3)
                    if relevant_docs:
                        context = "\n".join([doc.page_content for doc in relevant_docs])
                        logger.info("成功检索到相关文档片段", count=len(relevant_docs))
                    else:
                        logger.info("未找到相关文档")
                except Exception as e:
                    logger.error("文档检索失败", error=str(e))
            else:
                logger.warning("向量存储或嵌入模型未初始化，跳过文档检索")

            # 构建消息列表，包括系统消息和对话历史
            messages = []

            # 添加系统提示，包含检索到的文档上下文
            system_content = engine_prompt + f"\n\n文档上下文:\n{context}"
            messages.append(SystemMessage(content=system_content))

            # 添加对话历史
            for conv in conversations:
                if conv.get("type") == "user":
                    messages.append(HumanMessage(content=conv.get("content")))
                elif conv.get("type") == "assistant":
                    messages.append(AIMessage(content=conv.get("content")))

            logger.debug("准备调用AI模型", messages_count=len(messages))

            # 流式调用API
            response_stream = self.chat.stream(messages)

            # 返回流式响应
            for chunk in response_stream:
                if chunk.content:
                    yield chunk.content

            logger.info("聊天响应生成完成")

        except Exception as e:
            logger.error(
                "聊天处理过程中出错", error=str(e), error_type=type(e).__name__
            )
            yield f"处理请求时出错: {str(e)}"

    def process_json_input(
        self, json_input: Dict[str, Any]
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        处理LLMBackendRequest JSON输入并提取关键信息

        Args:
            json_input: LLMBackendRequest格式的JSON输入数据

        Returns:
            系统提示词、用户输入、对话历史的元组
        """
        logger.info("开始处理JSON输入", input_keys=list(json_input.keys()))

        # 提取引擎提示
        engine_prompt = json_input.get(
            "enginePrompt", "你是一个友好的AI助手，可以使用检索到的文档回答问题。"
        )

        # 提取活动文档和引用信息
        active_docs = json_input.get("active", {})
        references = json_input.get("reference", [])
        reference_nodes = json_input.get("referenceNodes", [])

        logger.debug(
            "提取的输入数据",
            active_docs_count=len(active_docs),
            references_count=len(references),
            reference_nodes_count=len(reference_nodes),
        )

        # 提取对话历史
        conversations = json_input.get("conversation", [])

        # 提取用户输入（最后一条用户消息）
        user_input = ""
        if conversations and conversations[-1].get("type") == "user":
            user_input = conversations[-1].get("content", "")
            logger.debug(
                "提取到用户输入",
                content_preview=user_input[:50]
                + ("..." if len(user_input) > 50 else ""),
            )

        # 增强系统提示词，添加上下文信息
        enhanced_prompt = engine_prompt + "\n\n"

        # 添加活动文档信息
        if active_docs:
            enhanced_prompt += "## 活动文档\n\n"
            for doc_id, doc in active_docs.items():
                doc_yaml = yaml.dump(doc, allow_unicode=True)
                enhanced_prompt += f"文档 ID: {doc_id}\n```yaml\n{doc_yaml}\n```\n\n"

        # 添加引用文档信息
        if references:
            enhanced_prompt += "## 引用文档\n\n"
            for i, ref in enumerate(references):
                ref_type = ref.get("type", "")
                ref_key = ref.get("key", "")
                ref_value = ref.get("value", "")
                enhanced_prompt += f"引用 {i + 1} (类型: {ref_type}):\n"
                if ref_type == "document":
                    enhanced_prompt += f"键: {ref_key}\n```yaml\n{ref_value}\n```\n\n"
                else:
                    enhanced_prompt += f"URL: {ref_key}\n内容: {ref_value}\n\n"

        # 添加引用节点信息
        if reference_nodes:
            enhanced_prompt += "## 引用节点\n\n"
            for i, node in enumerate(reference_nodes):
                node_yaml = yaml.dump(node, allow_unicode=True)
                enhanced_prompt += f"节点 {i + 1}:\n```yaml\n{node_yaml}\n```\n\n"

        logger.info(
            "JSON输入处理完成",
            prompt_length=len(enhanced_prompt),
            input_length=len(user_input),
        )
        return enhanced_prompt, user_input, conversations
