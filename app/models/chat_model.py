from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict, Any, Optional, Tuple
import os
import yaml
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
# from langchain_core.vectorstores import VectorStore
# from langchain_core.documents import Document
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
        try:
            self.chat = ChatOpenAI(
                api_key=self.openai_api_key,
                model_name="gemini-2.0-pro-exp",
                temperature=0.7,
                max_tokens=1000,
                base_url="https://api.bailili.top/v1"  # 自定义API端点
            )
            print("成功初始化聊天模型")
        except Exception as e:
            print(f"初始化聊天模型时出错: {str(e)}")
            raise e

        # 初始化嵌入模型和向量存储
        try:
            if not self.dashscope_api_key:
                print("警告: DashScope API密钥未设置，将无法使用向量检索功能")
                self.embeddings = None
                self.vectorstore = None
            else:
                # 使用DashScope Embedding
                self.embeddings = DashScopeEmbeddings(
                    model="text-embedding-v1",
                    dashscope_api_key=self.dashscope_api_key
                )

                # 使用In-Memory向量存储
                self.vectorstore = InMemoryVectorStore(self.embeddings)
                print("成功初始化嵌入模型和向量存储")
        except Exception as e:
            print(f"初始化嵌入模型或向量存储时出错: {str(e)}")
            self.embeddings = None
            self.vectorstore = None

        # 初始化向量数据库
        if self.vectorstore:
            self._initialize_vectorstore(docs_dir)
        else:
            print("由于嵌入模型或向量存储初始化失败，跳过文档加载")

        # 设置RAG提示模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个友好的AI助手。你必须使用以下上下文信息来回答用户的问题。如果问题无法从上下文中得到答案，请明确说明"无法从文档中找到相关信息"。

            请回答问题，并将结果组织成以下格式：

            回答：对问题的直接回答
            理由：解释为什么给出这个答案,包括使用了哪些上下文信息
            来源：信息来源（文档名称或页码）

            上下文信息:
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
            load_errors = []

            if os.path.exists(docs_dir):
                for filename in os.listdir(docs_dir):
                    file_path = os.path.join(docs_dir, filename)
                    if filename.endswith('.txt'):
                        try:
                            loader = TextLoader(file_path, encoding='utf-8')
                            doc_chunks = loader.load()
                            documents.extend(doc_chunks)
                            print(f"成功加载文档: {filename}")
                        except Exception as e:
                            print(f"加载文档 {filename} 时出错: {str(e)}")
                            continue

            if not documents:
                error_msg = "警告: 未找到文档或文档目录不存在" + (f"\n错误详情: {load_errors}" if load_errors else "")
                print(error_msg)
                return

            # 分割文档
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                texts = text_splitter.split_documents(documents)
                print(f"成功分割文档，共{len(texts)}个片段")
            except Exception as e:
                print(f"分割文档时出错: {str(e)}")
                raise e

            # 将文档添加到向量数据库
            try:
                if self.vectorstore and self.embeddings:
                    self.vectorstore.add_documents(texts)
                    print(f"成功加载 {len(texts)} 个文档片段到向量数据库")
                else:
                    print("错误: 向量存储或嵌入模型未初始化")
            except Exception as e:
                print(f"向量数据库添加文档时出错: {str(e)}")
                raise e

        except Exception as e:
            print(f"初始化向量数据库时出错: {str(e)}")
            raise e

    def stream_chat_with_user(self, engine_prompt: str, user_input: str, conversations: List[Dict[str, Any]]):
        """
        与用户进行流式对话，支持RAG

        Args:
            user_input: 用户输入的问题

        Yields:
            AI的回复内容块
        """
        try:
            # 从向量数据库检索相关文档
            context = "无可用文档上下文"

            if self.vectorstore and self.embeddings:
                try:
                    relevant_docs = self.vectorstore.similarity_search(user_input, k=3)
                    if relevant_docs:
                        context = "\n".join([doc.page_content for doc in relevant_docs])
                        print(f"成功检索到{len(relevant_docs)}个相关文档片段")
                    else:
                        print("未找到相关文档")
                except Exception as e:
                    print(f"文档检索出错: {str(e)}")
            else:
                print("向量存储或嵌入模型未初始化，跳过文档检索")

            print(f"用户问题: {user_input}")

            # 使用流式输出模式获取回复
            try:
                # 构建带有上下文的消息
                system_prompt = """
                你是一个友好的AI助手。你必须使用以下上下文信息来回答用户的问题。如果问题无法从上下文中得到答案，请明确说明"无法从文档中找到相关信息"。

                请回答问题，并将结果组织成以下格式：

                回答：对问题的直接回答
                理由：解释为什么给出这个答案,包括使用了哪些上下文信息
                来源：信息来源（文档名称或页码）

                上下文信息:
                {context}"""

                messages=[]
                messages.append(SystemMessage(content=engine_prompt))
                print(f"添加系统消息: {engine_prompt}")
                for i, conv in enumerate(conversations):
                    if conv.get("type") == "user":
                        messages.append(HumanMessage(content=conv.get("content")))
                        print(f"添加用户消息: {conv.get('content')}")
                    elif conv.get("type") == "assistant":
                        messages.append(AIMessage(content=conv.get("content")))
                        print(f"添加助手消息: {conv.get('content')}")

                # messages = []
                # messages.append(SystemMessage(content=system_prompt.format(context=context)))
                # messages.append(HumanMessage(content=user_input))

                # 流式调用API
                response_stream = self.chat.stream(messages)

                full_response = ""
                for chunk in response_stream:
                    if chunk.content:
                        content = chunk.content
                        full_response += content
                        yield content

                # # 将用户输入和AI回复添加到对话历史
                # self.messages.append(HumanMessage(content=user_input))
                # self.messages.append(AIMessage(content=full_response))
                # print("AI回复完成，已更新对话历史")

            except Exception as api_error:
                print(f"API调用错误: {str(api_error)}")
                error_message = f"与AI模型通信时出错: {str(api_error)}"
                yield error_message
                # 不在这里抛出异常，让函数继续执行到结束

        except Exception as e:
            print(f"聊天过程中出错: {str(e)}")
            yield f"处理请求时出错: {str(e)}"

    def chat_with_user(self, user_input: str) -> str:
        """
        与用户进行对话，支持RAG

        Args:
            user_input: 用户输入的问题

        Returns:
            AI的完整回复
        """
        full_response = ""
        for content in self.stream_chat_with_user(user_input):
            full_response += content
        return full_response

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

            # 复用文档检索逻辑
            context = "无可用文档上下文"

            if self.vectorstore and self.embeddings:
                try:
                    relevant_docs = self.vectorstore.similarity_search(question, k=3)
                    if relevant_docs:
                        context = "\n".join([doc.page_content for doc in relevant_docs])
                        print(f"chain中成功检索到{len(relevant_docs)}个相关文档片段")
                    else:
                        print("chain中未找到相关文档")
                except Exception as e:
                    print(f"chain中文档检索出错: {str(e)}")
            else:
                print("chain中向量存储或嵌入模型未初始化，跳过文档检索")

            # 构建系统提示词
            system_prompt = """你是一个友好的AI助手。你必须使用以下上下文信息来回答用户的问题。如果问题无法从上下文中得到答案，请明确说明"无法从文档中找到相关信息"。

            请回答问题，并将结果组织成以下格式：

            回答：对问题的直接回答
            理由：解释为什么给出这个答案,包括使用了哪些上下文信息
            来源：信息来源（文档名称或页码）

            上下文信息:
            {context}"""

            # 构建消息列表
            messages = []
            messages.append(SystemMessage(content=system_prompt.format(context=context)))
            messages.append(HumanMessage(content=question))

            return messages

        # 构建支持流式输出的chain
        chain = (
            RunnablePassthrough()
            | format_messages
            | self.chat
        )

        return chain

    def process_json_input(self, json_input: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
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

        # 提取用户输入
        user_input = ""
        if conversations and len(conversations) > 0:
            # 获取对话历史中最后一个元素的内容作为用户输入
            user_input = conversations[-1].get("content", "")

        # 构建提示词
        prompt = ""

        # 添加活动文档
        if active_docs:
            prompt += "## 活动文档\n\n"
            for doc_id, doc in active_docs.items():
                # 使用allow_unicode=True确保中文正确显示
                doc_yaml = yaml.dump(doc, allow_unicode=True)
                prompt += f"文档 ID: {doc_id}\n```yaml\n{doc_yaml}\n```\n\n"

        # 添加当前文档
        current_doc = json_input.get("current", {})
        if current_doc:
            prompt += "## 当前文档\n\n"
            # 使用allow_unicode=True确保中文正确显示
            current_doc_yaml = yaml.dump(current_doc, allow_unicode=True)
            prompt += f"```yaml\n{current_doc_yaml}\n```\n\n"

        # 添加引用文档
        if references:
            prompt += "## 引用文档\n\n"
            for i, ref in enumerate(references):
                ref_type = ref.get("type", "")
                ref_key = ref.get("key", "")
                ref_value = ref.get("value", "")

                prompt += f"引用 {i+1} (类型: {ref_type}):\n"
                if ref_type == "document":
                    # 使用allow_unicode=True确保中文正确显示
                    prompt += f"键: {ref_key}\n```yaml\n{ref_value}\n```\n\n"
                else:
                    prompt += f"URL: {ref_key}\n内容: {ref_value}\n\n"

        # 添加引用节点
        if reference_nodes:
            prompt += "## 引用节点\n\n"
            for i, node in enumerate(reference_nodes):
                # 使用allow_unicode=True确保中文正确显示
                node_yaml = yaml.dump(node, allow_unicode=True)
                prompt += f"节点 {i+1}:\n```yaml\n{node_yaml}\n```\n\n"

        # 添加模板
        # 添加模板
        if user_input:
            prompt += f"用户输入: {user_input}\n\n"

        # 更改应用到最后一个conv上
        if conversations and len(conversations) > 0:
            conversations[-1]["content"] = prompt

        return engine_prompt, user_input, conversations