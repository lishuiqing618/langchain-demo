import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from operator import itemgetter

from datetime import datetime

# ---- 引入 Day 3 的文件存储逻辑 ----
# 这里为了代码简洁，我们把之前的 FileChatMessageHistory 类直接拿过来用
# 在实际开发中，应该把它放在单独的 utils.py 文件里导入
import json
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

HISTORY_FILE = "chat_history_rag.json"

class FileChatMessageHistory(InMemoryChatMessageHistory):
    # ...(这里省略类的具体实现代码，直接服用 Day 3 的代码即可)
    # 确保包含 _load_messages, add_message, _save_to_file 等方法
    # 为了演示流程，我在下面快速写了一个简化版，如果跑不通请用 Day 3 那个完整的
    def __init__(self, session_id: str, file_path: str):
        # 先调用父类初始化
        super().__init__()

        # 使用对象的私有属性存储额外信息，避免与 Pydantic 模型冲突
        object.__setattr__(self, '_session_id', session_id)
        
        # 确保 file_path 不为 None
        if file_path is None:
            raise ValueError("file_path 不能为 None")
        
        object.__setattr__(self, '_file_path', file_path)
        
        # 记录当前会话的创建时间
        object.__setattr__(self, '_session_created_at', datetime.now())

        # 加载历史消息
        self._load_messages() 

    @property
    def session_id(self):
        return getattr(self, '_session_id', None)

    @property
    def file_path(self):
        return getattr(self, '_file_path', None)
            
    def _load_messages(self):
        """  从JSON 文件加载记录并转换为对象 """
        # 检查 file_path 是否存在
        if not hasattr(self, '_file_path') or self.file_path is None:
            print("错误: file_path 未设置")
            return
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 获取当前 session 的记录
            session_data = data.get(self.session_id, {})
            raw_messages = session_data.get('messages', [])

            if not raw_messages: 
                return

            # 关键步骤：把字典列表转换成 LangChain 的 Message 对象
            # messages_from_dict 是 LangChain 提供的工具函数
            loaded_messages = messages_from_dict(raw_messages)

            # 将加载的消息添加到内存历史中
            for msg in loaded_messages:
                super().add_message(msg)

        except FileNotFoundError:
            # 如果文件不存在，创建空文件
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({}, f)
        except json.JSONDecodeError:
            # 如果文件内容不是有效的JSON，创建空文件
            print(f"警告: {self.file_path} 文件内容无效，已重新初始化")
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({}, f)
        except Exception as e:
            # ✅ 这里打印详细错误，方便排查
            import traceback
            print(f"加载历史记录失败: {e}")
            print(traceback.format_exc()) # 打印错误堆栈

    def add_message(self, message: BaseMessage):
        """ 添加一条消息（内存） """
        super().add_message(message)
        # 每次添加后保存（实际生产中可能需要批量保存优化） 
        self._save_to_file() 
    
    def clear(self):
        """ 清空所有消息 """
        super().clear()
        self._save_to_file()

    def get_session_info(self):
        """ 获取当前会话数据 """
        
        # 检查 file_path 是否存在
        if not hasattr(self, '_file_path') or self.file_path is None:
            print("错误: file_path 未设置")
            return None
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            session_data = data.get(self.session_id, {})
            return {
                "session_id": self.session_id,
                "message_count": len(self.messages),
                "created_at": session_data.get('meta', {}).get('created_at'),
                "updated_at": session_data.get('meta', {}).get('updated_at')
            }
        except:
            return None
        
    def _save_to_file(self):
        """ 把当前的 Message 对象转换成字典并存入 JSON 文件 """
        
        # 检查 file_path 是否存在
        if not hasattr(self, '_file_path') or self.file_path is None:
            print("错误: file_path 未设置")
            return
        
        # 1. 读取现有所有数据（避免覆盖其他 user 的记录）
        all_data = {}
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或者内容无效，创建空的数据结构
            pass
        except Exception as e:
            print(f"读取历史记录失败: {e}")
            all_data = {}

        # 2. ✅ 关键修复：使用 messages_to_dict 转换格式
        # 这会将消息转换成 LangChain 标准的嵌套格式 {"type": "human", "data": {...}}
        # 这样 _load_messages 里的 messages_from_dict 才能读懂
        base_dicts = messages_to_dict(self.messages)

        message_dicts = []
        for item in base_dicts:
            # 在这里添加自定义的时间戳字段
            # 注意：timestamp 加在 data 里面
            item["data"]["timestamp"] = datetime.now().isoformat()
            message_dicts.append(item)

        # 3. 准备会话元数据
        session_meta = {
            "session_id": self.session_id,
            "created_at": getattr(self, '_session_created_at', datetime.now()).isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_count": len(message_dicts)
        }

        # 4. 构建当前会话的完整数据结构
        current_session_dict = {
            "meta": session_meta,
            "messages": message_dicts
        }

        # 5. 更新当前 session 的数据
        all_data[self.session_id] = current_session_dict

        # 6. 保存到文件
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4) # ident=4 美化格式

def get_session_history(session_id: str) -> FileChatMessageHistory:
    return FileChatMessageHistory(session_id=session_id, file_path=HISTORY_FILE)

# ================== 主程序开始 ==================

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("DASHSCOPE_BASE_URL")

llm = ChatOpenAI(base_url=base_url, model="qwen-plus")

# ==========================================
# Day 4：准备 RAG 知识库（模拟一份员工手册）
# ==========================================
# raw_text = """
#     公司休假制度：    
#     1. 年假：员工入职满一年后，享有5天年假；满十年享有10天。
#     2. 病假：员工每月享有1天带薪病假，需提供医院证明。
#     3. 事假：需提前三天申请，扣除当日全额工资。
#     4. 远程办公：每周五允许全员居家办公，但需在钉钉上打卡。
#     请注意：所有请假申请必须经过直属经理批准。
# """

# 指定 PDF 文档路径
pdf_path = "docs/yuwen.pdf"

# 加载 PDF 文档
print("正在加载 PDF...")
loader = PyPDFLoader(pdf_path)
# loader.load() 会把 PDF 的每一页变成一个 Document 对象
docs = loader.load()
print(f"PDF 加载完成，共 {len(docs)} 页。")

# 文档切分
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
# splits = text_splitter.split_text(raw_text)
# 初始化文本切分器
# 这里的 chunk_size 可以设置大一点，比如 500 或者 1000，因为 PDF 的内容通常比较多
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # 每块 500 字
    chunk_overlap=50,    # 块重叠 50 字，防止语义断裂
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""] # 优先按段落切分
)

# 执行切分
# 注意：这里用的是 split_documents 而不是 split_text
splits = text_splitter.split_documents(docs)
print(f"文档被切分成了 {len(splits)} 个片段。")

# 使用 DashScope Embeddings  
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=api_key
)
vetorstore = FAISS.from_documents(splits, embeddings)
retriever = vetorstore.as_retriever()

# ---- 核心融合部分 ----

# 1. 定义 Prompt
# 注意：这里同时使用了 MessagesPlaceholder (Memory) 和 {context} （RAG）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个 学习 助手。请根据下面的上下文和聊天历史回答问题。如果你不知道答案，就说你不知道，不要编造。"),
    # 占位符 1：存放历史聊天记录 （Memory）
    MessagesPlaceholder(variable_name="history"),
    # 占位符 2：存放 RAG 检索到的文档 （RAG）
    ("human", "上下文信息: \n{context}\n\n问题: {input}")
])

# 2. 定义格式化函数
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# 3. 构建内部链 Chain
rag_chain = (
    RunnablePassthrough.assign(
        # itemgetter("input") 提取输入
        # | retriever 进行检索（这一步输出的是 List）
        # | format_docs 把 List 变成 String
        # 注意：这里全程都是 LangChain 对象或函数的组合，符合 LCEL 规范
        context=itemgetter("input") | retriever | format_docs                                   
    )
    | prompt
    | llm
    | StrOutputParser()
)

# 4. 包装 Memory（变成有记忆的链）
# 这一步会字典在输入字典里注入"history" 字段
full_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history, # 告诉它怎么存/取历史
    input_messages_key="input", # 用户最新问题的 key
    history_messages_key="history" # 历史记录的 key
)

# -------- 测试 --------

print("========== 第一轮对话 ==========")
res1 = full_chain.invoke(
    {"input": "我想在15天内完成必做作业，不做选做作业，帮我安排"},
    config={"configurable": {"session_id": "user_123"}}
)
print(f"AI: {res1}")

# print("========== 第二轮对话（测试 Memory + RAG 结合） ==========")
# # 注意：这里没有提“年假”两个字，只说了“满十年”
# # AI 必须集合 Memory（知道我们在聊年假） 和 RAG（知道满十年的规则） 才能给出正确的答案
# res2 = full_chain.invoke(
#     {"input": "我想？"},
#     config={"configurable": {"session_id": "user_789"}}
# )
# print(f"AI: {res2}")