import json
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("DASHSCOPE_BASE_URL")

llm = ChatOpenAI(
    base_url=base_url,
    model="qwen-plus",
    temperature=0.7  # 0最严谨，1最随机
)

# ---- 1. 定义 Prompt（关键变化）----
# 注意这里多了一个 {messages} 占位符!
# 这是告诉 LangChain：请把历史聊天记录自动填在这个位置
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手。"),
    # 这是一个特殊的占位符，用于存放历史消息
    MessagesPlaceholder(variable_name="messages"),
    ("user", "{input}")  # 用户最新的输入
])

# ---- 2. 定义链 ----
chain = prompt | llm | StrOutputParser()

# ---- 3. 定义一个“获取历史记录”的函数 ----
# 这是一个工厂函数，根据 session_id 返回对应的历史对象
# 数据结构：
"""  
分文件存储
/data/chat_history/
    ├── user_123.json
    ├── user_456.json
    └── user_789.json

{
    "meta": {
        "session_id": "user_123",
        "created_at": "2023-10-27T10:00:00Z", // 创建时间
        "updated_at": "2023-10-27T10:05:00Z", // 最后更新时间
        "title": "学习LangChain计划",          // 对话标题（方便在列表展示）
        "tags": ["学习", "编程"]              // 标签
    },
    "messages": [
        {
            "role": "human",
            "content": "我想学 LangChain",
            "timestamp": "2023-10-27T10:00:05Z" // ✅ 增加每条消息的时间戳
        },
        {
            "role": "ai",
            "content": "太棒了，我们可以从环境搭建开始...",
            "timestamp": "2023-10-27T10:00:08Z"
        }
    ]
}

"""

# 定义文件路径
HISTORY_FILE = "chat_history.json"

# 1. 初始化文件（如果不存在）
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f: 
        json.dump({}, f)

# 2. 自定义一个 ChatHistory 类，继承自 BaseChatMessageHistory
class FileChatMessageHistory(InMemoryChatMessageHistory):
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

# 工具函数：查看所有会话统计
def get_all_session_stats():
    """ 获取所有会话的统计信息 """
    if not os.path.exists(HISTORY_FILE):
        return {}
    
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        stats = {}
        for session_id, session_data in all_data.items():
            meta = session_data.get('meta', {})
            stats[session_id] = {
                "message_count": len(session_data.get('messages', [])),
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at")
            }
        return stats
    except Exception as e:
        print(f"获取会话统计失败: {e}")
        return {}

# ---- 测试使用 ----
def get_session_history(session_id: str) -> FileChatMessageHistory:
    return FileChatMessageHistory(session_id=session_id, file_path=HISTORY_FILE)
    
# ---- 4. 包装成有记忆的链 ----
# 这是最关键的一步！
memorized_chain = RunnableWithMessageHistory(
    chain,
    get_session_history, # 告诉它怎么存/取历史
    input_messages_key="input", # 指定用户输入在 Prompt 里的变量名
    history_messages_key="messages" # 指定历史记录在 Prompt 里的变量名
)

# ---- 5. 调用有记忆的链 ----
print("第一次对话：")
res1 = memorized_chain.invoke(
    {"input": "你好，我叫小米，今年5岁，我喜欢吃苹果。"},
    config = {"configurable": {"session_id": "user_123"}} # 必须传 session_id
)
print(f"AI: {res1}")

print("\n第二次对话：")
res2 = memorized_chain.invoke(
    {"input": "我叫什么名字？"},
    config = {"configurable": {"session_id": "user_123"}} # 同一个 session_id
)
print(f"AI: {res2}")

print("\n第三次对话（新用户）：")
res3 = memorized_chain.invoke(
    {"input": "我叫什么名字？"},
    config = {"configurable": {"session_id": "user_456"}} # 新的 session_id
)
print(f"AI: {res3}（因为是新用户，AI 不知道你是谁）")

print("\n第四次对话：")
res4 = memorized_chain.invoke(
    {"input": "我今年几岁？"},
    config = {"configurable": {"session_id": "user_123"}} # 同一个 session_id
)
print(f"AI: {res4}")

print("\n第五次对话：")
res5 = memorized_chain.invoke(
    {"input": "我喜欢吃什么？"},
    config = {"configurable": {"session_id": "user_123"}} # 同一个 session_id
)
print(f"AI: {res5}")

# 查看会话统计
print("\n所有会话统计：")
stats = get_all_session_stats()
for session_id, stat in stats.items():
    print(f"会话 {session_id}: {stat['message_count']} 条消息，"
          f"创建于 {stat['created_at']}，更新于 {stat['updated_at']}")

# 查看特定会话信息
print("\n==== 特定会话信息 ====")
history_user_123 = get_session_history("user_123")
session_info = history_user_123.get_session_info()
print(f"会话 user_123 信息: {session_info}")