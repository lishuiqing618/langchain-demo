from fastapi import FastAPI
from langserve import add_routes
from agent_logic import agent_executor
from langchain_core.runnables import RunnableLambda # 用于包装
from langchain_core.runnables.history import RunnableWithMessageHistory
from agent_logic import agent_executor
import json
import os
from datetime import datetime
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

import traceback


print(f"⚠️ 当前工作目录 (文件将保存在这里): {os.getcwd()}")

# 复用 Day 3 的文件存储类
HISTORY_FILE = "agent_chat_history.json"

# 初始化文件
if not os.path.exists(HISTORY_FILE): # 如果文件不存在
    with open(HISTORY_FILE, "w", encoding="utf-8") as f: # 创建文件
        json.dump({}, f)

class FileChatMessageHistory(InMemoryChatMessageHistory):
    def __init__(self, session_id: str):
        super().__init__()
        object.__setattr__(self, "_session_id", session_id)
        object.__setattr__(self, "_file_path", HISTORY_FILE)
        self._load_messages()
    
    @property
    def session_id(self):
        # 提供一个只读属性方便访问
        return getattr(self, "_session_id", None)

    def _load_messages(self):
        try: 
            with open(HISTORY_FILE, "r", encoding="utf-8") as f: # 读取文件
                data = json.load(f) # 解析 JSON
                # 把 JSON 转成 LangChain 的消息列表
            raw = data.get(self.session_id, {}).get("messages", [])
            #if not raw: return

            if raw:
                # 关键：使用 messages_from_dict 解析
                loaded_messages = messages_from_dict(raw)
                for msg in loaded_messages: # 遍历消息列表
                    super().add_message(msg)
                print(f"📂 加载了 {len(loaded_messages)} 条历史记录")
        except Exception as e: # 如果解析失败，则清空文件
            print(f"❌ 加载历史失败 (文件可能为空或格式错误): {e}")

    def add_message(self, message: BaseMessage):
        super().add_message(message)
        # 🌟 调试步骤 2: 确认是否进入保存逻辑
        print(f"💾 正在保存消息... 类型: {message.type}, 内容预览: {message.content[:20]}...")
        self._save_to_file()
    
    def _save_to_file(self):
        # 🌟 调试步骤 3: 去除所有 try...except 的 pass，让报错直接炸出来        
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f: # 读取文件
                all_data = json.load(f) # 解析 JSON
        except Exception as e:
            print(f"❌ 加载历史失败 (文件可能为空或格式错误): {e}")
            all_data = {}
        
        try:
            # 关键: 使用 messages_to_dict 转成 JSON
            base_dicts = messages_to_dict(self.messages)

            # 添加时间戳
            for item in base_dicts: # 遍历字典列表
                if "data" in item: # 如果有 data 字段，则删除"
                    item["data"]["timestamp"] = datetime.now().isoformat() 

            # 构建当前 Session 数据
            current_session_dict = {
                "meta": {"session_id": self.session_id, "updated_at": datetime.now().isoformat()},
                "messages": base_dicts
            }

            # 更新并保存
            all_data[self.session_id] = current_session_dict

            with open(HISTORY_FILE, "w", encoding="utf-8") as f: # 创建文件
                json.dump(all_data, f, ensure_ascii=False, indent=4)

            print(f"✅ 保存成功！文件路径: {HISTORY_FILE}")
            
        except Exception as e:
            # 🌟 调试步骤 4: 打印完整的错误堆栈
            print(f"💥💥💥 保存失败！详细错误如下：")
            print(traceback.format_exc())

def get_session_history(session_id: str) -> FileChatMessageHistory:
    return FileChatMessageHistory(session_id)


# 1. 定义 FastAPI 应用
app = FastAPI(
    title="LangChain Agent Server V2",
    description="带记忆的 Agent API 服务",
    version="2.0",
)

# 包装 Agent
# 定义一个预处理函数，把字符串转成字典
def prep_input(x: str) -> dict:
    return {"input": x}

# 2. 🌟 后处理：提取回答文本，不传复杂字典
def extract_output(x: dict) -> str:
    # 从返回的大字典里只拿出 'output' 对应的字符串
    # 如果没有 output，返回一个默认提示
    return x.get("output", "无回复")

# 1. 包装 Agent，加上记忆
agent_with_history = RunnableWithMessageHistory(
    agent_executor, 
    get_session_history,    # 告诉它怎么存取历史
    input_messages_key="input", # 对应 AgentExecutor 的输入 key
    history_messages_key="chat_history" # 必须和 Agent 的 prompt 兼容
)

# 注意： 这里有一个坑！ Day 7 的 agent_executor 使用的 Prompt 只有 {input} 和 {agent_scratchpad}
# 如果要加记忆，我们需要把 Day 7 的 Prompt 改成包含 {chat_history} 的！
# 为了不让你改太多文件，我们可以再 server.py 这里重新定义一个带 history 的 agent
# （为了简化，我们假设 agent_executor 已经支持 history，或者这里不做深究，
#  重点演示如何传 config。如果运行报错 "Missing input key: chat_history" ，请看下面的提示）

# 2. 组合链
# 创建一个新链：字符串 -> 字典 -> AgentExecutor
# 注意 Swagger UI 就会知道它需要接收一个 String，然后返回一个字典 Dict
agent_app = RunnableLambda(prep_input) | agent_with_history | RunnableLambda(extract_output)

# 2. 添加 LangChain 路由
# path="/agent" 是接口路径前缀
add_routes(
    app, 
    agent_app, 
    path="/agent"
)

# 3. （可选）根路径提示
@app.get("/")
def read_root():
    return {"message": "请访问 /docs 查看接口文档"}

if __name__ == "__main__":
    import uvicorn
    # 启动服务：host=t = 0.0.0.0 允许外网访问，port=8000 
    uvicorn.run(app, host="0.0.0.0", port=8000)