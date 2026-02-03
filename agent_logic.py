import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 引入 RAG 相关（模拟员工手册数据）
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
  
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("DASHSCOPE_BASE_URL")

llm = ChatOpenAI(
    base_url=base_url,
    model="qwen-plus"
)

# ============================================================================
# 1. 创建工具
# ============================================================================
@tool
def multiply(a: int, b: int) -> int:
    """ 计算两个数字的乘积 a*b """
    return a * b

# ---- 工具 B：员工手册查询（把 RAG 变成工具） ----
# 这里我们硬编码一份简单的数据作为演示
raw_text = """
    公司休假制度：    
    1. 年假：员工入职满一年后，享有5天年假；满十年享有10天。
    2. 病假：员工每月享有1天带薪病假，需提供医院证明。
    3. 事假：需提前三天申请，扣除当日全额工资。
    4. 远程办公：每周五允许全员居家办公，但需在钉钉上打卡。
    请注意：所有请假申请必须经过直属经理批准。
"""
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = text_splitter.split_text(raw_text)
embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=api_key)
vectorstore = FAISS.from_texts(splits, embeddings)
retriever = vectorstore.as_retriever()

@tool
def query_company_manual(question: str) -> str:
    """ 查询员工手册来回答公司制度问题。输入应该是用户的具体问题。 """
    docs = retriever.invoke(question)
    return "\n\n".join([d.page_content for d in docs])

# 把工具放入列表
tools = [multiply, query_company_manual]

# ============================================================================
# 2. 创建代理 Agent
# ============================================================================

# 定义 Prompt
# 这里必须有 {agent_scratchpad}，这是 Agent 用来记录思考过程的地方
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手。请使用下面的工具来回答用户的问题"),
    MessagesPlaceholder(variable_name="chat_history", optional=True),   # 支持对话记录
    ("human", "{input}"),
    # 必须包含整个占位符，用于显示工具调用的中间过程
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 2.1 创建 Agent（大脑）
# bind_tools 把工具列表告诉 LLM，让 LLM 知道它有哪些能力
agent = create_tool_calling_agent(llm, tools, prompt)

# 2.2 创建 AgentExecutor（执行者）
# verbose=True 会打印出 Agent 的思考过程，非常有用！
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 导出给 server.py 使用
__all__ = ["agent_executor"]

""" 
# ============================================================================
# 3. 测试
# ============================================================================

print("========== 测试：复杂的混合问题 ==========")
# 这个问题需要：
# 1. 先调用 query_company_manual 查出 “10天”
# 2. 再调用 multiply 计算 10 * 5 = 50
response = agent_executor.invoke({"input": "公司规定满十年的年假是多少天？如果我有 5 个同事（其中3个同时满2年，2个同事满10年）一共有多少天年假？"})

print("\n========== 最终答案 ==========")
print(response["output"]) 
"""

