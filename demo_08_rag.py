import os
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai import ChatOpenAI
# 导入 DashScope 专用的 Embeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter # 用于从字典中取值

# 文本切分器
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 向量存储
from langchain_community.vectorstores import FAISS

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("DASHSCOPE_BASE_URL")

llm = ChatOpenAI(
    base_url=base_url,
    model="qwen-plus",
    temperature=0.7  # 0最严谨，1最随机
)

# ==========================================
# 步骤 1：准备数据（模拟一份员工手册）
# ==========================================
raw_text = """
    公司休假制度：    
    1. 年假：员工入职满一年后，享有5天年假；满十年享有10天。
    2. 病假：员工每月享有1天带薪病假，需提供医院证明。
    3. 事假：需提前三天申请，扣除当日全额工资。
    4. 远程办公：每周五允许全员居家办公，但需在钉钉上打卡。
    请注意：所有请假申请必须经过直属经理批准。
"""

# ==========================================
# 步骤 2：文档切分
# ==========================================
# 为什么切分？因为 LLM 一次读不完太长的文章。
# 1000 是一块的大小，200 是块之间的重叠（保证上下文连贯）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=20,
    length_function=len
)

splits = text_splitter.split_text(raw_text)
print(f"----文档被切分成了 {len(splits)} 个片段----")

# ==========================================
# 步骤 3：向量化与存储
# ==========================================
# 这里需要用到 OpenAI 的 Embedding 模型，把文字变成向量
# 注意：这一步会消耗 API Token（如果用 OpenAI Embedding）
# embeddings = OpenAIEmbeddings(
#    base_url=base_url,  # 如果你的 embedding 也是走 qwen/oneapi，这里需要配置
#    model="text-embedding-v1" # 这里的 model 取决于你的 provider，如果不确定可以不填
# )
# 替换为：使用 DashScope 原生类
# 注意：DashScopeEmbeddings 默认读取环境变量 DASHSCOPE_API_KEY
# 为了兼容现在的代码，我们手动把 OPENAI_API_KEY 传进去
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=api_key # 复用环境变量中的 OPENAI_API_KEY
)

# 创建向量数据库（FAISS）
vectorstore = FAISS.from_texts(splits, embeddings)

# 把它变成一个“检索器”
# k=2 表示每次只找最相似的 2 个片段
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ==========================================
# 步骤 4：构建 RAG 链（核心难点）
# ==========================================

# 4.1 定义 Prompt
# 这个 Prompt 需要两个变量：{context} （检索到的片段）和 {question}（用户问题）
prompt = ChatPromptTemplate.from_template("""
你是一个公司 HR 助手。请只根据下面的已知信息回答问题。
如果你不知道答案，就说你不知道，不要编造。

已知信息：
{context}
                                          
问题：
{question}
""")

# 4.2 定义文档格式化函数
# 检索器返回的是 Document 对象列表，我们需要把它们拼成字符串
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# 4.3 使用 LCEL 语法组合链
# 这里用到了 itemgetter 函数，它相当于 lambda x: x["question"]
rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,    # 1. 拿问题 -> 检索 -> 格式化
        "question": itemgetter("question")                              # 2. 拿问题 -> 直接传
    }
    | prompt    # 3. 填充 Prompt
    | llm       # 4. LLM 生成
    | StrOutputParser() # 5. 解析输出
)

# ==========================================
# 步骤 5：测试 RAG 链
# ==========================================

print("---- 测试 1：问关于病假 ----")
response1 = rag_chain.invoke({"question": "病假有多少天？"})
print(f"AI：{response1}\n")

print("---- 测试 2：问关于考勤 ----")
response2 = rag_chain.invoke({"question": "周四需要去公司吗？"})
print(f"AI：{response2}\n")

print("---- 测试 3：问未提及的内容 ----")
response3 = rag_chain.invoke({"question": "公司有食堂吗？"})
print(f"AI：{response3}\n")