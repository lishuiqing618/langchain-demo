import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 加载环境变量
load_dotenv()

# 设置 API Key 和 Base URL
api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("DASHSCOPE_BASE_URL")

# 1. 定义模型
llm = ChatOpenAI(
    base_url=base_url,
    model="qwen-plus",
    temperature=0.7  # 控制随机性，0最严谨，1最随机
)

# 2. 定义提示词模板
template = "给我讲一个关于 {topic} 的笑话。"
prompt = ChatPromptTemplate.from_template(template)

# 3. 创建链
# 这里的"|" 符号就像水管，把左边的输出，直接传给右边作为输入
chain = prompt | llm 

# 4. 调用链
# 这里的“狗”会替换掉 prompt 中的 {topic}
# prompt 生成后传给 llm，llm 生成结果
response = chain.invoke({"topic": "狗"})

# 5. 打印
print(f"AI 的回复：{response.content}")