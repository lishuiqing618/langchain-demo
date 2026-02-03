import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 加载环境变量
load_dotenv()

# 设置 API Key 和 Base URL
api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("DASHSCOPE_BASE_URL")

# 定义模型
llm = ChatOpenAI(
    base_url=base_url,
    model="qwen-plus",
    temperature=0.7  # 控制随机性，0最严谨，1最随机
)

prompt = ChatPromptTemplate.from_template(
    "请列出 {num} 个关于 {topic} 的有趣事实。不要任何开场白，直接列出。"
)

# 1. 定义输出解析器
# 它的作用很简单：把 AIMessage 对象里的 .content 提取出来编程字符串
output_parser = StrOutputParser()

# 2. 创建更长的链
# Prompt -> LLM（模型）-> OutputParser（解析器）
chain = prompt | llm | output_parser

# 3. 调用链
response = chain.invoke({"topic": "猫", "num": 5})

print(f"类型：{type(response)}")
print(f"内容：\n{response}")