import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 1. 设置 API Key (为了安全，建议不要直接写在代码里，而是设置环境变量)

api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("DASHSCOPE_BASE_URL")

if not api_key:
    raise ValueError("请确保.env文件已正确配置，并设置环境变量 OPENAI_API_KEY")

# 2. 初始化客户端
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

# 3. 发送请求
print("正在询问 qwen-plus...")
response = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "你是一个乐于助人的助手。"},
        {"role": "user", "content": "用一句话介绍一下 Python。"}
    ],
    temperature=0.7 # 控制随机性，0最严谨，1最随机
)

# 4. 输出结果
print(f"Qwen的回答: {response.choices[0].message.content}")