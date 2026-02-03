import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 加载环境变量
load_dotenv()

# 1. 设置 API Key (为了安全，建议不要直接写在代码里，而是设置环境变量)
api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("DASHSCOPE_BASE_URL")

if not api_key:
    raise ValueError("请确保.env文件已正确配置，并设置环境变量 OPENAI_API_KEY")

# 2. 初始化模型
# 这里的 ChatOpenAI 是对 OpenAI 的封装
llm = ChatOpenAI(
    base_url=base_url,
    model="qwen-plus",
    temperature=0 # 控制随机性，0最严谨，1最随机
)

# --- 核心知识点：PromptTemplate ---

# 3. 定义一个提示词模板
# 注意这里的 {topic} 和 {language}, 他们是变量，稍后会被填充
template = "把下面的内容翻译成{language}：\n{text}"

# 4. 创建一个 PromptTemplate 对象
prompt = ChatPromptTemplate.from_template(template)

# 5. 格式化提示词
# 我们可以把 template 想象成一个函数，传入参数，得到最终的提示词 Prompt
final_prompt = prompt.format(text="我们可以把 template 想象成一个函数，传入参数，得到最终的提示词 Prompt", language="英文")

print(f"--- 发送给 AI 的完整 prompt ---\n{final_prompt}\n----------------------------------")

# 6. 调用模型
# 直接把格式化好的 Prompt 传给模型llm.invoke()，得到结果
response = llm.invoke(final_prompt)

# 7. 输出结果
print(f"--- AI 的回复 ---\n{response}\n----------------------------------")