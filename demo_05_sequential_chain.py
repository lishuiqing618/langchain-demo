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

# ---- 定义第一步：生成句子 ----
prompt_1 = ChatPromptTemplate.from_template(
    "根据这个单词生成一个英文句子： {word} 。"
)
chain_1 = prompt_1 | llm | StrOutputParser()

# ---- 定义第二步：翻译 ----
# 注意看这里的输入变量！它需要接收第一步的输出
prompt_2 = ChatPromptTemplate.from_template(
    "把这句话翻译成中文：{line}"    # "把这句话翻译成中文：{input}"
)
chain_2 = prompt_2 | llm | StrOutputParser()

# ---- 定义第三步：评价句子 ----
# 注意看这里的输入变量！它需要接收第一步的输出
prompt_3 = ChatPromptTemplate.from_template(
    "评价这个翻译的信、达、雅程度：{gree}"  #  "评价这个翻译的信、达、雅程度：{input}"
)
chain_3 = prompt_3 | llm | StrOutputParser()

# ---- 组合两条链 ----
# chain_1 的输出，自动作为 chain_2 的输入
# 注意：chain_2 的 Prompt 里的变量名 {line} 要和 chain_1 的输出对应上
# LangChain 会自动将 chain_1 的输出作为 chain_2 的输入 {line}
full_chain = {"gree": chain_1 | chain_2  } | chain_3

# 也可以这样写：但需要把 chain_1 后面所有的chain的变量都改成{input}--这是规定：前面链的输出为后面链的输入，后面链的输入为{input}
# full_chain = chain_1 | chain_2 | chain_3  

# ---- 调用完整链条 ----
print("正在评价这个翻译的信、达、雅程度...")
# 我们只需要给最开始的 chain_1 传入 word 即可
result = full_chain.invoke({"word": "panda"})
print(result)