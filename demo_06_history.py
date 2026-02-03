from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

# 1. 初始化一个内存中的历史记录对象
# 在实际应用中，通常会换成 Redis 或 数据库 存历史
history = InMemoryChatMessageHistory()

# 2. 添加第一轮对话
history.add_user_message("你好，我叫小米。")
history.add_ai_message("你好小米！很高兴认识你。")

# 3. 添加第二轮对话
history.add_user_message("我叫什么名字？")

# 4. 查看现在的对话历史
print("----当前对话历史----")
for message in history.messages:
    print(f"{message.type}: {message.content}")

print("\n---- 发送给 AI 的内容 ----")
# 如果我们把 history.messages 发给 AI, AI 就能回答了
print(history.messages)