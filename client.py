from langserve import RemoteRunnable

# 链接到我们刚刚启动的服务
remote_agent = RemoteRunnable("http://localhost:8000/agent")

print("========== 第一轮对话 ==========")
# 像调用本地 chain 一样调用它
response = remote_agent.invoke(
    "我叫小明，今年5岁。",
    config={"configurable": {"session_id": "user_web_123"}},
)

print(f"AI: {response}")


print("========== 第二轮对话（测试记忆） ==========")
response = remote_agent.invoke(
    "我今年几岁？",
    config={"configurable": {"session_id": "user_web_123"}},
)
print(f"AI: {response}")
# print(response["output"])   # AgentExecutor 返回的是一个字典，我们要取 output