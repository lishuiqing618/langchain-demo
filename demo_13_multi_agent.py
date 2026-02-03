import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("DASHSCOPE_BASE_URL")

# 使用通过的 ChatOpenAI （在实际 Multi-Agent 中，不同 Agent 可以用不同的模型/温度）
llm = ChatOpenAI(base_url=base_url, model="qwen-plus")

# ==========================================
# 1. 定义 State （团队共享的白板）
# ==========================================

# 🔥 定义一个合并函数
def merge_messages(left: list[BaseMessage], right: list[BaseMessage] | BaseMessage) -> list[BaseMessage]:
    # 如果 right 是单个消息，把它包成列表
    if not isinstance(right, list):
        right = [right]
    # 返回拼接后的新列表
    return left + right

class TeamState(TypedDict):
    # messages: 存储所有的交流记录
    messages: Annotated[List[BaseMessage], merge_messages]
    # current_writer: 当前由谁负责（可选，用于路由）
    next_action: str

# ==========================================
# 2. 定义 Agents（节点）
# ==========================================

# ---- Agent A：研究员 ----
def researcher_node(state: TeamState):
    # 1. 获取用户的原始问题（State 的第一条消息）
    query = str(state["messages"][0].content)

    print(f"🔍 [Researcher] 正在联网搜索: {query}...") 
    # 2. 初始化搜索工具
    # DuckDuckGoSearchRun 是一个轻量级、免费的搜索工具
    search = DuckDuckGoSearchRun()

    # 3. 执行搜索
    try:
        # invoke 会返回搜索结果的摘要字符串
        search_result = search.invoke(query)
    except Exception as e:
        # 网络错误或超时处理
        search_result = f"搜索遇到点问题: {e}"
        print(f"⚠️ 搜索异常: {e}")

    # 4. 研究员把结果告诉团队
    message = AIMessage(content=f"这是我从网上查到的实时资料：\n\n{search_result}")
    return {"messages": [message]}

# ---- Agent B：作家 ----
def writer_node(state: TeamState):
    print("✍️ [Writer] 正在撰写博客...")
    # 1. 获取历史消息（包括研究员的资料）
    messages = state["messages"]

    # 2. 构建上下文（手动拼接资料）
    context = ""
    # 倒序遍历，找到研究员发的资料（包含 "资料" 关键字的信息）
    for msg in reversed(messages):
        if "资料" in msg.content:
            context = msg.content
            break
    
    # 3. 构建完整的 Prompt
    # 🔥 修复点：使用 f-string 或者 + 号，确保它是字符串操作
    prompt_text = (
        "你是一个6年级的小学生。请根据下面的资料写一篇简短、有趣的文章（500字以内）：\n"
        f"{context}"
    )

    # 4. 让 LLM 根据资料写文章
    # 🔥 修复点：直接传 HumanMessage，不要用奇怪的拼法   
    response = llm.invoke([HumanMessage(content=prompt_text)])

    print(f"📝 [Writer] 写作完成：{response.content[:30]}...")

    # 5. 🔥 关键修复：必须返回包含 AIMessage 的字典，以更新 State
    return {"messages": [response]}

# ---- Agent C：发布者 ----
def publisher_node(state: TeamState):
    print("📢 [Publisher] 正在审核文章...")
    messages = state["messages"]
    last_message = messages[-1]
    content = last_message.content

    # 简单的审核逻辑：检查字数是否超过 10 字
    if len(content) < 10:
        print("❌ [Publisher] 文章太短，打回重写！")
        # 返回一条反馈消息
        return {"messages": [AIMessage(content="审核不通过：文章太短，请扩充。")], "next_action": "rewrite"}
    else:
        print("✅ [Publisher] 审核通过，发布！")
        return {"messages": [AIMessage(content="审核通过！文章已发布。")], "next_action": "end"}

# ==========================================
# 3. 定义路由逻辑（不再定义为节点，而是直接再边里用）
# ==========================================

# 我们不再西 supervisor_router 函数了，而是写三个专门的路由函数

# 1. 研究员干完活，去哪？
def route_after_researcher(state: TeamState):
    # 研究员只干一次活，干完肯定把资料扔给 Writer
    return "writer"

# 2. 作家干完活，去哪?
def route_after_writer(state: TeamState):
    # 🔥 简化版：作家写完，永远发给发布者审核
    print("🔄 [Router] 初稿/重写完成，送审核...")
    return "publisher"
    
# 3. 发布者干完活，去哪？
def route_after_publisher(state: TeamState):
    # 检查审核结果
    # Publisher 的逻辑是：如果不通过，返回消息里会有“不通过”
    last_msg = state["messages"][-1]

    if "不通过" in last_msg.content:
        # 没过，回炉重造（回 Writer）
        print("🔄 [Router] 审核驳回，退回重写...")
        return "writer"
    else:
        # 过了，结束
        print("🏁 [Router] 审核通过，结束流程。")
        return END

# ==========================================
# 4. 构建图
# ==========================================

workflow = StateGraph(TeamState)

# 添加节点
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("publisher", publisher_node)

# 设置入口
workflow.set_entry_point("researcher")

# --- 添加边 ---

# 研究员 -> 路由 -> 作家
workflow.add_conditional_edges(
    "researcher",
    route_after_researcher,
    {"writer": "writer"}
)
# 作家 -> 路由 -> 发布者
workflow.add_conditional_edges(
    "writer",
    route_after_writer,
    {"publisher": "publisher"}
)
# 发布者 -> 路由 -> (作家 或 结束)
workflow.add_conditional_edges(
    "publisher",
    route_after_publisher,
    {"writer": "writer", END: END}
)

app = workflow.compile()

# ==========================================
# 🌟 修改：只有直接运行本文件时才执行测试
# 这样 import 这个文件时，不会打印一堆东西
# ==========================================
if __name__ == "__main__":
    print("=== Multi-Agent 团队启动 ===")
    raw_inputs = {
        "messages": HumanMessage(content="帮我写一篇关于特斯拉财报的博客")
    }

    # 🌟 在 inputs 前面加上 AgentState 进行类型断言
    # 这行代码告诉编辑器：“别管了，我确定这是对的”
    from typing import cast
    inputs = cast(TeamState, raw_inputs)

    final_state = app.invoke(inputs)
    print(final_state["messages"][-1].content)





""" 
# ==========================================
# 5. 运行
# ==========================================

print("=== Multi-Agent 团队启动 ===")
raw_inputs = {
    "messages": HumanMessage(content="帮我写一篇关于特斯拉财报的博客")
}

# 🌟 在 inputs 前面加上 AgentState 进行类型断言
# 这行代码告诉编辑器：“别管了，我确定这是对的”
from typing import cast
inputs = cast(TeamState, raw_inputs)

for event in app.stream(inputs):
    # 这里的 event 结构会比较深，我们简单打印节点名称
    for node_name, node_output in event.items():
        if node_name != "__start__" and node_name != "__end__":
            print(f"--> 节点 {node_name} 完成")

print("\n=== 最终成果 ===")
final_state = app.invoke(inputs)
print(final_state["messages"][-1].content)
 """