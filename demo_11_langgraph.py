import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# ============================================================================
# 1. 定义 Tools（和之前一样）
# ============================================================================
@tool
def multiply(a: int, b: int) -> int:
    """ 两个数相乘 a*b """
    return a * b   

@tool
def query_company_manual(question: str) -> str:
    """ 搜索员工手册来回答公司制度问题。输入应该是用户具体的问题。 """
    return "公司规定：满十年享有10天年假。"

tools = [multiply, query_company_manual]

# ============================================================================
# 2. 定义 State（状态）
# ============================================================================
# 这个类定义了我们在图里传递的数据结构
class AgentState(TypedDict):
    # message 是一个列表，包含所有历史消息
    messages: Annotated[list, lambda x, y: x + y]

# ============================================================================
# 3. 定义 LLM
# ============================================================================
load_dotenv()
llm = ChatOpenAI(
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen-plus",
    temperature=0
)

# 把工具绑定给 LLM
llm_with_tools = llm.bind_tools(tools)

# ============================================================================
# 4. 定义 Nodes（节点）
# ============================================================================

# 节点 A: Agent 思考节点
def agent_node(state: AgentState):
    messages = state["messages"]
    # 调用大模型
    response = llm_with_tools.invoke(messages)
    # 返回新的消息列表（原来的 + 新生成的）
    return {"messages": [response]}

# 节点 B：工具执行节点
# LangGraph 提供了预置的 ToolNode，可以直接用
tool_node = ToolNode(tools)

# ============================================================================
# 5. 定义 Edges（边/路由逻辑）
# ============================================================================

# 路由函数：决定下一步是去工具，还是结束
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    # 如果最后一条消息包含工具调用请求
    if last_message.tool_calls: 
        # 去工具节点
        return "tools"
    # 否则，结束
    return END

# ============================================================================
# 6. 构件图（核心部分）
# ============================================================================

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# 设置入口点
workflow.set_entry_point("agent")

# 添加边（Conditional Edge：条件边）
# 从 agent 出发，根据 should_continue 的结果决定去向
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

# 添加普通边
# 工具执行完后，必须回到 Agent，让它思考下一步
workflow.add_edge("tools", "agent")

# ============================================================================
# 7. 编译并运行
# ===========================================================================

app = workflow.compile()

# 测试
print("===== LangGraph Agent 启动 =====")
inputs = {
    "messages": [
        HumanMessage(content="公司规定满十年的年假是多少天？如果我有 5 个同事，一共有多少天年假？")
    ]
}

# 🌟 在 inputs 前面加上 AgentState 进行类型断言
# 这行代码告诉编辑器：“别管了，我确定这是对的”
from typing import cast
safe_inputs = cast(AgentState, inputs)

# 🌟 stream 打印中间过程，这是 LangGraph 最大的魅力
for event in app.stream(safe_inputs):  
    for node_name, node_output in event.items():  # 遍历每个节点
        print(f"----- 节点：{node_name} -----")
        # 打印最新的一条消息
        print(f"输出：{node_output['messages'][-1].content}")

print("===== LangGraph Agent 结束 =====")
final_state = app.invoke(safe_inputs)
print(final_state["messages"][-1].content)