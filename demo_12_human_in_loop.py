import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
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
    human_feedback: str # 新增：用来存人工输入 "approve" 或 "reject"

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

    # 新增逻辑：如果监测到有反馈（而且不是 ok），说明刚被拒绝过。
    # 我们在这里清空它，为下一次审核做准备。
    if state.get("human_feedback"):
        print("🔄 [Agent] 检测到之前的反馈，正在重置状态并重新生成...")
        # 返回更新，清空 feedback
        # 注意：这里我们依然要调用 LLM，所以返回的内容既包含新消息，也包含状态更新
        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "human_feedback": ""    # 清空
        }

    # 调用大模型
    response = llm_with_tools.invoke(messages)
    # 返回新的消息列表（原来的 + 新生成的）
    return {"messages": [response]}

# 节点 B：工具执行节点
# LangGraph 提供了预置的 ToolNode，可以直接用
tool_node = ToolNode(tools)

# 🔥 新增：人工审核节点
def human_node(state: AgentState):
    # 获取 Agent 的最后一条回复
    last_message = state["messages"][-1]

    print("\n" + "="*30)
    print(f"👨‍💻 人工审核阶段")
    print("="*30)
    print(f"AI 建议：{last_message.content}")
    print("-"*30)

    # 模拟等待用户输入（在实际 API 中，这里会挂起流程，等待前端传回 config）
    user_input = input("请审核（输入'ok' 批准，其他任何内容拒绝）：")

    # 返回一条 HumanMessage，记录审核意见
    # 这条消息会加入 State，并被 Agent 看到
    return {
        "messages":[HumanMessage(content=f"人工审核结果：{user_input}")],
        "human_feedback": user_input
    }


# ============================================================================
# 5. 定义 Edges（边/路由逻辑）
# ============================================================================

# 路由函数：决定下一步是去工具，还是结束
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    # 1. 如果需要调工具，去工具节点
    if isinstance(last_message, AIMessage) and last_message.tool_calls: 
        # 去工具节点
        return "tools"
    
    # 2. 检查 State 里是否有人工反馈
    # 我们不再通过判断 last_message 类型，而是直接看状态字段
    feedback = state.get("human_feedback")

    # 如果 feedback 不为空，说明刚刚经过了 human_node
    if feedback:
        if feedback == "ok":
            # 批准了！结束
            print("✅ 审核通过，流程结束。")
            return END
        else:
            # 拒绝了！回退给 Agent 重新思考
            print("❌ 审核拒绝，退回 Agent 重新生成。")
            return "agent"
        
    # 3. Agent 正常生成了回复，还没给人看，先去人工节点
    return "human"

# ============================================================================
# 6. 构件图（核心部分）
# ============================================================================

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("human", human_node) # 🔥 添加人工节点

# 设置入口点
workflow.set_entry_point("agent")

# 添加边（Conditional Edge：条件边）
# 从 agent 出发，根据 should_continue 的结果决定去向
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "agent": "agent",   # 🔥 关键：允许返回 agent (自旋/重试)
        "human": "human",   # Agent 生成回答后，去 human
        END: END            # (理论上不会直接到这，因为都要先审核)
    }
)

# 添加普通边
# 工具执行完后，必须回到 Agent，让它思考下一步
workflow.add_edge("tools", "agent")

# 🔥 人工节点执行完，回 Agent（或者通过 should_continue 判断去哪）
# 这里我们让它回路由函数统一判断
# workflow.add_edge("human", "agent")
workflow.add_conditional_edges(
    "human",
    should_continue,    # 复用同一个路由
    {
        "tools": "tools",
        "agent": "agent",   # 🔥 关键：允许返回 agent (自旋/重试)
        "human": "human",   # Agent 生成回答后，去 human
        END: END            # (理论上不会直接到这，因为都要先审核)
    }
)

# ============================================================================
# 7. 编译并运行
# ===========================================================================

app = workflow.compile()

# 测试
print("===== LangGraph 人机协同 Agent 启动 =====")
inputs = {
    "messages": [
        HumanMessage(content="3乘以3等于多少？")
    ],
    "human_feedback": ""    # 初始为空
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
# final_state = app.invoke(safe_inputs)
# print(final_state["messages"][-1].content)