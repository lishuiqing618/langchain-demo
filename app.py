import streamlit as st
from typing import cast
from langchain_core.messages import HumanMessage, AIMessage

# 导入我们刚才写好的 Multi-Agent 图
# 注意：这里回加载 .env 并初始化模型
from demo_13_multi_agent import app, TeamState

# =============================================================================
# 辅助函数：提取真正的文章内容
# =============================================================================
def extract_article(mesages):
    """ 
    从消息列表中倒序查找，跳过 Publisher 的评论，
    找到 Writer 生成的文章正文。
    """
    for msg in reversed(mesages):
        content = msg.content
        # 如果消息里面包含这些关键词，说明是 Publisher 的 "官话" ，跳过
        if "审核" in content or "通过" in content or "不通过" in content:
            continue
        # 否则，这里就是我们要的文章
        return content
    
    return "未找到文章内容"




# =============================================================================
# 1. 页面配置
# =============================================================================
st.set_page_config(
    page_title="AI Content Team",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI 创作团队 (Multi-Agent)")
st.markdown("输入一个主题，让 AI 团队（研究员、作家、发布者）自动为你创作文章。")

# =============================================================================
# 2. 侧边栏：历史记录（简单实现）
# =============================================================================
with st.sidebar:
    st.header("团队状态")
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    for i, item in enumerate(st.session_state.history):
        with st.expander(f"任务 #{i+1}: {item['topic'][:20]}..."):
            st.text_area("结果", item['result'], height=200, key=f"history_result_{i}")

# =============================================================================
# 3. 用户输入区
# =============================================================================
user_input = st.text_area("请输入创作主题：", height=200, placeholder="例如：马斯克的星舰发射...")

if st.button("🚀 开始创作", type="primary"):
    if not user_input:
        st.warning("请先输入一个主题！")
    else:
        # ===========================================================================
        # 4. 运行 Multi-Agent （核心逻辑）
        # ===========================================================================

        # 2. 准备输入
        inputs = {"messages": [HumanMessage(content=user_input)]}
        safe_inputs = cast(TeamState, inputs)

        final_result = ""
        # 3. 创建状态栏容器
        # st.status 是一个可以折叠的进度条
        with st.status("🏢 团队正在工作中...", expanded=True) as status:

            # 4. 开始流式执行
            try: 
                for event in app.stream(safe_inputs):
                    for node_name, node_output in event.items():
                        print(f"-->：{node_name}")
                        if node_name == "__start__" or node_name == "__end__":
                            continue

                        # 根据节点更新标题和日志
                        if node_name == "Researcher":
                            status.update(label="🔍 [Researcher] 正在联网搜索...", state="running")
                            # 🔥 关键：使用 st.write 追加日志，更稳定
                            status.write("🔍 研究员正在查阅最新资料...")

                        elif node_name == "Writer":
                            status.update(label="✍️  [Writer] 正在撰写文章...", state="running")
                            status.write("✍️ 作家正在根据资料撰写内容...")

                        elif node_name == "Publisher":
                            status.update(label="📢 [Publisher] 正在审核...", state="running")
                            # 判断审核结果
                            msg_content = ""
                            if isinstance(node_output, dict) and "messages" in node_output:
                                msg_content = node_output["messages"][-1].content
                            
                            if "不通过" in msg_content:
                                status.write(f"📢 **审核驳回**: {msg_content}")
                            elif "通过" in msg_content:
                                status.write(f"📢 **审核通过**: 文章已发布！")
                            else:
                                status.write("📢 发布者正在进行质量检查...")

                # 获取最终结果
                final_state = app.invoke(safe_inputs)
                final_result = extract_article(final_state["messages"])

                # 标记完成
                status.update(label="✅ 任务完成！", state="complete", expanded=False)

            except Exception as e:
                status.update(label="⚠️ 错误！", state="error")
                status.write(f"⚠️ 错误：{e}")        

        # ===========================================================================
        # 5. 展示最终结果
        # ===========================================================================
        st.divider()
        st.subheader("📄 最终文章")
        st.markdown(final_result)

        # 保存到历史记录
        st.session_state.history.append({
            "topic": user_input,
            "result": final_result
        })

        # 提供下载按钮
        st.download_button(
            label="📥 下载文章",
            data=final_result,
            file_name=f"blog_{user_input[:10]}.txt",
            mime="text/plain"

        )

                       