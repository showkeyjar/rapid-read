import streamlit as st
import requests
import json

st.title("PDF快速阅读助手")
st.subheader("层次化摘要工具")

# 模型选择
model_provider = st.selectbox(
    "选择大模型提供商",
    ("OpenAI", "DeepSeek", "Ollama"),
    index=0
)

# 初始化会话状态
if 'summaries' not in st.session_state:
    st.session_state.summaries = {}
if 'current_level' not in st.session_state:
    st.session_state.current_level = 1

# PDF上传
uploaded_file = st.file_uploader("上传PDF文件", type="pdf")

if uploaded_file:
    # 发送到后端处理
    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
    response = requests.post("http://localhost:8000/upload/", files=files)
    
    if response.status_code == 200:
        text = response.json()["text"]
        st.session_state.full_text = text
        
        # 生成初始摘要
        summary_request = {
            "text": text, 
            "level": 1,
            "provider": model_provider.lower()
        }
        summary_response = requests.post(
            "http://localhost:8000/summarize/", 
            json=summary_request
        )
        
        if summary_response.status_code == 200:
            st.session_state.summaries[1] = summary_response.json()["summary"]

# 显示当前层级摘要
current_summary = st.session_state.summaries.get(st.session_state.current_level, "")
if current_summary:
    if st.session_state.current_level >= 5:  # 显示原始内容
        st.subheader("原始内容")
        st.text_area("完整内容", value=st.session_state.full_text, height=400)
    else:
        st.subheader(f"第{st.session_state.current_level}层摘要")
        
        # 分割摘要为可点击的点
        points = [p for p in current_summary.split('\n') if p.strip()]
        for i, point in enumerate(points):
            if st.button(point, key=f"level_{st.session_state.current_level}_point_{i}"):
                # 点击时生成更深层摘要
                if st.session_state.current_level < 5:  # 最多5层
                    next_level = st.session_state.current_level + 1
                    summary_request = {
                        "text": st.session_state.full_text, 
                        "level": next_level,
                        "provider": model_provider.lower()
                    }
                    summary_response = requests.post(
                        "http://localhost:8000/summarize/", 
                        json=summary_request
                    )
                    
                    if summary_response.status_code == 200:
                        st.session_state.summaries[next_level] = summary_response.json()["summary"]
                        st.session_state.current_level = next_level
                        st.experimental_rerun()

# 导航控制
col1, col2 = st.columns(2)
with col1:
    if st.button("← 上一层级") and st.session_state.current_level > 1:
        st.session_state.current_level -= 1
        st.experimental_rerun()
with col2:
    if st.button("重置到第一层"):
        st.session_state.current_level = 1
        st.experimental_rerun()
