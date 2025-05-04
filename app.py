import streamlit as st
import requests
import json
import logging
import time
import os
from PyPDF2 import PdfReader
import io
from io import BytesIO

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 在文件开头添加配置
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
REQUEST_TIMEOUT = 120  # 120秒
MAX_RETRIES = 3  # 最大重试次数
CHUNK_SIZE = 1024 * 1024  # 1MB
MAX_PAGES_PER_REQUEST = 10  # 每次请求处理的最大页数

# 添加LLM相关函数
def generate_summary(text: str, level: int, provider: str = "openai") -> str:
    """
    生成文本摘要
    :param text: 输入文本
    :param level: 摘要层级(1-3)
    :param provider: llm提供商(openai/deepseek/ollama)
    :return: 摘要文本
    """
    try:
        if provider.lower() == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的文本摘要助手"},
                    {"role": "user", "content": f"请用中文生成第{level}级摘要，保留关键信息:\n{text}"}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
            
        elif provider.lower() == "deepseek":
            import requests
            headers = {
                "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": f"生成第{level}级摘要:\n{text}"}
                ]
            }
            response = requests.post(
                f"{os.getenv('DEEPSEEK_API_BASE', 'https://api.deepseek.com/v1')}/chat/completions",
                headers=headers,
                json=data
            )
            return response.json()["choices"][0]["message"]["content"]
            
        elif provider.lower() == "ollama":
            import requests
            response = requests.post(
                f"{os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')}/api/chat",
                json={
                    "model": os.getenv("OLLAMA_MODEL", "phi4-reasoning:plus"),
                    "messages": [
                        {"role": "user", "content": f"生成中文摘要(层级{level}):\n{text}"}
                    ],
                    "stream": False
                }
            )
            return response.json()["message"]["content"]
            
    except Exception as e:
        logger.error(f"LLM调用失败: {str(e)}")
        return f"摘要生成失败: {str(e)}"

def process_pdf(file) -> dict:
    """处理PDF文件并返回文本"""
    try:
        # 分块读取
        contents = b""
        file_size = 0
        while True:
            chunk = file.read(CHUNK_SIZE)
            if not chunk:
                break
            contents += chunk
            file_size += len(chunk)
            
        # 检查大小
        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"文件超过{MAX_FILE_SIZE//1024//1024}MB限制")
            
        # 处理PDF
        pdf_reader = PdfReader(io.BytesIO(contents))
        page_count = min(len(pdf_reader.pages), MAX_PAGES_PER_REQUEST)
        text_pages = []
        
        for i in range(page_count):
            text_pages.append(pdf_reader.pages[i].extract_text() or "")
            
        return {"text": "\n".join(text_pages), "total_pages": page_count}
        
    except Exception as e:
        logger.error(f"PDF处理错误: {str(e)}")
        raise

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
    logger.info(f"User uploaded file: {uploaded_file.name} ({uploaded_file.size} bytes)")
    
    # 检查文件大小
    if uploaded_file.size > MAX_FILE_SIZE:
        logger.error(f"File too large: {uploaded_file.size} bytes")
        st.error(f"文件过大，请上传小于{MAX_FILE_SIZE//1024//1024}MB的文件")
        st.stop()
    
    try:
        # 显示进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 分阶段处理
        status_text.text("准备处理PDF...")
        progress_bar.progress(5)
        
        # 处理PDF
        pdf_data = process_pdf(uploaded_file)
        progress_bar.progress(30)
        
        # 保存文本
        st.session_state.full_text = pdf_data["text"]
        progress_bar.progress(50)
        
        # 生成摘要
        status_text.text("生成摘要中...")
        summary_request = {
            "text": st.session_state.full_text, 
            "level": 1,
            "provider": model_provider.lower()
        }
        summary_response = requests.post(
            "http://localhost:8000/summarize/", 
            json=summary_request
        )
        progress_bar.progress(90)
        
        if summary_response.status_code == 200:
            st.session_state.summaries[1] = summary_response.json()["summary"]
            progress_bar.progress(100)
            status_text.text("处理完成！")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
        else:
            progress_bar.empty()
            status_text.error(f"摘要生成失败: {summary_response.text}")
            
    except Exception as e:
        progress_bar.empty()
        status_text.error(f"发生意外错误: {str(e)}")
        logger.error(f"Frontend error: {str(e)}", exc_info=True)

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
