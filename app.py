import streamlit as st
import json
import logging
import time
import os
from PyPDF2 import PdfReader
import io
from io import BytesIO
import requests
import random
import concurrent.futures
from functools import partial, lru_cache
import re
import hashlib
import threading

# 确保UTF-8日志处理
class UTF8StreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            if not isinstance(msg, str):
                msg = str(msg)
            stream = self.stream
            stream.write(msg.encode('utf-8').decode('utf-8') + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log', encoding='utf-8'),
        UTF8StreamHandler()  # 替换原来的StreamHandler
    ]
)
logger = logging.getLogger(__name__)

# 在文件开头添加配置
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
REQUEST_TIMEOUT = 120  # 120秒
MAX_RETRIES = 3  # 最大重试次数
CHUNK_SIZE = 1024 * 1024  # 1MB
MAX_PAGES_PER_REQUEST = 10  # 每次请求处理的最大页数
MAX_WORKERS = 8  # 根据CPU核心数调整

# 全局模型管理器
class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model_loaded = False
            cls._instance._lock = threading.Lock()
        return cls._instance
    
    def ensure_model_loaded(self, provider: str):
        """确保模型已加载(线程安全)"""
        if self._model_loaded:
            return True
            
        with self._lock:
            if not self._model_loaded:  # 双重检查锁定
                if provider == "ollama" and not self._load_ollama_model():
                    raise RuntimeError("模型加载失败")
                self._model_loaded = True
        return True
    
    def _load_ollama_model(self) -> bool:
        """实际加载Ollama模型"""
        try:
            model_name = os.getenv('OLLAMA_MODEL')
            resp = requests.post(
                f"{os.getenv('OLLAMA_API_BASE')}/api/generate",
                json={"model": model_name, "prompt": ""},
                timeout=60
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"模型预热失败: {str(e)}")
            return False

# 在进程启动时初始化
model_manager = ModelManager()

# 添加LLM相关函数
def generate_summary(text: str, level: int, provider: str) -> str:
    """优化版摘要生成(速度优先)"""
    prompt_templates = {
        1: "请用中文简洁总结以下文本的关键内容(限300字):\n{text}",
        2: "请用中文概括以下章节的核心观点(限200字):\n{text}",
        3: "请用3-5句中文总结全文主旨:\n{text}"
    }
    
    # 更严格的长度限制
    max_input_len = {1: 8000, 2: 5000, 3: 3000}[level]
    text = text[:max_input_len]
    
    try:
        if provider.lower() == "ollama":
            response = requests.post(
                f"{os.getenv('OLLAMA_API_BASE')}/api/generate",
                json={
                    "model": os.getenv("OLLAMA_MODEL"),
                    "prompt": prompt_templates[level].format(text=text),
                    "options": {
                        "temperature": 0.5,  # 提高创造性
                        "num_ctx": 2048,     # 减小上下文窗口
                        "timeout": 60000,    # 1分钟超时
                        "num_predict": 300   # 限制输出长度
                    },
                    "stream": True
                },
                stream=True,
                timeout=90  # 1.5分钟
            )
            
            # 处理流式响应
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    full_response += chunk.get("response", "")
                    
                    # 检查是否完成或出错
                    if chunk.get("done", False):
                        break
                    if chunk.get("error"):
                        raise RuntimeError(chunk["error"])
            
            return full_response.strip()
        else:
            # 其他模型实现...
            return "其他模型暂未实现"
            
    except Exception as e:
        logger.error(f"摘要生成失败: {str(e)}")
        return f"[摘要生成失败]"
    
    return "[空摘要]"

@lru_cache(maxsize=100)
def get_text_hash(text: str) -> str:
    """生成文本指纹"""
    return hashlib.md5(text.encode()).hexdigest()

@lru_cache(maxsize=100)
def generate_with_retry(prompt: str, level: int, provider: str, max_retries: int = 3) -> str:
    """带模型恢复的重试机制"""
    for attempt in range(max_retries):
        try:
            if not check_ollama_service():
                raise RuntimeError("模型服务不可用")
                
            return generate_summary(prompt, level, provider)
            
        except Exception as e:
            logger.warning(f"尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))

def _generate_with_retry_impl(text: str, level: int) -> str:
    """
    智能重试机制
    返回: 成功时返回摘要，失败时返回错误信息
    """
    provider = "ollama"  # 当前只实现Ollama
    
    # 根据文本长度和层级计算基础超时
    base_timeout = min(60, 10 + len(text) / 1000)  # 每1000字符增加1秒，上限60秒
    
    for attempt in range(MAX_RETRIES):
        try:
            # 动态调整超时(指数退避)
            current_timeout = base_timeout * (attempt + 1)
            
            # 调用增强版生成函数
            result = generate_summary(text, level, provider)
            
            # 检查结果有效性
            if "[摘要生成失败" in result:
                raise RuntimeError(result)
                
            return result
            
        except requests.exceptions.Timeout:
            logger.warning(f"请求超时(尝试 {attempt + 1}/{MAX_RETRIES})")
            if attempt == MAX_RETRIES - 1:
                return "[错误] 请求超时，请检查Ollama服务状态"
                
        except Exception as e:
            logger.error(f"尝试 {attempt + 1} 失败: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                return f"[错误] 最终尝试失败: {str(e)}"
        
        # 智能等待(指数退避 + 抖动)
        wait_time = min(30, (2 ** attempt) + random.uniform(0, 1))
        time.sleep(wait_time)
    
    return "[错误] 摘要生成失败"

def warmup_ollama_model():
    """带进度显示的模型预热"""
    model_name = os.getenv("OLLAMA_MODEL", "phi4-reasoning:plus")
    status = st.empty()
    progress = st.progress(0)
    
    try:
        status.info(f"正在预热 {model_name}...")
        
        # 分步骤预热
        steps = ["初始化", "加载权重", "准备推理"]
        for i, step in enumerate(steps):
            progress.progress(int((i+1)/len(steps)*100))
            status.info(f"{step}...")
            time.sleep(1)  # 模拟预热步骤
            
        # 实际预热请求
        requests.post(
            f"{os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')}/api/show",
            json={"name": model_name},
            timeout=120
        )
        
        progress.progress(100)
        status.success(f"{model_name} 预热完成！")
    except requests.exceptions.Timeout:
        status.warning("预热超时（服务可能已就绪）")
    except Exception as e:
        progress.progress(0)
        status.error(f"预热失败: {str(e)}")
        raise
    finally:
        time.sleep(2)
        progress.empty()
        status.empty()

def preload_ollama_model():
    """带进度条的模型预加载"""
    if not check_ollama_service():
        raise ConnectionError("无法连接Ollama服务")
    
    model_name = os.getenv("OLLAMA_MODEL", "phi4-reasoning:plus")
    
    # 在页面顶部创建状态容器
    status_container = st.empty()
    progress_bar = st.progress(0)
    status_container.info(f"开始加载模型: {model_name}")
    
    try:
        # 分阶段预加载
        stages = ["pull", "load", "warmup"]
        for i, stage in enumerate(stages):
            progress = int((i / len(stages)) * 100)
            progress_bar.progress(progress)
            
            if stage == "pull":
                status_container.info(f"下载模型 {model_name}...")
                resp = requests.post(
                    f"{os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')}/api/pull",
                    json={"name": model_name},
                    stream=True
                )
                # 处理下载进度
                for _ in resp.iter_lines():
                    progress = min(progress + 2, 33)
                    progress_bar.progress(progress)
                    
            elif stage == "load":
                status_container.info(f"加载模型到内存...")
                resp = requests.post(
                    f"{os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')}/api/generate",
                    json={"model": model_name, "prompt": " "},
                    timeout=120
                )
                progress_bar.progress(66)
                
            else:  # warmup
                status_container.info(f"预热模型...")
                warmup_ollama_model()
                progress_bar.progress(100)
                
        status_container.success(f"{model_name} 模型已就绪！")
        
    except Exception as e:
        progress_bar.progress(0)
        status_container.error(f"加载失败: {str(e)}")
        raise
    finally:
        time.sleep(2)
        progress_bar.empty()
        status_container.empty()

def check_ollama_service() -> bool:
    """增强版模型检查"""
    try:
        # 1. 检查服务可用性
        resp = requests.get(f"{os.getenv('OLLAMA_API_BASE')}/api/tags", timeout=10)
        if resp.status_code != 200:
            logger.error(f"Ollama服务不可用 (HTTP {resp.status_code})")
            return False
            
        # 2. 检查模型是否已加载
        active_model = os.getenv('OLLAMA_MODEL', 'phi4-mini-reasoning')
        models = [m['name'] for m in resp.json().get('models', [])]
        
        if active_model not in models:
            logger.error(f"模型 {active_model} 未加载，可用模型: {', '.join(models)}")
            
            # 尝试自动拉取模型
            pull_resp = requests.post(
                f"{os.getenv('OLLAMA_API_BASE')}/api/pull",
                json={"name": active_model},
                timeout=300
            )
            
            if pull_resp.status_code == 200:
                logger.info(f"正在下载模型 {active_model}...")
                return True
            else:
                logger.error(f"模型 {active_model} 下载失败")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"模型检查失败: {str(e)}")
        return False

def setup_model_management():
    """优化模型预热"""
    with st.sidebar.expander("模型管理", expanded=True):
        if st.button("🔥 预热模型"):
            try:
                with st.spinner("正在预热模型..."):
                    if model_manager.ensure_model_loaded("ollama"):
                        st.success("✅ 模型已预热")
                    else:
                        st.error("❌ 预热失败")
            except Exception as e:
                st.error(f"预热错误: {str(e)}")
                logger.error(f"模型预热异常: {str(e)}", exc_info=True)

# 在进程启动时初始化
model_manager = ModelManager()

# 添加LLM相关函数
def generate_summary(text: str, level: int, provider: str) -> str:
    """优化版摘要生成(速度优先)"""
    prompt_templates = {
        1: "请用中文简洁总结以下文本的关键内容(限300字):\n{text}",
        2: "请用中文概括以下章节的核心观点(限200字):\n{text}",
        3: "请用3-5句中文总结全文主旨:\n{text}"
    }
    
    # 更严格的长度限制
    max_input_len = {1: 8000, 2: 5000, 3: 3000}[level]
    text = text[:max_input_len]
    
    try:
        if provider.lower() == "ollama":
            response = requests.post(
                f"{os.getenv('OLLAMA_API_BASE')}/api/generate",
                json={
                    "model": os.getenv("OLLAMA_MODEL"),
                    "prompt": prompt_templates[level].format(text=text),
                    "options": {
                        "temperature": 0.5,  # 提高创造性
                        "num_ctx": 2048,     # 减小上下文窗口
                        "timeout": 60000,    # 1分钟超时
                        "num_predict": 300   # 限制输出长度
                    },
                    "stream": True
                },
                stream=True,
                timeout=90  # 1.5分钟
            )
            
            # 处理流式响应
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    full_response += chunk.get("response", "")
                    
                    # 检查是否完成或出错
                    if chunk.get("done", False):
                        break
                    if chunk.get("error"):
                        raise RuntimeError(chunk["error"])
            
            return full_response.strip()
        else:
            # 其他模型实现...
            return "其他模型暂未实现"
            
    except Exception as e:
        logger.error(f"摘要生成失败: {str(e)}")
        return f"[摘要生成失败]"
    
    return "[空摘要]"

@lru_cache(maxsize=100)
def get_text_hash(text: str) -> str:
    """生成文本指纹"""
    return hashlib.md5(text.encode()).hexdigest()

@lru_cache(maxsize=100)
def generate_with_retry(prompt: str, level: int, provider: str, max_retries: int = 3) -> str:
    """带模型恢复的重试机制"""
    for attempt in range(max_retries):
        try:
            if not check_ollama_service():
                raise RuntimeError("模型服务不可用")
                
            return generate_summary(prompt, level, provider)
            
        except Exception as e:
            logger.warning(f"尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))

def _generate_with_retry_impl(text: str, level: int) -> str:
    """
    智能重试机制
    返回: 成功时返回摘要，失败时返回错误信息
    """
    provider = "ollama"  # 当前只实现Ollama
    
    # 根据文本长度和层级计算基础超时
    base_timeout = min(60, 10 + len(text) / 1000)  # 每1000字符增加1秒，上限60秒
    
    for attempt in range(MAX_RETRIES):
        try:
            # 动态调整超时(指数退避)
            current_timeout = base_timeout * (attempt + 1)
            
            # 调用增强版生成函数
            result = generate_summary(text, level, provider)
            
            # 检查结果有效性
            if "[摘要生成失败" in result:
                raise RuntimeError(result)
                
            return result
            
        except requests.exceptions.Timeout:
            logger.warning(f"请求超时(尝试 {attempt + 1}/{MAX_RETRIES})")
            if attempt == MAX_RETRIES - 1:
                return "[错误] 请求超时，请检查Ollama服务状态"
                
        except Exception as e:
            logger.error(f"尝试 {attempt + 1} 失败: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                return f"[错误] 最终尝试失败: {str(e)}"
        
        # 智能等待(指数退避 + 抖动)
        wait_time = min(30, (2 ** attempt) + random.uniform(0, 1))
        time.sleep(wait_time)
    
    return "[错误] 摘要生成失败"

def warmup_ollama_model():
    """带进度显示的模型预热"""
    model_name = os.getenv("OLLAMA_MODEL", "phi4-reasoning:plus")
    status = st.empty()
    progress = st.progress(0)
    
    try:
        status.info(f"正在预热 {model_name}...")
        
        # 分步骤预热
        steps = ["初始化", "加载权重", "准备推理"]
        for i, step in enumerate(steps):
            progress.progress(int((i+1)/len(steps)*100))
            status.info(f"{step}...")
            time.sleep(1)  # 模拟预热步骤
            
        # 实际预热请求
        requests.post(
            f"{os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')}/api/show",
            json={"name": model_name},
            timeout=120
        )
        
        progress.progress(100)
        status.success(f"{model_name} 预热完成！")
    except requests.exceptions.Timeout:
        status.warning("预热超时（服务可能已就绪）")
    except Exception as e:
        progress.progress(0)
        status.error(f"预热失败: {str(e)}")
        raise
    finally:
        time.sleep(2)
        progress.empty()
        status.empty()

def preload_ollama_model():
    """带进度条的模型预加载"""
    if not check_ollama_service():
        raise ConnectionError("无法连接Ollama服务")
    
    model_name = os.getenv("OLLAMA_MODEL", "phi4-reasoning:plus")
    
    # 在页面顶部创建状态容器
    status_container = st.empty()
    progress_bar = st.progress(0)
    status_container.info(f"开始加载模型: {model_name}")
    
    try:
        # 分阶段预加载
        stages = ["pull", "load", "warmup"]
        for i, stage in enumerate(stages):
            progress = int((i / len(stages)) * 100)
            progress_bar.progress(progress)
            
            if stage == "pull":
                status_container.info(f"下载模型 {model_name}...")
                resp = requests.post(
                    f"{os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')}/api/pull",
                    json={"name": model_name},
                    stream=True
                )
                # 处理下载进度
                for _ in resp.iter_lines():
                    progress = min(progress + 2, 33)
                    progress_bar.progress(progress)
                    
            elif stage == "load":
                status_container.info(f"加载模型到内存...")
                resp = requests.post(
                    f"{os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')}/api/generate",
                    json={"model": model_name, "prompt": " "},
                    timeout=120
                )
                progress_bar.progress(66)
                
            else:  # warmup
                status_container.info(f"预热模型...")
                warmup_ollama_model()
                progress_bar.progress(100)
                
        status_container.success(f"{model_name} 模型已就绪！")
        
    except Exception as e:
        progress_bar.progress(0)
        status_container.error(f"加载失败: {str(e)}")
        raise
    finally:
        time.sleep(2)
        progress_bar.empty()
        status_container.empty()

def check_ollama_service() -> bool:
    """增强版模型检查"""
    try:
        # 1. 检查服务可用性
        resp = requests.get(f"{os.getenv('OLLAMA_API_BASE')}/api/tags", timeout=10)
        if resp.status_code != 200:
            logger.error(f"Ollama服务不可用 (HTTP {resp.status_code})")
            return False
            
        # 2. 检查模型是否已加载
        active_model = os.getenv('OLLAMA_MODEL', 'phi4-mini-reasoning')
        models = [m['name'] for m in resp.json().get('models', [])]
        
        if active_model not in models:
            logger.error(f"模型 {active_model} 未加载，可用模型: {', '.join(models)}")
            
            # 尝试自动拉取模型
            pull_resp = requests.post(
                f"{os.getenv('OLLAMA_API_BASE')}/api/pull",
                json={"name": active_model},
                timeout=300
            )
            
            if pull_resp.status_code == 200:
                logger.info(f"正在下载模型 {active_model}...")
                return True
            else:
                logger.error(f"模型 {active_model} 下载失败")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"模型检查失败: {str(e)}")
        return False

def process_pdf(file) -> dict:
    """带详细诊断的PDF处理"""
    status = st.empty()
    progress = st.progress(0)
    logger.info(f"开始处理PDF: {file.name} ({file.size/1024:.1f}KB)")
    
    try:
        # 阶段1: 读取文件内容
        status.info("📂 读取PDF文件内容...")
        progress.progress(10)
        
        # 正确读取Streamlit UploadedFile对象
        contents = file.getvalue()
        progress.progress(30)
        
        # 阶段2: 解析PDF
        status.info("🔍 解析PDF结构...")
        progress.progress(35)
        pdf_reader = PdfReader(BytesIO(contents))
        page_count = len(pdf_reader.pages)
        logger.info(f"发现 {page_count} 页")
        
        # 阶段3: 提取文本
        status.info(f"📝 提取文本(共{page_count}页)...")
        text_pages = []
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text() or ""
            text_pages.append(text)
            progress.progress(35 + int((i+1)/page_count*60))
            if i % 5 == 0:  # 每5页记录一次
                logger.debug(f"已处理 {i+1}/{page_count} 页，当前页长度: {len(text)}字符")
        
        full_text = "\n".join(text_pages)
        progress.progress(100)
        status.success(f"✅ PDF处理完成！共提取 {len(full_text)} 字符")
        logger.info(f"PDF处理完成，总字符数: {len(full_text)}")
        
        return {"text": full_text, "pages": page_count}
        
    except Exception as e:
        progress.progress(0)
        status.error(f"❌ PDF处理失败: {str(e)}")
        logger.error(f"PDF处理错误: {str(e)}", exc_info=True)
        raise
    finally:
        time.sleep(2)
        progress.empty()
        status.empty()

def process_chunk_parallel(chunk: str, chunk_id: int, provider: str) -> tuple[int, str]:
    """线程安全的块处理"""
    try:
        model_manager.ensure_model_loaded(provider)
        return (chunk_id, generate_summary(chunk, level=1, provider=provider))
    except Exception as e:
        logger.error(f"分块处理失败: {str(e)}")
        return (chunk_id, f"[处理失败: {str(e)}]")

def get_optimal_chunk_size(text: str) -> int:
    """动态计算最佳分块大小"""
    # 技术文档特征检测
    if any(keyword in text[:1000] for keyword in 
           ['Abstract', '引言', '实验方法', '参考文献']):
        return 4000  # 技术文档用小分块
    
    # 小说/散文检测
    if any(keyword in text[:1000] for keyword in 
           ['第[一二三四]章', 'CHAPTER', '......']):
        return 8000  # 文学作品用大分块
        
    return 6000  # 默认值

def split_into_chunks(text: str) -> list[str]:
    """智能分块函数"""
    CHUNK_SIZE = get_optimal_chunk_size(text)
    
    # 按段落分块(保持语义完整)
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) > CHUNK_SIZE and current_chunk:
            chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk += "\n" + para
    
    if current_chunk:
        chunks.append(current_chunk)
    
    logger.info(f"将文本分成 {len(chunks)} 个块，平均大小: {sum(len(c) for c in chunks)//len(chunks)}字符")
    return chunks

def generate_hierarchical_summary(text: str, provider: str) -> dict:
    """完整的分层摘要处理(带暂停功能)"""
    # 初始化控制面板
    control_placeholder = st.empty()
    with control_placeholder.container():
        st.markdown("**处理控制**")
        col1, col2 = st.columns(2)
        with col1:
            pause_btn = st.button("⏸️ 暂停", key="pause_btn")
        with col2:
            resume_btn = st.button("▶️ 继续", key="resume_btn", disabled=True)
    
    # 初始化进度组件
    progress_bar = st.progress(0)
    status_text = st.empty()
    details = st.expander("处理详情", expanded=True)
    
    # 处理状态
    result = {'chunk_summaries': [], 'section_summaries': [], 'final_summary': "", 'errors': []}
    
    try:
        # 阶段1: 预处理
        status_text.markdown("**阶段1/4**: 预处理文本...")
        text = preprocess_text(text)
        progress_bar.progress(5)
        check_pause(pause_btn, resume_btn, control_placeholder)
        
        # 阶段2: 分块处理
        status_text.markdown("**阶段2/4**: 分块处理中...")
        chunks = split_into_chunks(text)
        
        with details:
            chunk_status = st.empty()
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                check_pause(pause_btn, resume_btn, control_placeholder)
                with details:
                    chunk_status.write(f"🔄 处理分块 {i+1}/{len(chunks)}")
                futures.append(executor.submit(process_chunk_parallel, chunk, i, provider))
                progress_bar.progress(5 + int(45*(i+1)/len(chunks)))
            
            for future in concurrent.futures.as_completed(futures):
                check_pause(pause_btn, resume_btn, control_placeholder)
                chunk_id, summary = future.result()
                result['chunk_summaries'].append((chunk_id, summary))
        
        # 阶段3: 章节摘要
        status_text.markdown("**阶段3/4**: 生成章节摘要...")
        check_pause(pause_btn, resume_btn, control_placeholder)
        
        # [章节处理逻辑]
        progress_bar.progress(80)
        
        # 阶段4: 最终摘要
        status_text.markdown("**阶段4/4**: 生成最终摘要...")
        check_pause(pause_btn, resume_btn, control_placeholder)
        
        result['final_summary'] = generate_final_summary(result['section_summaries'])
        progress_bar.progress(100)
        status_text.success("✅ 处理完成！")
        
    except Exception as e:
        progress_bar.progress(100)
        status_text.error(f"❌ 处理失败: {str(e)}")
        result['errors'].append(str(e))
    
    return result

def check_pause(pause_btn: bool, resume_btn: bool, placeholder):
    """检查并处理暂停状态"""
    if pause_btn:
        with placeholder.container():
            st.session_state.paused = True
            st.button("▶️ 继续", key="resume_btn")
        
        while st.session_state.paused:
            time.sleep(0.5)
            if resume_btn:
                st.session_state.paused = False
                break

def preprocess_text(text: str) -> str:
    """终极文本预处理"""
    # 移除PDF常见噪声
    patterns = [
        r'\d{1,3}\s+[\u4e00-\u9fa5]+\s+\d{1,3}',  # 页眉页码
        r'©.*\d{4}',  # 版权信息
        r'http[s]?://\S+',  # URL链接
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # 邮箱
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    
    # 智能段落过滤
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # 保留有效段落
        if len(line) > 25 or \
           any(c.isalpha() for c in line) or \
           line.endswith(('。', '!', '?', ';')):
            lines.append(line)
    
    return '\n'.join(lines)

def generate_final_summary(section_summaries: list[str]) -> str:
    """生成最终摘要"""
    if not section_summaries:
        return "[无有效摘要内容]"
    
    # 合并所有章节摘要
    combined = "\n\n".join(section_summaries)
    
    try:
        # 调用Ollama生成最终摘要
        response = requests.post(
            f"{os.getenv('OLLAMA_API_BASE')}/api/generate",
            json={
                "model": os.getenv("OLLAMA_MODEL"),
                "prompt": f"请根据以下章节摘要，用中文生成一个结构化的最终总结(500字以内):\n\n{combined}",
                "options": {"temperature": 0.3, "num_ctx": 4096}
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    result += data.get("response", "")
                    if data.get("done", False):
                        break
            return result.strip()
        else:
            return "[摘要生成失败]"
    except Exception as e:
        logger.error(f"最终摘要生成错误: {str(e)}")
        return "[摘要生成异常]"

# 初始化会话状态
if 'summaries' not in st.session_state:
    st.session_state.summaries = {}
    
if 'full_text' not in st.session_state:
    st.session_state.full_text = ""
    
if 'current_level' not in st.session_state:
    st.session_state.current_level = 1

# 主界面
st.title("PDF快速阅读助手")
st.subheader("层次化摘要工具")

# 侧边栏设置
with st.sidebar:
    model_provider = st.selectbox(
        "选择摘要模型",
        ["OpenAI", "DeepSeek", "Ollama"],
        key="sidebar_model_provider"
    )
    
    setup_model_management()

# PDF上传
uploaded_file = st.file_uploader("上传PDF文件", type="pdf")

if uploaded_file:
    try:
        # 使用单一状态容器
        status_container = st.empty()
        progress_bar = st.progress(0)
        
        status_container.info("开始处理PDF...")
        progress_bar.progress(10)
        
        # 1. 解析PDF
        pdf_data = process_pdf(uploaded_file)
        progress_bar.progress(30)
        
        # 2. 分层摘要
        status_container.info("开始分层摘要...")
        result = generate_hierarchical_summary(
            text=pdf_data['text'],
            provider=model_provider
        )
        progress_bar.progress(90)
        
        # 保存结果
        st.session_state["full_text"] = pdf_data["text"]
        st.session_state["chunk_summaries"] = result["chunk_summaries"]
        st.session_state["section_summaries"] = result["section_summaries"]
        st.session_state["final_summary"] = result["final_summary"]
        
        progress_bar.progress(100)
        status_container.success("处理完成！")
        
        # 显示结果导航
        tab1, tab2, tab3 = st.tabs(["详细摘要", "章节摘要", "最终摘要"])
        
        with tab1:
            for i, summary in enumerate(st.session_state["chunk_summaries"]):
                with st.expander(f"分块 {i+1}", expanded=(i<3)):
                    st.write(summary)
        
        with tab2:
            for i, summary in enumerate(st.session_state["section_summaries"]):
                st.markdown(f"### 章节 {i+1}")
                st.write(summary)
        
        with tab3:
            st.write(st.session_state["final_summary"])
            
    except Exception as e:
        st.error(f"处理失败: {str(e)}")
        logger.error(f"处理错误: {str(e)}", exc_info=True)
    finally:
        time.sleep(2)
        progress_bar.empty()
        status_container.empty()

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
