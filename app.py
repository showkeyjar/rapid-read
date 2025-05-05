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
MAX_WORKERS = 12  # 根据CPU核心数调整
OLLAMA_TIMEOUT = 180  # 3分钟超时
OLLAMA_RETRY_DELAY = 10  # 重试间隔(秒)

def process_pdf(file) -> dict:
    """处理PDF文件并返回文本内容"""
    try:
        # 读取文件内容
        contents = file.getvalue()
        
        # 解析PDF
        pdf_reader = PdfReader(BytesIO(contents))
        page_count = len(pdf_reader.pages)
        logger.info(f"Processing PDF with {page_count} pages")
        
        # 提取文本
        text_pages = []
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text() or ""
            text_pages.append(text)
            
        return {
            "text": "\n".join(text_pages),
            "pages": page_count
        }
        
    except Exception as e:
        logger.error(f"PDF处理失败: {str(e)}")
        raise

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
        """先检查服务健康再加载模型"""
        if not check_ollama_health():
            raise RuntimeError("Ollama服务响应缓慢或不可用")
        
        if self._model_loaded:
            return True
            
        with self._lock:
            if not self._model_loaded:
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

def check_ollama_health() -> bool:
    """检查Ollama服务健康状态"""
    try:
        start_time = time.time()
        resp = requests.get(
            f"{os.getenv('OLLAMA_API_BASE')}/api/tags",
            timeout=10
        )
        latency = time.time() - start_time
        
        if resp.status_code == 200:
            logger.info(f"Ollama服务正常 (响应时间: {latency:.2f}s)")
            return latency < 5  # 响应时间超过5秒视为警告
        else:
            logger.warning(f"Ollama服务异常 (HTTP {resp.status_code})")
            return False
    except Exception as e:
        logger.error(f"Ollama健康检查失败: {str(e)}")
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

def process_chunk_parallel(chunk: str, chunk_id: int, provider: str) -> tuple[int, str]:
    """强制中文输出的分块处理，英文自动重试"""
    for attempt in range(3):
        try:
            model_manager.ensure_model_loaded(provider)
            prompt = (
                "你是一名中文专业文献总结助手。"
                "请用简体中文、结构化、条理清晰地总结以下内容（限300字）：\n\n"
                f"{chunk}"
            )
            response = requests.post(
                f"{os.getenv('OLLAMA_API_BASE')}/api/generate",
                json={
                    "model": os.getenv("OLLAMA_MODEL"),
                    "prompt": prompt,
                    "options": {
                        "temperature": 0.2,
                        "num_ctx": 2048
                    }
                },
                timeout=OLLAMA_TIMEOUT
            )
            result = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    result += data.get("response", "")
                    if data.get("done", False):
                        break
            # 检查是否为英文，若是则报错或重试
            if re.search(r'[a-zA-Z]{8,}', result) and not re.search(r'[\u4e00-\u9fa5]', result):
                raise ValueError("模型返回英文摘要，重试")
            return (chunk_id, result.strip() or "[无有效摘要]")
        except Exception as e:
            logger.warning(f"分块 {chunk_id} 处理失败: {str(e)}")
            if attempt == 2:
                return (chunk_id, f"[处理失败: {str(e)}]")
            time.sleep(OLLAMA_RETRY_DELAY * (attempt+1))

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
    """完整修复版分层摘要"""
    result = {
        'chunks': [],       # 原始分块
        'summaries': [],    # 分块摘要
        'sections': [],     # 可下钻章节
        'final_summary': ""
    }
    
    try:
        # 1. 分块处理
        chunks = split_into_chunks(text)
        result['chunks'] = chunks
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_chunk_parallel, chunk, i, provider) 
                      for i, chunk in enumerate(chunks)]
            
            for future in concurrent.futures.as_completed(futures):
                chunk_id, summary = future.result()
                result['summaries'].append((chunk_id, summary))
        
        # 2. 构建可下钻章节
        result['summaries'].sort(key=lambda x: x[0])
        summaries = [s for _, s in result['summaries'] if not s.startswith('[')]
        
        SECTION_SIZE = 3  # 每节包含的分块数
        for section_idx in range(0, len(summaries), SECTION_SIZE):
            section_chunks = summaries[section_idx:section_idx+SECTION_SIZE]
            section_content = "\n\n".join([
                f"### 分块 {section_idx+i+1}\n{chunk}" 
                for i, chunk in enumerate(section_chunks)
            ])
            
            result['sections'].append({
                "title": f"第{section_idx//SECTION_SIZE +1}节",
                "content": section_content,
                "chunk_ids": list(range(section_idx, section_idx+len(section_chunks)))
            })
        
        # 3. 生成最终摘要
        if result['sections']:
            section_texts = [f"{s['title']}: {s['content'][:200]}..." for s in result['sections']]
            result['final_summary'] = generate_final_summary(section_texts)
            
    except Exception as e:
        logger.error(f"处理失败: {str(e)}", exc_info=True)
        result['error'] = str(e)
    
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

def display_drilldown_summary(result):
    # 初始化状态
    if "current_level" not in st.session_state:
        st.session_state.current_level = 1
    if "selected_section" not in st.session_state:
        st.session_state.selected_section = 0
    if "selected_chunk" not in st.session_state:
        st.session_state.selected_chunk = 0

    # 1级：首页/总览
    if st.session_state.current_level == 1:
        st.markdown("## 最终摘要")
        st.markdown(result['final_summary'])
        st.markdown("## 章节摘要")
        for idx, section in enumerate(result['sections']):
            st.markdown(f"**{section['title']}**\n\n{section['content']}")
            if st.button(f"下钻到{section['title']}", key=f"drill_section_{idx}"):
                st.session_state.selected_section = idx
                st.session_state.current_level = 2
                st.rerun()

    # 2级：章节下分块摘要
    elif st.session_state.current_level == 2:
        section = result['sections'][st.session_state.selected_section]
        st.markdown(f"### {section['title']} 摘要")
        st.markdown(section['content'])
        st.markdown("---")
        st.markdown("#### 分块摘要")
        for i, idx in enumerate(section['chunk_ids']):
            chunk_summary = next((s for j, s in result['summaries'] if j == idx), None)
            if chunk_summary:
                st.markdown(f"**分块 {idx+1}**\n{chunk_summary}")
                if st.button(f"下钻到分块 {idx+1}", key=f"drill_chunk_{idx}"):
                    st.session_state.selected_chunk = idx
                    st.session_state.current_level = 3
                    st.rerun()
        if st.button("返回章节总览"):
            st.session_state.current_level = 1
            st.rerun()

    # 3级：分块原文
    elif st.session_state.current_level == 3:
        chunk_id = st.session_state.selected_chunk
        st.markdown(f"### 分块 {chunk_id+1} 原文")
        st.text(result['chunks'][chunk_id][:2000])
        if st.button("返回上一层（章节）"):
            st.session_state.current_level = 2
            st.rerun()

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
        st.session_state["chunk_summaries"] = result["summaries"]
        st.session_state["section_summaries"] = result["sections"]
        st.session_state["final_summary"] = result["final_summary"]
        
        progress_bar.progress(100)
        status_container.success("处理完成！")
        
        # 显示结果导航
        display_drilldown_summary(result)
        
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
        st.rerun()
with col2:
    if st.button("重置到第一层"):
        st.session_state.current_level = 1
        st.rerun()
