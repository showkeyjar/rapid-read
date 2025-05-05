# PDF快速阅读助手

一个基于大模型的层次化PDF摘要工具，能够自动生成可点击的多级内容提纲。

## 功能特性

- 支持PDF文件上传和内容提取
- 使用大模型生成5个关键点的内容摘要
- 支持最多5层级的摘要深度
- 点击摘要条目可查看更详细的下一级摘要
- 达到第5层级时自动显示PDF原始内容
- 支持多种大模型提供商(OpenAI/DeepSeek/Ollama)

## 技术架构

- 前端：Streamlit (Python)
- 后端：FastAPI (Python)
- PDF处理：PyPDF2
- 大模型支持：
  - OpenAI GPT
  - DeepSeek
  - Ollama本地模型

## 启动方式

只需运行单个命令：
```bash
streamlit run app.py
```

应用将在本地启动并自动打开浏览器窗口。

## 配置说明

在`.env`文件中配置您的API密钥：
```ini
OPENAI_API_KEY=您的OpenAI密钥
DEEPSEEK_API_KEY=您的DeepSeek密钥  # 可选
OLLAMA_MODEL=phi4-reasoning:plus  # 可选
```

## 模型要求

1. 确保已安装并运行Ollama服务
2. 下载所需模型：
```bash
ollama pull phi4-mini-reasoning
```
3. 或者使用其他支持的模型，修改.env文件中的`OLLAMA_MODEL`变量

支持的模型列表：
- phi4-mini-reasoning (推荐)
- mistral
- llama2
- gemma:7b

## 项目结构

```
rapid-read/
├── app.py          # 前端Streamlit应用
├── main.py         # 后端FastAPI服务
├── requirements.txt # Python依赖
├── README.md       # 项目说明
└── .env            # 环境配置
```

## 常见问题

Q: 如何添加自己的Ollama模型？
A: 修改`.env`文件中的`OLLAMA_MODEL`为您本地已下载的模型名称

Q: 摘要层级不够深怎么办？
A: 可以修改`app.py`中的`st.session_state.current_level < 5`条件

Q: 如何调整摘要质量？
A: 可以修改`main.py`中的prompt模板和temperature参数

## 贡献指南

欢迎提交Issue和PR！

## 许可证

MIT License
