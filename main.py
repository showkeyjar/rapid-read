from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2
import io
import openai
import ollama
import deepseek
import os
from dotenv import load_dotenv
from enum import Enum

class LLMProvider(str, Enum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummaryRequest(BaseModel):
    text: str
    level: int = 1
    provider: LLMProvider = LLMProvider.OPENAI

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    pdf_file = io.BytesIO(contents)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    return {"text": text}

def generate_openai_summary(text: str, level: int):
    prompt = f"""Generate exactly 5 key points summarizing this text for level {level}:
    {text}
    
    Format as a numbered list with concise points (15 words max each)."""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates hierarchical summaries of documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def generate_deepseek_summary(text: str, level: int):
    prompt = f"""Generate exactly 5 key points summarizing this text for level {level}:
    {text}
    
    Format as a numbered list with concise points (15 words max each)."""
    
    client = deepseek.Client(api_key=DEEPSEEK_API_KEY)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates hierarchical summaries of documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def generate_ollama_summary(text: str, level: int):
    prompt = f"""Generate exactly 5 key points summarizing this text for level {level}:
    {text}
    
    Format as a numbered list with concise points (15 words max each)."""
    
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates hierarchical summaries of documents."},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.3}
    )
    return response['message']['content']

@app.post("/summarize/")
async def generate_summary(request: SummaryRequest):
    if request.level >= 5:  # 达到最大层级时返回原文
        return {"summary": "原始内容:\n\n" + request.text}
        
    if request.provider == LLMProvider.OPENAI:
        summary = generate_openai_summary(request.text, request.level)
    elif request.provider == LLMProvider.DEEPSEEK:
        summary = generate_deepseek_summary(request.text, request.level)
    elif request.provider == LLMProvider.OLLAMA:
        summary = generate_ollama_summary(request.text, request.level)
    
    return {"summary": summary}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
