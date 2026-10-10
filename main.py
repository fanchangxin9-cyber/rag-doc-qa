from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.chains import RetrievalQA
import os

# 密钥单独放在 secret.py 里（该文件已被 .gitignore 排除，不会泄露到 GitHub）
from secret import DEEPSEEK_API_KEY

app = FastAPI(title="RAG文档问答服务")

# 1. 配置大模型（密钥从 secret.py 读取，不要直接写在这里）
os.environ["OPENAI_API_KEY"] = DEEPSEEK_API_KEY
LLM = ChatOpenAI(model="deepseek-chat", openai_api_base="https://api.deepseek.com")
# 向量模型：改用本地免费模型（DeepSeek 没有向量化服务，原来的写法会报 404）
# cache_dir 指定模型存放位置，避免重复下载；HF_ENDPOINT 是国内下载模型的镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
EMBEDDING = FastEmbedEmbeddings(model_name="BAAI/bge-small-zh-v1.5", cache_dir="./fastembed_cache")

# 向量库持久化目录
import shutil

DB_PATH = "./faiss_db"

# 2. 网页界面：有人访问根网址 "/" 时，把 index.html 这个页面返回给浏览器
@app.get("/")
def index():
    return FileResponse("index.html")

# 3. 上传PDF，切片并存入向量库
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # 保存临时文件
    temp_path = f"./temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # 加载文档+文本切片
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    # 构建向量库并本地保存
    # 清理临时目录
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    db = FAISS.from_documents(split_docs, EMBEDDING)
    db.save_local(DB_PATH)
    os.remove(temp_path)
    return {"msg": "文档入库成功", "chunk_num": len(split_docs)}

# 4. 问答接口：检索知识库后交给大模型回答
@app.get("/query")
async def query(question: str):
    # 加载本地向量库
    db = FAISS.load_local(DB_PATH, EMBEDDING, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k":3})
    qa_chain = RetrievalQA.from_chain_type(llm=LLM, chain_type="stuff", retriever=retriever)
    res = qa_chain.run(question)
    return {"answer": res}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)