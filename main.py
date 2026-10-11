from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import FastEmbedEmbeddings
import os
import shutil

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

# 向量库存放的文件夹（多份文档会累积在这里）
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

    # 加载文档 + 切片
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

     # 守卫：这份 PDF 里一个字都没读出来（可能是扫描件/图片版）
    if not split_docs:
        raise HTTPException(status_code=400, detail="这份 PDF 里读不到文字，可能是扫描件或图片版，请换一份能选中文字的 PDF")

    # 给每一段贴上「来自哪个文件」的标签，方便以后显示引用来源
    # 这是一个 for 循环：把 split_docs 里的每一段，依次装进 d 这个盒子
    for d in split_docs:
        d.metadata["file_name"] = file.filename

    # 向量库已存在 → 先查重，再追加；不存在 → 新建一个
    if os.path.exists(DB_PATH):
        db = FAISS.load_local(DB_PATH, EMBEDDING, allow_dangerous_deserialization=True)

        # 查重：这个文件是不是已经入库了？
        for d in db.docstore._dict.values():
            if d.metadata.get("file_name") == file.filename:
                os.remove(temp_path)
                raise HTTPException(status_code=400, detail=f"「{file.filename}」已经在知识库里了，不用重复上传")

        db.add_documents(split_docs)
    else:
        db = FAISS.from_documents(split_docs, EMBEDDING)

    db.save_local(DB_PATH)
    os.remove(temp_path)
    return {
        "msg": "文档入库成功",
        "file": file.filename,
        "chunk_num": len(split_docs),
        "total": db.index.ntotal,
    }

# 4. 问答接口：先检索相关原文，再连同问题一起交给大模型
@app.get("/query")
async def query(question: str):

        # 守卫：库里还没有任何文档，就友好提示，而不是崩掉
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=400, detail="知识库还是空的，请先上传一份 PDF")

    # ① 加载向量库
    db = FAISS.load_local(DB_PATH, EMBEDDING, allow_dangerous_deserialization=True)

    # ② 检索：找出和问题最相关的 3 段原文
    docs = db.similarity_search(question, k=3)

    # ③ 把这几段原文拼成一段「参考资料」（又是一个 for 循环）
    #    += 是简写：context = context + 新内容
    context = ""
    for d in docs:
        context += d.page_content + "\n\n"

    # ④ 把「资料 + 问题」一起交给大模型，让它照着资料回答
    prompt = (
        "请只根据下面提供的资料回答问题；资料里没有的信息，就回答「资料中没有提到」。\n\n"
        f"资料：\n{context}\n"
        f"问题：{question}"
    )
    answer = LLM.invoke(prompt).content

    # ⑤ 同时把引用来源整理出来，方便你核对大模型有没有乱说
    sources = []
    for d in docs:
        sources.append({
            "file": d.metadata.get("file_name", "未知文件"),
            "page": d.metadata.get("page", 0) + 1,
            "text": d.page_content[:100],
        })

    return {"answer": answer, "sources": sources}

# 5. 列出知识库里已经有哪些文件
@app.get("/documents")
def list_documents():
    # 库里还是空的
    if not os.path.exists(DB_PATH):
        return {"files": [], "total": 0}

    db = FAISS.load_local(DB_PATH, EMBEDDING, allow_dangerous_deserialization=True)

    # 数一数每个文件各有多少段
    counts = {}
    for d in db.docstore._dict.values():
        name = d.metadata.get("file_name", "未知文件")
        counts[name] = counts.get(name, 0) + 1

    # 整理成一份清单
    files = []
    for name, count in counts.items():
        files.append({"file": name, "chunk_num": count})

    return {"files": files, "total": db.index.ntotal}

# 6. 从知识库里删除某一份文件
@app.delete("/documents")
def delete_document(file: str):
    # 守卫：库是空的
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=400, detail="知识库还是空的，没有可删除的文件")

    db = FAISS.load_local(DB_PATH, EMBEDDING, allow_dangerous_deserialization=True)

    # 把「不属于这个文件」的段挑出来，这些是要保留的
    keep = []
    for d in db.docstore._dict.values():
        if d.metadata.get("file_name") != file:
            keep.append(d)

    # 保留的数量和原来一样多 → 说明一段都没删掉 → 这个文件不在库里
    if len(keep) == db.index.ntotal:
        raise HTTPException(status_code=400, detail=f"知识库里没有「{file}」这个文件")

    # FAISS 不能直接删单个向量 → 干脆删掉旧库，用剩下的段重建
    shutil.rmtree(DB_PATH)
    if len(keep) > 0:
        new_db = FAISS.from_documents(keep, EMBEDDING)
        new_db.save_local(DB_PATH)

    return {"msg": "已删除", "file": file, "left": len(keep)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)