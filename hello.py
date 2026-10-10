# ============================================
# 我的第一个网页服务（学习用，只有 4 步）
# 文件名：hello.py
# 启动命令：uvicorn hello:app --reload
# ============================================

# ① 借工具：从 fastapi 这个“零件盒”里，拿出叫 FastAPI 的零件
from fastapi import FastAPI

# ② 创建服务本体，并给它起名叫 app（uvicorn hello:app 里的 app 就是它）
app = FastAPI()

# ③ 挂一个接口：当有人访问根网址 “/” 时，就执行下面的 home 函数
@app.get("/")
def home():
    # ④ 把一段数据返回给浏览器（字典 / 花括号写法，{键: 值} 的形式）
    return {"message": "从现在开始，我将彻底开始自学Python和JAVA"}