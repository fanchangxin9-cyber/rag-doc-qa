# Git 速查卡

> 只记一句话：**改文件 → `git add .` → `git commit`。有存档，才敢乱改。**

# 开始之前：一次性准备

`git init` 在`RAG_demo` 里建仓库（多了个隐藏的`.git` 文件夹）
写`.gitignore` 列出「别管这些」：`venv` 、`faiss_db` 、`fastembed_cache` 、`__pycache__` 、`secret.py`
`git branch -m main`# 把当前分支改名为 main

# 日常三件套

```powershell
git status                     # ① 我改了啥？
git add .                      # ② 改动装进购物车
git commit -m "说明这次改了啥"   # ③ 存档

## 改砸了想还原

 ```powershell
git restore 文件名       # 丢掉这个文件还没存档的改动
git restore .           # 丢掉所有未存档改动
git log --oneline       # 查看所有存档（带版本号）
git reset --hard 版本号  # 退回到那一次存档的状态

## 千万别提交这些

`venv/`、`__pycache__/`、密钥文件 —— 已经写在 `.gitignore` 里，Git 会自动跳过。