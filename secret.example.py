"""
secret.example.py —— 密钥文件模板（这是给人看的样例，不是真文件）

怎么用：
  1. 复制一份，改名成 secret.py
  2. 把下面的值换成你自己的 API Key

其实现在不建它也能用：
  直接在网页上的「模型设置」里填密钥、点「保存」即可，
  密钥会存进 settings.json —— 那个文件同样已被 .gitignore 排除。

留着这个模板，只是给"想在配置文件里预设密钥"的人一个示例。
"""

# 大模型 API Key。
# 各家格式不同，比如智谱的是「32 位十六进制 + 一个点 + 16 位字符」：
#     xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.yyyyyyyyyyyyyyyy
# 复制时务必整段复制（中间那个点别漏），只复制前半截会报 401。
DEEPSEEK_API_KEY = "在这里填你自己的密钥"