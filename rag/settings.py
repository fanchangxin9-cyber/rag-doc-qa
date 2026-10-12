"""
settings.py —— 运行时设置（工具模块）

职责：记住「每家服务商各自的地址/模型/密钥」，以及「现在用哪一家」。

settings.json 的结构：
{
  "current": "zhipu",                       ← 现在用哪家
  "providers": {
      "zhipu":    {"base_url": "…", "model": "…", "api_key": "…"},
      "deepseek": {"base_url": "…", "model": "…", "api_key": "…"},
      "ollama":   {"base_url": "…", "model": "…", "api_key": ""},
      "custom":   {"base_url": "…", "model": "…", "api_key": "…"}
  }
}

**每家各存一份**，这是关键 —— 切过去时密钥自动带出，不用重填。
（之前只存得下一套，切到别家就会拿着上一家的密钥去认证，直接 401。）

⚠️ 安全红线（两条，都要守住）：
   ① settings.json 里存着 API 密钥 → 它必须写进 .gitignore，绝不能提交
   ② 对外只提供 public_view()（脱敏版）→ 密钥永远不回传给浏览器
"""

import json
import os

import config


def _empty_store() -> dict:
    """全新的空设置（第一次运行时用）：默认配置 DeepSeek，密钥来自 secret.py"""
    return {
        "current": config.DEFAULT_PROVIDER,
        "providers": {
            config.DEFAULT_PROVIDER: {
                "base_url": config.LLM_BASE_URL,
                "model": config.LLM_MODEL,
                "api_key": config.LLM_API_KEY,
            }
        },
    }


def _read() -> dict:
    """把整个设置文件读出来。

    文件不存在、内容坏了、或者是旧版结构（一层扁平的），
    一律退回默认 —— 不让一个坏掉的配置文件把整个服务弄得起不来。
    """
    if not os.path.exists(config.SETTINGS_PATH):
        return _empty_store()

    try:
        with open(config.SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[警告] settings.json 读不出来，先用默认设置：{e}")
        return _empty_store()

    if not isinstance(data, dict) or "providers" not in data:
        print("[警告] settings.json 还是旧版结构，已按默认设置启动（下次保存会自动升级）")
        return _empty_store()

    return data


def current_id() -> str:
    """现在用的是哪一家"""
    return _read().get("current") or config.DEFAULT_PROVIDER


def get(provider_id: str = None) -> dict:
    """取某一家存过的配置；不传参数就是"当前这家"。

    这家从来没配过，就返回 None，由调用方决定怎么提示。
    """
    store = _read()
    pid = provider_id or store.get("current") or config.DEFAULT_PROVIDER
    return store["providers"].get(pid)


def save(provider_id: str, base_url: str, model: str, api_key: str) -> None:
    """把某一家的配置存下来，并把它设为"当前这家"。

    约定：api_key 传空字符串 = "沿用这家原来那把"。
    所以只想改模型名的时候，不用重新粘贴密钥。
    """
    store = _read()
    old = store["providers"].get(provider_id) or {}

    store["providers"][provider_id] = {
        "base_url": base_url,
        "model": model,
        "api_key": api_key or old.get("api_key", ""),
    }
    store["current"] = provider_id

    with open(config.SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def public_view() -> dict:
    """给前端看的**脱敏版**。

    每一家只报"密钥有没有填"，**绝不报密钥真值**。
    浏览器拿不到真值，也就没法被人从页面里偷走。
    """
    store = _read()
    providers = {}
    for pid, cfg in store["providers"].items():
        providers[pid] = {
            "base_url": cfg.get("base_url", ""),
            "model": cfg.get("model", ""),
            "has_key": bool(cfg.get("api_key")),
        }
    return {"current": store.get("current"), "providers": providers}
