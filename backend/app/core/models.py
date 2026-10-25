"""项目配置：从 .env 读取参数，提供多服务商模型工厂。

云服务通过OpenAI兼容接口接入；本地对话使用原生ChatOllama，
嵌入仍由独立的OpenAIEmbeddings工厂配置。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

# ===== 对话模型服务商注册表 =====
# key_env 为 None 表示无需密钥（本地 Ollama）
CHAT_PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
        "doc": "platform.deepseek.com",
    },
    "qwen": {  # 阿里云百炼
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
        "key_env": "DASHSCOPE_API_KEY",
        "doc": "bailian.console.aliyun.com",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4-flash",
        "key_env": "ZHIPU_API_KEY",
        "doc": "open.bigmodel.cn",
    },
    "moonshot": {  # Kimi
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-8k",
        "key_env": "MOONSHOT_API_KEY",
        "doc": "platform.moonshot.cn",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
        "doc": "platform.openai.com",
    },
    "siliconflow": {  # 硅基流动聚合平台，可调用多种开源模型
        "base_url": "https://api.siliconflow.cn/v1",
        "default_model": "deepseek-ai/DeepSeek-V3",
        "key_env": "SILICONFLOW_API_KEY",
        "doc": "cloud.siliconflow.cn",
    },
    "ollama": {  # 模型标签由使用者显式配置，不假定已安装某个模型
        "base_url": "http://127.0.0.1:11434/v1",
        "default_model": "",
        "key_env": None,
        "doc": "ollama.com",
    },
}

# ===== 嵌入模型服务商注册表（仅收录提供嵌入接口的服务商）=====
EMBEDDING_PROVIDERS = {
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "default_model": "BAAI/bge-m3",
        "key_env": "SILICONFLOW_API_KEY",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "embedding-3",
        "key_env": "ZHIPU_API_KEY",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "text-embedding-v3",
        "key_env": "DASHSCOPE_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "text-embedding-3-small",
        "key_env": "OPENAI_API_KEY",
    },
    "ollama": {
        "base_url": "http://127.0.0.1:11434/v1",
        "default_model": "",
        "key_env": None,
    },
}

# ===== 重排模型服务商注册表 =====
# 这些服务商都采用同一套重排接口协议（Jina / Cohere 风格），
# 因此一份实现即可通用。自建 vLLM 服务（/v1/rerank）也兼容。
RERANK_PROVIDERS = {
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "default_model": "BAAI/bge-reranker-v2-m3",
        "key_env": "SILICONFLOW_API_KEY",
        "doc": "cloud.siliconflow.cn",
    },
    "jina": {
        "base_url": "https://api.jina.ai/v1",
        "default_model": "jina-reranker-v2-base-multilingual",
        "key_env": "JINA_API_KEY",
        "doc": "jina.ai",
    },
    "cohere": {
        "base_url": "https://api.cohere.com/v2",
        "default_model": "rerank-v3.5",
        "key_env": "COHERE_API_KEY",
        "doc": "dashboard.cohere.com",
    },
}

# 关闭重排的取值（设为 none 时直接返回融合结果）
RERANK_DISABLED = "none"

# ===== 当前选择的服务商与模型（在 .env 中配置）=====
CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "").strip().lower()
CHAT_MODEL = os.getenv("CHAT_MODEL", "").strip()
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "").strip()
RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "none").strip().lower() or "none"
RERANK_MODEL = os.getenv("RERANK_MODEL", "").strip()
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', '').strip() or 'http://127.0.0.1:11434'
OLLAMA_EMBEDDING_BASE_URL = os.getenv('OLLAMA_EMBEDDING_BASE_URL', '').strip() or 'http://127.0.0.1:11434/v1'
# ===== 检索与分块参数 =====
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "4"))            # 最终交给模型的片段数
ANSWER_REVIEW = os.getenv("ANSWER_REVIEW", "false").lower() in {"true", "1", "yes"}
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "true").lower() in {"true", "1", "yes"}
RUN_TIMEOUT = int(os.getenv("RUN_TIMEOUT", "").strip() or ("600" if CHAT_PROVIDER == "ollama" else "180"))


def _get_provider_conf(name: str, registry: dict, kind: str) -> dict:
    """根据名称查找服务商配置，找不到时给出可选列表。"""
    if not name:
        variable = "CHAT_PROVIDER" if kind == "对话模型" else "EMBEDDING_PROVIDER"
        raise RuntimeError(f"{kind}尚未配置，需要在 .env 中配置 {variable}。")
    conf = registry.get(name)
    if conf is None:
        available = "、".join(registry)
        raise ValueError(f"未知的{kind}服务商「{name}」，可选：{available}")
    return conf


def chat_provider_conf() -> dict:
    conf = _get_provider_conf(CHAT_PROVIDER, CHAT_PROVIDERS, "对话模型")
    if CHAT_PROVIDER == 'ollama' and not CHAT_MODEL:
        raise RuntimeError('本地对话模型需要在 .env 中配置 CHAT_MODEL，使用 ollama list 中已安装的完整模型标签。')
    return conf


def embedding_provider_conf() -> dict:
    conf = _get_provider_conf(EMBEDDING_PROVIDER, EMBEDDING_PROVIDERS, "嵌入模型")
    if EMBEDDING_PROVIDER == 'ollama' and not EMBEDDING_MODEL:
        raise RuntimeError('本地嵌入模型需要在 .env 中配置 EMBEDDING_MODEL，使用已安装的 Ollama 嵌入模型标签。')
    return conf


def _get_api_key(conf: dict, provider_name: str, kind: str) -> str:
    """读取服务商密钥；key_env 为 None 的本地服务返回占位符。"""
    if conf["key_env"] is None:
        return "not-needed"  # Ollama 不校验密钥，但不能传空
    api_key = os.getenv(conf["key_env"], "").strip()
    # 空值或仍是模板占位符（如“sk-你的xxx密钥”）都视为未配置
    if not api_key or "你的" in api_key:
        doc = conf.get("doc")
        tip = f"，获取地址：{doc}" if doc else ""
        raise RuntimeError(
            f"{kind}使用 {provider_name} 时，需要在 .env 中配置 {conf['key_env']}{tip}"
        )
    return api_key


def get_llm():
    """返回当前配置的对话模型。"""
    conf = chat_provider_conf()
    if CHAT_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=CHAT_MODEL,
                          base_url=OLLAMA_BASE_URL,
                          temperature=0.4, reasoning=False, num_ctx=16384, num_predict=2400,
                          keep_alive='10m', client_kwargs={'timeout': 180})
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=CHAT_MODEL or conf["default_model"],
        api_key=_get_api_key(conf, CHAT_PROVIDER, "对话模型"),
        base_url=conf["base_url"],
        temperature=0,
        timeout=45,
        max_retries=1,
        max_tokens=1800,
    )


def get_embeddings():
    """返回当前配置的嵌入模型。"""
    conf = embedding_provider_conf()
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL or conf["default_model"],
        api_key=_get_api_key(conf, EMBEDDING_PROVIDER, "嵌入模型"),
        base_url=OLLAMA_EMBEDDING_BASE_URL if EMBEDDING_PROVIDER == 'ollama' else conf["base_url"],
        # 第三方 OpenAI 兼容接口必须关闭，否则会用 tiktoken 报错
        check_embedding_ctx_length=False,
        request_timeout=30,
        max_retries=1,
        chunk_size=32,
    )


def embedding_profile() -> str:
    conf = embedding_provider_conf()
    return f"{EMBEDDING_PROVIDER}:{EMBEDDING_MODEL or conf['default_model']}"


def get_active_models() -> dict:
    """返回当前实际生效的服务商与模型名（用户指定优先，否则用服务商默认）。"""
    chat_conf = CHAT_PROVIDERS.get(CHAT_PROVIDER, {})
    embed_conf = EMBEDDING_PROVIDERS.get(EMBEDDING_PROVIDER, {})
    return {
        "chat_provider": CHAT_PROVIDER,
        "chat_model": CHAT_MODEL or chat_conf.get("default_model", "未知"),
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model": EMBEDDING_MODEL or embed_conf.get("default_model", "未知"),
    }


def missing_api_keys() -> list[str]:
    """启动前检查：返回当前配置下缺失的密钥提示列表。"""
    problems = []
    for kind, name, registry in (
        ("对话模型", CHAT_PROVIDER, CHAT_PROVIDERS),
        ("嵌入模型", EMBEDDING_PROVIDER, EMBEDDING_PROVIDERS),
    ):
        conf = registry.get(name)
        if not conf or not conf["key_env"]:
            continue
        key_value = os.getenv(conf["key_env"], "").strip()
        # 空值或仍是模板占位符都视为未配置
        if not key_value or "你的" in key_value:
            problems.append(
                f"{kind}（{name}）：请在 .env 中填入 {conf['key_env']}"
            )
    return problems
