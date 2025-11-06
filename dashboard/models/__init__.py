
from typing import Dict, Type
from .base import BaseModel
from .deepseek import DeepseekModel
from .qwen import QwenModel
from .llama import LlamaModel

REGISTRY = {
    "deepseek_v3": DeepseekModel,
    "deepseek": DeepseekModel,
    "qwen": QwenModel,
    "qwen2": QwenModel,
    "llama": LlamaModel,
    "llama2": LlamaModel,
    "llama3": LlamaModel,
}

def build_model(cfg:Dict, **kwargs) -> BaseModel:
    # 如果传入的 cfg 包含 text/vision 或其它嵌套字段，优先把它们解包合并到顶层，方便 from_json 解析
    cfg_in = dict(cfg or {})
    # 合并常见子配置层（不会覆盖已存在的顶层键）
    for key in ("text_config", "vision_config"):
        sub = cfg_in.get(key)
        if isinstance(sub, dict):
            for k, v in sub.items():
                cfg_in.setdefault(k, v)

    mt = cfg_in.get("model_type") or (cfg_in.get("architectures",[None])[0] if cfg_in.get("architectures") else None)
    mt = (mt or "").lower()
    if "deepseek" in mt: key = "deepseek_v3"
    elif "qwen" in mt: key = "qwen"
    elif "llama" in mt: key = "llama"
    else: key = mt
    cls = REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown/unsupported model_type='{mt}'. Supported: {list(REGISTRY.keys())}")
    return cls.from_json(cfg_in, **kwargs)
