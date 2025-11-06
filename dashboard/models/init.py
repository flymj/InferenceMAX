# models/__init__.py
from .deepseek import DeepseekModel
from .qwen import QwenModel
from .llama import LlamaModel

REGISTRY = {
    "deepseek_v3": DeepseekModel,
    "deepseek": DeepseekModel,
    "qwen": QwenModel,
    "qwen3": QwenModel,
    "llama": LlamaModel,
}

def _detect_key(cfg: dict) -> str:
    key = (cfg.get("model_type") or "").lower()
    if not key and cfg.get("architectures"):
        arch = (cfg["architectures"][0] or "").lower()
        if "deepseek" in arch: return "deepseek_v3"
        if "qwen" in arch:     return "qwen"
        if "llama" in arch:    return "llama"
    return key or "qwen"

def build_model(cfg: dict):
    cfg_clean = dict(cfg or {})
    key = _detect_key(cfg_clean)
    cls = REGISTRY.get(key, QwenModel)
    # 所有子类都有 @classmethod from_json
    return cls.from_json(cfg_clean)

