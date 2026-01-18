import torch
import torchvision
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

from utils.registry import MODELS

# 模型到模块的映射表
MODEL_MODULE_MAP = {
    'PoundNet': 'networks.poundnet_detector',
}

def _normalize_model_name(model_name: str) -> str:
    return model_name.strip().lower()


def load_model_module(model_name: str) -> Optional[Any]:
    module_path = MODEL_MODULE_MAP.get(model_name)
    if module_path is None:
        normalized = _normalize_model_name(model_name)
        for key, value in MODEL_MODULE_MAP.items():
            if _normalize_model_name(key) == normalized:
                module_path = value
                break
    if module_path is None:
        raise ValueError(
            f"Unknown model arch '{model_name}'. Available: {sorted(MODEL_MODULE_MAP.keys())}"
        )
    module = importlib.import_module(module_path)
    return module

def get_model(conf):
    print("Loading model...")
    model_name = conf.arch
    load_model_module(model_name)

    if model_name in MODELS:
        if hasattr(conf, 'model'):
            kwargs = conf.model
        else:
            kwargs = {}
        return MODELS.build(model_name, **kwargs)
