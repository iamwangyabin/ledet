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

def load_model_module(model_name: str) -> Optional[Any]:
    module_path = MODEL_MODULE_MAP.get(model_name)
    module = importlib.import_module(module_path)
    return module

def get_model(conf):
    print("Loading model...")
    model_name = conf.arch
    load_model_module(model_name)
    return MODELS.build(model_name, cfg=conf)
