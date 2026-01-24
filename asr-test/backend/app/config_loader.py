# backend/app/config_loader.py
import yaml
from typing import Dict, Any

CONFIG_PATH = "config.yaml"

_config = None

def load_config() -> Dict[str, Any]:
    """設定ファイルを読み込み、キャッシュする"""
    global _config
    if _config is None:
        try:
            with open(CONFIG_PATH, 'r') as f:
                _config = yaml.safe_load(f)
        except FileNotFoundError:
            # ここでは空の辞書を返す。呼び出し元で適切に処理する。
            _config = {}
    return _config

def get_model_config(model_name: str) -> Dict[str, Any]:
    """指定されたモデルの設定を取得する"""
    config = load_config()
    return config.get("models", {}).get(model_name)

def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """指定されたデータセットの設定を取得する"""
    config = load_config()
    return config.get("datasets", {}).get(dataset_name)

def get_training_config() -> Dict[str, Any]:
    """グローバルな学習設定を取得する"""
    config = load_config()
    return config.get("training", {})
