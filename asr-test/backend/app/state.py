# backend/app/state.py
# グローバルな状態管理

# --- グローバルな状態管理 ---
training_status = {
    "is_training": False,
}

# --- モデルのキャッシュ ---
# PoCのため、グローバル変数でモデルを保持
_model_cache = {}
