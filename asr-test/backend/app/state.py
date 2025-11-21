# backend/app/state.py
# グローバルな状態管理

# --- グローバルな状態管理 ---
training_status = {
    "is_training": False,
    "current_epoch": 0,
    "current_step": 0,
    "total_epochs": 0,
    "total_steps": 0,
    "current_loss": 0.0,
    "current_learning_rate": 0.0,
    "progress": 0.0,
    "latest_logs": [],
}

# --- モデルのキャッシュ ---
# PoCのため、グローバル変数でモデルを保持
_model_cache = {}
