#!/usr/bin/env python3
"""
Test script for training plot functionality
学習曲線グラフ化機能のテストスクリプト
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Add scripts directory to path to import our modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from ljspeech_demo import TrainingPlotCallback, TTSModel

def test_training_plot_callback():
    """
    TrainingPlotCallbackの動作をテストする関数
    """
    print("📊 学習曲線グラフ化機能のテストを開始します...")
    
    # テスト用の出力ディレクトリを作成
    test_output_dir = os.path.join(script_dir, "test_outputs")
    os.makedirs(test_output_dir, exist_ok=True)
    
    # TrainingPlotCallbackを初期化
    plot_callback = TrainingPlotCallback(
        model_output_dir=test_output_dir,
        model_type=TTSModel.FASTSPEECH2
    )
    
    print("🧪 模擬学習データを生成中...")
    
    # 模擬学習データを生成（実際の学習カーブに似せる）
    num_epochs = 20
    base_loss = 2.0
    base_mae = 0.5
    
    for epoch in range(num_epochs):
        # 学習が進むにつれて損失とMAEが減少するパターンを模擬
        noise_loss = np.random.normal(0, 0.1)
        noise_mae = np.random.normal(0, 0.02)
        
        # 指数的減衰 + ノイズ
        current_loss = base_loss * np.exp(-0.1 * epoch) + noise_loss
        current_mae = base_mae * np.exp(-0.08 * epoch) + noise_mae
        
        # 負の値にならないようにクリップ
        current_loss = max(0.01, current_loss)
        current_mae = max(0.001, current_mae)
        
        # バリデーション損失（少し高めに設定）
        val_loss = current_loss * 1.2 + np.random.normal(0, 0.05)
        val_mae = current_mae * 1.1 + np.random.normal(0, 0.01)
        val_loss = max(0.01, val_loss)
        val_mae = max(0.001, val_mae)
        
        # ログデータを作成
        logs = {
            'loss': current_loss,
            'mae': current_mae,
            'val_loss': val_loss,
            'val_mae': val_mae
        }
        
        print(f"  エポック {epoch + 1:2d}: Loss={current_loss:.4f}, MAE={current_mae:.4f}")
        
        # コールバックを呼び出し
        plot_callback.on_epoch_end(epoch, logs)
    
    # 学習履歴を保存
    print("\n💾 学習履歴を保存中...")
    plot_callback.save_training_history()
    
    print("\n✅ テスト完了!")
    print(f"📁 生成されたファイルは以下のディレクトリで確認できます:")
    print(f"   {plot_callback.plot_dir}")
    
    # 生成されたファイルをリスト表示
    plot_files = os.listdir(plot_callback.plot_dir)
    print("\n📋 生成されたファイル:")
    for file in sorted(plot_files):
        file_path = os.path.join(plot_callback.plot_dir, file)
        file_size = os.path.getsize(file_path)
        print(f"  • {file} ({file_size:,} bytes)")
    
    return plot_callback.plot_dir

def main():
    """メイン関数"""
    print("=" * 60)
    print("🧪 Training Plot Callback Test")
    print("=" * 60)
    print()
    
    try:
        plot_dir = test_training_plot_callback()
        
        print("\n" + "=" * 60)
        print("📊 グラフ確認方法:")
        print("=" * 60)
        print(f"1. ファイルマネージャーで以下のディレクトリを開く:")
        print(f"   {plot_dir}")
        print()
        print(f"2. 'latest_training_progress.png' を開いて最新のグラフを確認")
        print(f"3. 'training_history.json' でデータの詳細を確認")
        print()
        print("🎯 実際の学習時には、以下のようにコールバックが使用されます:")
        print("   training_plot_callback = TrainingPlotCallback(model_output_dir, model_type)")
        print("   model.fit(dataset, callbacks=[training_plot_callback, ...])")
        
    except Exception as e:
        print(f"❌ テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
