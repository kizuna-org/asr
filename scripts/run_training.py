#!/usr/bin/env python3
"""
FastSpeech 2 Training Launcher Script
This script provides easy commands to run FastSpeech 2 training with different configurations.
"""

import os
import sys
import subprocess

def print_usage():
    """Print usage information."""
    print("=" * 60)
    print("🚀 FastSpeech 2 Training Launcher")
    print("=" * 60)
    print()
    print("📖 使用方法:")
    print("  python run_training.py [mode] [epochs]")
    print()
    print("📊 モード:")
    print("  mini  - 最初の10サンプルで学習 (デフォルト)")
    print("  full  - フルデータセットで学習")
    print("  test  - モデルのテストのみ実行")
    print()
    print("🎯 例:")
    print("  python run_training.py mini         # ミニモードで2000エポック学習")
    print("  python run_training.py mini 100     # ミニモードで100エポック学習")
    print("  python run_training.py full         # フルモードで2000エポック学習")
    print("  python run_training.py full 5000    # フルモードで5000エポック学習")
    print("  python run_training.py test         # モデルテストのみ")
    print()
    print("📝 詳細:")
    print("  • miniモード: 最初の10サンプルを使用")
    print("    - 音声生成は最初のサンプルのテキストと同じ")
    print("    - 開発・デバッグ用")
    print("  • fullモード: 全LJSpeechデータセットを使用")
    print("    - 本格的なトレーニング用")
    print("  • testモード: モデルの動作確認のみ")
    print("  • epochs: エポック数 (デフォルト: 2000)")
    print()

def run_command(cmd, description):
    """Run a command with description."""
    print(f"🔧 {description}")
    print(f"💻 実行コマンド: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} 完了")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失敗: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {description} が中断されました")
        return False

def main():
    """Main function."""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "help"
    
    # Get epochs if provided
    epochs = 2000  # default
    if len(sys.argv) > 2:
        try:
            epochs = int(sys.argv[2])
        except ValueError:
            print(f"❌ エラー: エポック数は整数で指定してください: {sys.argv[2]}")
            return
    
    # Change to scripts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    if mode in ["help", "-h", "--help"]:
        print_usage()
        return
    
    # Activate virtual environment path
    venv_python = "../.venv/bin/python"
    
    if mode == "mini":
        print("🎯 ミニモードでFastSpeech 2トレーニングを開始します")
        print("📊 最初の10サンプルを使用")
        print(f"🔄 エポック数: {epochs}")
        print("🎤 音声生成: 最初のサンプルのテキストと同じ")
        print()
        
        cmd = [venv_python, "ljspeech_demo.py", "--mode", "mini", "--epochs", str(epochs)]
        run_command(cmd, f"ミニモードトレーニング ({epochs}エポック)")
        
    elif mode == "full":
        print("🎯 フルモードでFastSpeech 2トレーニングを開始します")
        print("📊 全LJSpeechデータセットを使用")
        print(f"🔄 エポック数: {epochs}")
        print("⚠️  注意: 長時間の処理になります")
        print()
        
        response = input("続行しますか？ (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("❌ トレーニングをキャンセルしました")
            return
        
        cmd = [venv_python, "ljspeech_demo.py", "--mode", "full", "--epochs", str(epochs)]
        run_command(cmd, f"フルモードトレーニング ({epochs}エポック)")
        
    elif mode == "test":
        print("🧪 FastSpeech 2モデルのテストを実行します")
        print()
        
        cmd = [venv_python, "test_fastspeech2.py"]
        run_command(cmd, "モデルテスト")
        
    else:
        print(f"❌ 不明なモード: {mode}")
        print()
        print_usage()
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  プログラムが中断されました")
        sys.exit(1) 
