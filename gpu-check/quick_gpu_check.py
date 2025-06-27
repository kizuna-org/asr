#!/usr/bin/env python3
"""
TensorFlowでGPUの検知状況を簡易確認するスクリプト
"""

import tensorflow as tf
import sys

def quick_gpu_check():
    """GPUの利用可能性を簡易確認"""
    print("🔍 TensorFlow GPU 簡易チェック")
    print("-" * 40)
    
    # TensorFlowのバージョン
    print(f"📦 TensorFlow: {tf.__version__}")
    
    # GPU検知
    gpus = tf.config.list_physical_devices('GPU')
    gpu_count = len(gpus)
    
    if gpu_count > 0:
        print(f"✅ GPU検知: {gpu_count}個のGPUが利用可能")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
    else:
        print("❌ GPU検知: GPUが見つかりませんでした")
    
    # CUDA確認
    cuda_available = tf.test.is_built_with_cuda()
    print(f"🔧 CUDA Support: {'✅ 有効' if cuda_available else '❌ 無効'}")
    
    # 簡易計算テスト
    print("🧮 計算テスト:", end=" ")
    try:
        if gpu_count > 0:
            with tf.device('/GPU:0'):
                result = tf.reduce_sum(tf.random.normal([1000, 1000]))
                print("✅ GPU計算成功")
        else:
            result = tf.reduce_sum(tf.random.normal([1000, 1000]))
            print("⚠️ CPU計算のみ")
    except Exception as e:
        print(f"❌ 計算エラー: {e}")
    
    print("-" * 40)
    
    # 終了コード
    if gpu_count > 0 and cuda_available:
        print("🎉 GPU環境は正常に動作しています！")
        return 0
    elif gpu_count > 0:
        print("⚠️ GPUは検知されましたが、CUDA supportに問題があります")
        return 1
    else:
        print("❌ GPU環境に問題があります")
        return 2

if __name__ == "__main__":
    exit_code = quick_gpu_check()
    sys.exit(exit_code) 
