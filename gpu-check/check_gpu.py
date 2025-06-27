#!/usr/bin/env python3
"""
TensorFlowでGPUの検知状況を確認するスクリプト
"""

import tensorflow as tf
import os

def check_gpu_availability():
    """GPUの利用可能性を詳細に確認"""
    print("=" * 60)
    print("TensorFlow GPU 検知状況")
    print("=" * 60)
    
    # TensorFlowのバージョン確認
    print(f"TensorFlow Version: {tf.__version__}")
    print()
    
    # GPU利用可能性の基本チェック
    print("1. GPU利用可能性の基本チェック:")
    print(f"   GPU available: {tf.config.list_physical_devices('GPU')}")
    print(f"   Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"   GPU support: {tf.test.is_gpu_available()}")
    print()
    
    # 物理デバイスの詳細確認
    print("2. 物理デバイス一覧:")
    physical_devices = tf.config.list_physical_devices()
    for device in physical_devices:
        print(f"   {device}")
    print()
    
    # GPU詳細情報
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print("3. GPU詳細情報:")
        for i, gpu in enumerate(gpu_devices):
            print(f"   GPU {i}: {gpu}")
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                for key, value in gpu_details.items():
                    print(f"     {key}: {value}")
            except Exception as e:
                print(f"     詳細情報取得エラー: {e}")
        print()
    else:
        print("3. GPU詳細情報:")
        print("   GPUが検知されませんでした")
        print()
    
    # 論理デバイス確認
    print("4. 論理デバイス:")
    logical_devices = tf.config.list_logical_devices()
    for device in logical_devices:
        print(f"   {device}")
    print()
    
    # CUDAとcuDNNのバージョン確認
    print("5. CUDA/cuDNN情報:")
    try:
        # CUDA version
        print(f"   CUDA Version: {tf.sysconfig.get_build_info()['cuda_version']}")
        print(f"   cuDNN Version: {tf.sysconfig.get_build_info()['cudnn_version']}")
    except Exception as e:
        print(f"   CUDA/cuDNN情報取得エラー: {e}")
    print()
    
    # 環境変数確認
    print("6. 関連環境変数:")
    cuda_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'LD_LIBRARY_PATH']
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")
    print()
    
    # 簡単なGPU計算テスト
    print("7. GPU計算テスト:")
    try:
        with tf.device('/GPU:0'):
            # 簡単な行列乗算テスト
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"   GPU計算結果: \n{c.numpy()}")
            print("   ✅ GPU計算が正常に実行されました")
    except Exception as e:
        print(f"   ❌ GPU計算エラー: {e}")
        try:
            with tf.device('/CPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print(f"   CPU計算結果: \n{c.numpy()}")
                print("   ⚠️ CPUでの計算は正常に実行されました")
        except Exception as cpu_e:
            print(f"   ❌ CPU計算もエラー: {cpu_e}")
    print()
    
    # メモリ情報（GPUがある場合）
    if gpu_devices:
        print("8. GPU メモリ情報:")
        try:
            for i, gpu in enumerate(gpu_devices):
                memory_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                print(f"   GPU {i} メモリ:")
                print(f"     Current: {memory_info['current'] / 1024**2:.2f} MB")
                print(f"     Peak: {memory_info['peak'] / 1024**2:.2f} MB")
        except Exception as e:
            print(f"   メモリ情報取得エラー: {e}")
    
    print("=" * 60)

if __name__ == "__main__":
    check_gpu_availability() 
