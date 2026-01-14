#!/usr/bin/env python3
"""
GPU and CUDA environment check script
"""

import sys
import os
import subprocess

def check_cuda_installation():
    """Check CUDA installation"""
    print("=== CUDA Installation Check ===")
    
    # Check CUDA_HOME
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    print(f"CUDA_HOME: {cuda_home}")
    
    # Check if CUDA directory exists
    if os.path.exists(cuda_home):
        print(f"✅ CUDA directory exists: {cuda_home}")
    else:
        print(f"❌ CUDA directory not found: {cuda_home}")
        return False
    
    # Check CUDA version
    try:
        result = subprocess.run([f'{cuda_home}/bin/nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ CUDA compiler (nvcc) is available")
            print(f"CUDA Version: {result.stdout.split('release ')[1].split(',')[0]}")
        else:
            print("❌ CUDA compiler (nvcc) failed")
            print(f"Error: {result.stderr}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("⚠️  CUDA compiler (nvcc) not found or timeout")
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ nvidia-smi is available")
            print("GPU Information:")
            print(result.stdout)
        else:
            print("❌ nvidia-smi failed")
            print(f"Error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        return False
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi timeout")
        return False
    
    return True

def check_pytorch_cuda():
    """Check PyTorch CUDA support"""
    print("\n=== PyTorch CUDA Check ===")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("✅ CUDA is available in PyTorch")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("❌ CUDA is not available in PyTorch")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking PyTorch CUDA: {e}")
        return False
    
    return True

def check_tensorflow_gpu():
    """Check TensorFlow GPU support"""
    print("\n=== TensorFlow GPU Check ===")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        if tf.config.list_physical_devices('GPU'):
            print("✅ GPU devices available in TensorFlow")
            gpus = tf.config.list_physical_devices('GPU')
            for i, gpu in enumerate(gpus):
                print(f"GPU {i}: {gpu}")
        else:
            print("❌ No GPU devices available in TensorFlow")
            return False
            
    except ImportError:
        print("❌ TensorFlow not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking TensorFlow GPU: {e}")
        return False
    
    return True

def check_environment_variables():
    """Check CUDA-related environment variables"""
    print("\n=== Environment Variables Check ===")
    
    env_vars = [
        'CUDA_HOME',
        'LD_LIBRARY_PATH',
        'CUDA_VISIBLE_DEVICES',
        'TF_FORCE_GPU_ALLOW_GROWTH',
        'TF_GPU_ALLOCATOR'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

def main():
    """Main function"""
    print("🚀 Starting GPU and CUDA environment check...\n")
    
    # Check environment variables
    check_environment_variables()
    
    # Check CUDA installation
    cuda_ok = check_cuda_installation()
    
    # Check PyTorch CUDA
    pytorch_ok = check_pytorch_cuda()
    
    # Check TensorFlow GPU
    tensorflow_ok = check_tensorflow_gpu()
    
    print("\n=== Summary ===")
    print(f"CUDA Installation: {'✅ OK' if cuda_ok else '❌ FAILED'}")
    print(f"PyTorch CUDA: {'✅ OK' if pytorch_ok else '❌ FAILED'}")
    print(f"TensorFlow GPU: {'✅ OK' if tensorflow_ok else '❌ FAILED'}")
    
    if cuda_ok and pytorch_ok:
        print("\n🎉 GPU environment is ready for ASR tasks!")
        return 0
    else:
        print("\n⚠️  GPU environment has issues. Please check the configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
