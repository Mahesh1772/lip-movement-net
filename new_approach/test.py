import sys
import tensorflow as tf
import torch
import os
import subprocess
import platform

def get_cuda_version():
    try:
        if platform.system() == 'Windows':
            nvcc_output = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
            return nvcc_output
    except:
        return "CUDA not found in PATH"

def main():
    print("\n=== System Information ===")
    print(f"Python version: {sys.version}")
    
    print("\n=== CUDA Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    
    print("\n=== TensorFlow Information ===")
    print(f"TensorFlow version: {tf.__version__}")
    print("\nAvailable TF devices:")
    for device in tf.config.list_physical_devices():
        print(device)
    
    print("\n=== NVIDIA CUDA Toolkit ===")
    print(get_cuda_version())
    
    print("\n=== Environment Variables ===")
    print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")

if __name__ == "__main__":
    main()