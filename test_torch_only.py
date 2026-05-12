import os
import sys

# Apply the fix here too
try:
    venv_path = os.path.dirname(sys.executable)
    torch_lib_path = os.path.join(venv_path, "Lib", "site-packages", "torch", "lib")
    os.add_dll_directory(torch_lib_path)
except Exception:
    pass

print("Attempting to import torch...")
import torch
print(f"SUCCESS: Torch version {torch.__version__} loaded.")

print("Checking CUDA (GPU) availability...")
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    try:
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU Detected: {device_name}")
        
        # Test a small tensor operation on the GPU
        x = torch.rand(5, 3).cuda()
        print("GPU Tensor Test: Passed (Tensor created on VRAM)")
    except Exception as e:
        print(f"GPU Tensor Test: FAILED ({e})")
else:
    print("WARNING: Python cannot see your RTX 5090. Check drivers.")