"""
Script to verify CUDA availability for PyTorch.
Run with: python scripts/verify_cuda.py
"""
import torch
import sys

def verify_cuda():
    print("=" * 60)
    print("PyTorch CUDA Verification")
    print("=" * 60)

    # Basic info
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")

    # CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        
        # Test tensor creation on GPU
        try:
            x = torch.randn(5, 3).cuda()
            print(f"\n✓ Successfully created tensor on GPU")
            print(f"  Tensor device: {x.device}")
            
            # Test computation
            y = torch.randn(3, 4).cuda()
            z = torch.mm(x, y)
            print(f"  ✓ Successfully performed computation on GPU")
            
        except Exception as e:
            print(f"\n✗ Failed to use GPU: {e}")
            return False
        
        return True
    else:
        print("\n⚠ CUDA is not available")
        print("  This could mean:")
        print("  - PyTorch was installed without CUDA support (CPU-only version)")
        print("  - No NVIDIA GPU is available")
        print("  - CUDA drivers are not installed")
        print("\n  To install PyTorch with CUDA support:")
        print("  pip install -r requirements-cuda.txt")
        return False

if __name__ == "__main__":
    success = verify_cuda()
    print("\n" + "=" * 60)
    sys.exit(0 if success else 1)

