import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU device name:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
    
    # Test GPU computation
    x = torch.rand(5, 3)
    print("\nTensor on CPU:", x)
    x = x.cuda()
    print("Tensor on GPU:", x)
    print("Device of tensor:", x.device) 