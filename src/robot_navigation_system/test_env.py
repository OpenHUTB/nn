import sys
print(f"Python版本: {sys.version}")

try:
    import numpy as np
    print(f"NumPy版本: {np.__version__}")
except ImportError as e:
    print(f"NumPy导入失败: {e}")

try:
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"PyTorch导入失败: {e}")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print(f"Matplotlib版本: {matplotlib.__version__}")
except ImportError as e:
    print(f"Matplotlib导入失败: {e}")

print("\n环境检查完成！")