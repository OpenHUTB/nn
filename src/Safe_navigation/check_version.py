import airsim
import pkg_resources
import inspect

print("=== AirSim 环境诊断 ===")

# 1. 检查安装的版本
try:
    version = pkg_resources.get_distribution("airsim").version
    print(f"1. 已安装的 airsim 包版本: {version}")
except Exception as e:
    print(f"1. 无法获取版本: {e}")

# 2. 查看 CarClient 的构造函数签名
print("\n2. CarClient.__init__ 接受的参数:")
try:
    # 获取构造函数签名，跳过第一个参数'self'
    sig = inspect.signature(airsim.CarClient.__init__)
    params = list(sig.parameters.keys())[1:]  # 移除'self'
    print("   ", params)
except Exception as e:
    print(f"   获取失败: {e}")

# 3. 尝试导入其他可能相关的模块
print("\n3. 尝试直接导入常用客户端...")
try:
    from airsim import CarClient
    print("    ✓ 成功从 airsim 导入 CarClient")
except ImportError as e:
    print(f"    ✗ 导入失败: {e}")