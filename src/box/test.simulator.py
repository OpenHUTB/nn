import sys
import os
import importlib.util
import numpy as np

# 1. 手动指定 simulator.py 的绝对路径
simulator_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "simulator.py"))

<<<<<<< HEAD

if not os.path.exists(simulator_path):
    raise FileNotFoundError(f"simulator.py 不存在：{simulator_path}")

# 3. 动态加载 simulator.py 模块
spec = importlib.util.spec_from_file_location("simulator", simulator_path)
simulator_module = importlib.util.module_from_spec(spec)
sys.modules["simulator"] = simulator_module
spec.loader.exec_module(simulator_module)

=======
# 2. 检查文件是否存在
if not os.path.exists(simulator_path):
    raise FileNotFoundError(f"simulator.py 不存在：{simulator_path}")

# 3. 动态加载 simulator.py 模块
spec = importlib.util.spec_from_file_location("simulator", simulator_path)
simulator_module = importlib.util.module_from_spec(spec)
sys.modules["simulator"] = simulator_module
spec.loader.exec_module(simulator_module)

>>>>>>> f5c965a634bc42a4261d8907d2ed5530a8647006
# 4. 从加载的模块中导入 Simulator 类
Simulator = simulator_module.Simulator

# 配置仿真器路径（根据你的目录结构）
current_dir = os.path.dirname(os.path.abspath(__file__))
simulator_folder = os.path.join(current_dir, "simulators", "arm_simulation")

# 检查仿真器目录是否存在，如果不存在则创建
if not os.path.exists(simulator_folder):
    print(f"创建仿真器目录：{simulator_folder}")
    os.makedirs(simulator_folder, exist_ok=True)

print("=" * 50)
print("机械臂仿真环境")
print("=" * 50)
print(f"仿真器目录: {simulator_folder}")
print(f"当前工作目录: {os.getcwd()}")

try:
    # 创建仿真环境
    env = Simulator.get(
        simulator_folder=simulator_folder,
        render_mode="human"
    )
    
    print("仿真环境创建成功!")
    print(f"模型关节数(nq): {env.model.nq}")
    print(f"模型执行器数(nu): {env.model.nu}")
    print(f"模型速度数(nv): {env.model.nv}")
    print("=" * 50)
    
    # 重置环境
    obs, info = env.reset(seed=42)
    print(f"初始观测值: {obs}")
    print(f"观测值形状: {obs.shape}")
    print(f"动作空间形状: {env.action_space.shape}")
    print("=" * 50)
    
    print("\n开始仿真...")
    print("按ESC键或关闭窗口可结束仿真")
    print("-" * 50)
    
    # 运行仿真
    for step in range(1000):
        # 随机采样动作
        action = env.action_space.sample()
        
        # 执行一步
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 每50步打印一次信息
        if step % 50 == 0:
            print(f"Step {step:4d} | 奖励: {reward:7.3f} | 终止: {terminated} | 截断: {truncated}")
        
        # 如果环境终止或截断，则重置
        if terminated or truncated:
            print(f"\n环境终止/截断 (Step {step})，重置仿真...")
            obs, info = env.reset()
    
    print("\n仿真完成!")
    
except Exception as e:
    print(f"\n发生错误: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    
finally:
<<<<<<< HEAD
   
=======
    # 关闭环境
>>>>>>> f5c965a634bc42a4261d8907d2ed5530a8647006
    print("\n关闭仿真环境...")
    try:
        if 'env' in locals():
            env.close()
    except:
        pass
    print("仿真结束")