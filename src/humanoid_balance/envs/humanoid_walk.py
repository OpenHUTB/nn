import gymnasium as gym
import torch
import time
import numpy as np
from stable_baselines3 import SAC
import mujoco
import zipfile
import shutil
from pathlib import Path

# --- 1. 动态注入兼容性补丁 ---
if not hasattr(mujoco.MjData, 'solver_iter'):
    mujoco.MjData.solver_iter = property(lambda self: self.solver_niter)

def run_simulation(zip_path_str: str = "humanoid_final_walking.zip"):
    zip_path = Path(zip_path_str)
    extract_dir = Path("temp_model_extract")
    
    if not zip_path.exists():
        print(f"致命错误：未找到权重包 {zip_path.name}")
        return

    # --- 2. 环境与模型架构初始化 ---
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 显式指定渲染模式
        env = gym.make("Humanoid-v4", render_mode="human")
        print(f"物理环境启动成功 | 运行设备: {device}")
        
        model = SAC("MlpPolicy", env, verbose=0, device=device)
    except Exception as e:
        print(f"环境初始化失败: {e}")
        return

    # --- 3. 权重动态提取与对齐 ---
    try:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract("policy.pth", extract_dir)
        
        state_dict = torch.load(extract_dir / "policy.pth", map_location=device, weights_only=True)
        model.policy.load_state_dict(state_dict, strict=False)
        print("✅ 权重加载成功")
    except Exception as e:
        print(f"❌ 权重加载故障: {e}")
        env.close()
        return

    # --- 4. 稳健仿真循环 (本次修改重点) ---
    try:
        obs, _ = env.reset()
        
        # 【新增：渲染预热】强制触发 GLFW 初始化，解决“Not Initialized”报错
        print("正在激活渲染上下文...")
        env.render() 
        
        ACTION_SCALE = 0.88 
        print("演示开始：按 Ctrl+C 停止")
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action = np.clip(action * ACTION_SCALE, -1.0, 1.0)
            
            obs, _, terminated, truncated, _ = env.step(action)
            
            # 【优化：异常捕获】防止单帧渲染错误导致整个程序崩溃
            try:
                env.render()
            except Exception as render_err:
                print(f"警告：单帧渲染跳过 ({render_err})")
                break
                
            time.sleep(0.005) 
            if terminated or truncated:
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\n用户手动停止模拟。")
    except Exception as e:
        print(f"运行中发生异常: {e}")
    finally:
        # 【关键：安全释放】确保 GLFW 句柄被正确关闭，释放窗口资源
        print("正在清理系统资源...")
        env.close()
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        print("资源已安全回收。")

if __name__ == "__main__":
    run_simulation()