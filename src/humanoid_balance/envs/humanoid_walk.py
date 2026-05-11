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
        env = gym.make("Humanoid-v4", render_mode="human")
        print(f"物理环境启动成功 | 运行设备: {device}")
        
        model = SAC("MlpPolicy", env, verbose=0, device=device)
    except Exception as e:
        print(f"环境初始化失败: {e}")
        return

    # --- 3. 权重提取与对齐 ---
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
        print(f"❌ 加载故障: {e}")
        env.close()
        return

    # --- 4. 优化后的仿真循环 (本次修改重点) ---
    try:
        obs, _ = env.reset()
        env.render() # 预热
        
        # 控制参数
        ACTION_SCALE = 0.88
        SMOOTH_FACTOR = 0.7  # 【新增】平滑因子，0-1之间，值越大动作越平滑
        prev_action = np.zeros(env.action_space.shape)
        
        # 物理步长控制 (200Hz -> 0.005s)
        dt = 0.005
        print(f"演示开始：已启用高精度同步与动作平滑器 (Factor: {SMOOTH_FACTOR})")
        
        while True:
            start_time = time.perf_counter()
            
            # 1. 模型预测
            action, _ = model.predict(obs, deterministic=True)
            
            # 2. 【核心修改】动作平滑滤波 (EMA Filter)
            # 防止关节因为权重不完全匹配而产生剧烈抖动
            current_action = action * ACTION_SCALE
            smoothed_action = SMOOTH_FACTOR * prev_action + (1 - SMOOTH_FACTOR) * current_action
            prev_action = smoothed_action
            
            # 3. 执行动作
            obs, _, terminated, truncated, _ = env.step(np.clip(smoothed_action, -1.0, 1.0))
            
            # 4. 异常安全渲染
            try:
                env.render()
            except:
                break
                
            # 5. 【核心修改】高精度时间补偿
            # 计算代码运行消耗的时间，只 sleep 剩余部分，保证仿真速度恒定
            elapsed = time.perf_counter() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
            
            if terminated or truncated:
                obs, _ = env.reset()
                prev_action = np.zeros(env.action_space.shape) # 重置平滑器
                
    except KeyboardInterrupt:
        print("\n用户手动停止模拟。")
    finally:
        env.close()
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        print("资源已安全回收。")

if __name__ == "__main__":
    run_simulation()