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
        model = SAC("MlpPolicy", env, verbose=0, device=device)
    except Exception as e:
        print(f"初始化失败: {e}")
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

    # --- 4. 增强型性能评价循环 (本次修改重点) ---
    # 【新增】性能统计变量
    session_rewards = []
    episode_counts = 0
    total_steps = 0
    start_wall_time = time.time()

    try:
        obs, _ = env.reset()
        env.render()
        
        current_ep_reward = 0
        current_ep_steps = 0
        ACTION_SCALE = 0.88
        SMOOTH_FACTOR = 0.7
        prev_action = np.zeros(env.action_space.shape)
        dt = 0.005
        
        print(f"演示开始：已启用性能监控系统。按 Ctrl+C 结束并查看报告。")
        
        while True:
            step_start = time.perf_counter()
            
            # 控制逻辑
            action, _ = model.predict(obs, deterministic=True)
            smoothed_action = SMOOTH_FACTOR * prev_action + (1 - SMOOTH_FACTOR) * (action * ACTION_SCALE)
            prev_action = smoothed_action
            
            obs, reward, terminated, truncated, _ = env.step(np.clip(smoothed_action, -1.0, 1.0))
            
            # 【新增】实时数据累加
            current_ep_reward += reward
            current_ep_steps += 1
            total_steps += 1
            
            try:
                env.render()
            except:
                break
                
            # 时间同步
            elapsed = time.perf_counter() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
            
            # 【新增】回合结束统计
            if terminated or truncated:
                session_rewards.append(current_ep_reward)
                episode_counts += 1
                print(f"回合 {episode_counts} 结束 | 得分: {current_ep_reward:.2f} | 步数: {current_ep_steps}")
                
                # 重置
                obs, _ = env.reset()
                prev_action = np.zeros(env.action_space.shape)
                current_ep_reward = 0
                current_ep_steps = 0
                
    except KeyboardInterrupt:
        print("\n检测到用户中断，正在汇总分析数据...")
    finally:
        # --- 5. 【新增】生成最终评价报告 ---
        duration = time.time() - start_wall_time
        print("\n" + "="*30)
        print("      仿真性能报告")
        print("="*30)
        print(f"总运行时间: {duration:.2f} 秒")
        print(f"物理步总计: {total_steps} 步")
        print(f"完成回合数: {episode_counts} 次")
        if session_rewards:
            print(f"平均每回合得分: {np.mean(session_rewards):.2f}")
            print(f"单回合最高得分: {np.max(session_rewards):.2f}")
        print(f"控制平滑系数: {SMOOTH_FACTOR}")
        print("="*30)

        env.close()
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        print("资源已回收。")

if __name__ == "__main__":
    run_simulation()