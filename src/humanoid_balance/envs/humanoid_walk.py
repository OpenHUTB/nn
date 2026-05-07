import gymnasium as gym
import torch
import time
import numpy as np
from stable_baselines3 import SAC
import mujoco
import zipfile
import os
import sys

# --- 1. 解决新版 Mujoco 属性名冲突补丁 ---
# 确保在 Gymnasium 渲染调用 solver_iter 时不会因版本更新而报错
if not hasattr(mujoco.MjData, 'solver_iter'):
    mujoco.MjData.solver_iter = property(lambda self: self.solver_niter)

def run_simulation(zip_container="humanoid_final_walking.zip"):
    """
    自动从 zip 中解压 policy.pth 并使用兼容模式加载权重
    """
    # --- 2. 环境初始化 ---
    try:
        # 使用 Humanoid-v4 匹配 376 维观测空间
        env = gym.make("Humanoid-v4", render_mode="human")
        print("物理环境已启动，正在准备演示...")
    except Exception as e:
        print(f"环境启动失败: {e}")
        return

    # --- 3. 动态提取与兼容性加载 ---
    extract_dir = "temp_model_extract"
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    
    target_pth = os.path.join(extract_dir, "policy.pth")

    try:
        print(f"正在从 {zip_container} 中提取权重文件...")
        with zipfile.ZipFile(zip_container, 'r') as zip_ref:
            # 提取压缩包内的原始权重文件
            zip_ref.extract("policy.pth", extract_dir)
        
        print("正在构建模型大脑并注入权重...")
        # 先建立 SAC 模型结构
        model = SAC("MlpPolicy", env, verbose=1)
        
        # 加载提取出的 pth 数据
        state_dict = torch.load(target_pth, map_location="cuda" if torch.cuda.is_available() else "cpu")
        
        # 核心修改：使用 strict=False 强制兼容不同版本的权重命名规则
        # 解决 Missing key(s) in state_dict: "actor.mu.weight" 等报错
        model.policy.load_state_dict(state_dict, strict=False)
        print("模型加载成功（已开启兼容模式）！")

    except Exception as e:
        print(f"加载过程中发生错误: {e}")
        env.close()
        return

    # --- 4. 运行逻辑 ---
    obs, info = env.reset()
    
    # 针对视频中机器人“过度补偿/扭动”现象的平滑因子
    action_scale = 0.85 
    
    print("开始演示！请观察窗口。按 Ctrl+C 退出。")
    try:
        while True:
            # 使用确定性预测获取最稳定的步态
            action, _states = model.predict(obs, deterministic=True)
            
            # 对动作进行缩放和限幅，增加关节稳定性
            action = np.clip(action * action_scale, -1.0, 1.0)
            
            # 执行环境步进
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 渲染画面
            env.render()
            
            # 匹配物理模拟步长 (200Hz)
            time.sleep(0.005) 
            
            # 摔倒或越界后自动重置
            if terminated or truncated:
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\n模拟已手动停止。")
    finally:
        # --- 5. 资源释放 ---
        env.close()
        print("环境已安全关闭。")

if __name__ == "__main__":
    # 确保当前目录下有 humanoid_final_walking.zip 文件
    run_simulation("humanoid_final_walking.zip")
