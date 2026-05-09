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
# 解决新版 Mujoco (3.x+) 移除 solver_iter 导致的渲染错误
if not hasattr(mujoco.MjData, 'solver_iter'):
    mujoco.MjData.solver_iter = property(lambda self: self.solver_niter)

def run_simulation(zip_path_str: str = "humanoid_final_walking.zip"):
    """
    改进点：
    1. 使用 Pathlib 解决 Windows/Linux 路径斜杠差异
    2. 增加自动清理机制，防止残留 temp 文件
    3. 安全加载模型权重
    """
    # 路径对象化
    zip_path = Path(zip_path_str)
    extract_dir = Path("temp_model_extract")
    
    if not zip_path.exists():
        print(f"致命错误：当前目录下未找到权重压缩包 {zip_path.name}")
        return

    # --- 2. 环境与模型架构初始化 ---
    try:
        # 检测设备加速
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 显式指定渲染后端，避免部分环境崩溃
        env = gym.make("Humanoid-v4", render_mode="human")
        print(f"物理环境启动成功 | 运行设备: {device}")
        
        # 预先构建 SAC 结构，verbose=0 减少无关日志干扰
        model = SAC("MlpPolicy", env, verbose=0, device=device)
    except Exception as e:
        print(f"环境或架构初始化失败: {e}")
        return

    # --- 3. 权重动态提取与对齐 ---
    try:
        # 如果目录已存在则先清理，确保权重最新
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract("policy.pth", extract_dir)
        
        target_pth = extract_dir / "policy.pth"
        
        # weights_only=True 是 PyTorch 推荐的安全加载方式
        state_dict = torch.load(target_pth, map_location=device, weights_only=True)
        
        # 强制兼容加载：strict=False 允许忽略非核心层的键名差异
        model.policy.load_state_dict(state_dict, strict=False)
        print("✅ 权重加载成功：核心控制参数已注入")
        
    except Exception as e:
        print(f"❌ 权重处理故障: {e}")
        env.close()
        return

    # --- 4. 稳健仿真循环 ---
    obs, _ = env.reset()
    # 稍微提高平滑因子，平衡稳定性和响应速度
    ACTION_SCALE = 0.88 
    
    print("开始演示：人形机器人动态平衡控制（按 Ctrl+C 停止）")
    try:
        while True:
            # 开启确定性预测
            action, _ = model.predict(obs, deterministic=True)
            
            # 联合缩放与限幅，抑制高频抖动
            action = np.clip(action * ACTION_SCALE, -1.0, 1.0)
            
            obs, _, terminated, truncated, _ = env.step(action)
            env.render()
            
            # 匹配 200Hz 物理仿真步长
            time.sleep(0.005) 
            
            if terminated or truncated:
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\n模拟已手动停止。")
    finally:
        # --- 5. 资源安全释放与文件清理 ---
        env.close()
        if extract_dir.exists():
            shutil.rmtree(extract_dir) # 彻底删除临时解压文件
        print("环境已关闭，临时资源已清理。")

if __name__ == "__main__":
    run_simulation()
    # ... 前面代码保持不变 ...

    # --- 4. 稳健仿真循环 ---
    try:
        obs, _ = env.reset()
        # 【新增：渲染预热】在循环前先调用一次渲染，强制触发 GLFW 初始化
        env.render() 
        
        ACTION_SCALE = 0.88 
        print("开始演示：渲染上下文已激活...")
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action = np.clip(action * ACTION_SCALE, -1.0, 1.0)
            
            obs, _, terminated, truncated, _ = env.step(action)
            
            # 【优化】捕获单步渲染可能的异常
            try:
                env.render()
            except Exception as render_err:
                print(f"渲染帧跳过: {render_err}")
                break
                
            time.sleep(0.005) 
            if terminated or truncated:
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\n用户手动停止。")
    except Exception as e:
        print(f"运行中发生未知错误: {e}")
    finally:
        # 【关键】无论发生什么，必须确保 env.close() 被执行，释放 GLFW 句柄
        print("正在安全释放渲染资源...")
        env.close()
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
