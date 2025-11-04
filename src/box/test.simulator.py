from uitb.simulator import Simulator

# 加载仿真器（需替换为实际仿真器路径，如 "uitb/simulators/arm_simulation"）
simulator_folder = "uitb/simulators/your_sim_folder"  
env = Simulator.get(
    simulator_folder=simulator_folder,
    render_mode="human",  # "human" 弹出Pygame窗口；"rgb_array" 输出图片数组
    render_mode_perception="embed"  # 感知图像嵌入主窗口
)

# 重置环境
obs, info = env.reset(seed=42)
print("初始观测值：", {k: v.shape for k in obs})

# 运行 1000 步仿真（随机动作示例）
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if step % 100 == 0:
        print(f"第 {step} 步 | 奖励：{reward:.2f} | 终止状态：{terminated}")
    if terminated or truncated:
        obs, info = env.reset()

# 关闭仿真（释放资源）
env.close()
