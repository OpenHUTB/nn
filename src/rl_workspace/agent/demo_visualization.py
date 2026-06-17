import numpy as np
import matplotlib.pyplot as plt
from src.rl.visualizer import LivePlot, PolicyHeatmap

# 测试实时曲线
print("测试 LivePlot...")
live = LivePlot(xlabel='Episode', ylabel='Reward', title='Demo Training Curve')
for episode in range(100):
    # 模拟奖励（噪声 + 上升趋势）
    reward = 50 + episode * 0.5 + np.random.randn() * 5
    live.update('reward', episode, reward)
live.save("demo_live_plot.png")
print("实时曲线已保存为 demo_live_plot.png")

# 测试策略热力图
print("测试 PolicyHeatmap...")
heatmap = PolicyHeatmap(num_actions=4, env_name='DemoEnv')
for step in range(0, 101, 10):
    # 模拟策略概率（逐渐趋于某个动作）
    probs = np.exp([step/30, 0, 0, 0])
    probs = probs / probs.sum()
    heatmap.update(probs, step)
heatmap.save("demo_heatmap.png")
print("热力图已保存为 demo_heatmap.png")

print("演示完成！请关闭图形窗口。")
plt.show(block=True)