import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取黑匣子数据
df = pd.read_csv("blackbox.csv")

# 创建画布（2行2列，共4张图）
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Vehicle Blackbox Analysis (Speed / Accel / Control)", fontsize=16)

# ======================
# 图1：速度曲线
# ======================
axes[0,0].plot(df["time"], df["speed"], color="#1f77b4", linewidth=2, label="Speed (km/h)")
axes[0,0].set_title("Vehicle Speed")
axes[0,0].set_ylabel("Speed (km/h)")
axes[0,0].grid(True)
axes[0,0].legend()

# ======================
# 图2：三轴加速度
# ======================
axes[0,1].plot(df["time"], df["accel_x"], color="red", label="Accel X (longitudinal)")
axes[0,1].plot(df["time"], df["accel_y"], color="green", label="Accel Y (lateral)")
axes[0,1].set_title("Acceleration")
axes[0,1].set_ylabel("m/s²")
axes[0,1].grid(True)
axes[0,1].legend()

# ======================
# 图3：转向角
# ======================
axes[1,0].plot(df["time"], df["steer"], color="orange", linewidth=2)
axes[1,0].set_title("Steering Angle")
axes[1,0].set_ylabel("Steer (-1 ~ 1)")
axes[1,0].grid(True)

# ======================
# 图4：油门 + 刹车
# ======================
axes[1,1].plot(df["time"], df["throttle"], color="green", label="Throttle", linewidth=2)
axes[1,1].plot(df["time"], df["brake"], color="red", label="Brake", linewidth=2)
axes[1,1].set_title("Throttle & Brake")
axes[1,1].set_ylabel("Control (0 ~ 1)")
axes[1,1].grid(True)
axes[1,1].legend()

# 统一X轴
for ax in axes.flat:
    ax.set_xlabel("Time (s)")

plt.tight_layout()
plt.show()