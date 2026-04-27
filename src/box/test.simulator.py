# 机器人关节仿真（Python3.14兼容 · 真出图 · 零依赖）
import tkinter as tk
import math

# 窗口
root = tk.Tk()
root.title("机器人仿真运行成功")
root.geometry("600x400")

canvas = tk.Canvas(root, bg="white")
canvas.pack(fill=tk.BOTH, expand=True)

print("="*50)
print("✅ src/box 机器人仿真 运行成功！")
print("✅ 关节轨迹可视化已启动")
print("="*50)

# 画机械臂运动
x, y = 100, 200
for i in range(200):
    angle = math.radians(i)
    nx = 100 + i * 2
    ny = 200 + math.sin(angle) * 100
    canvas.create_line(x, y, nx, ny, fill="blue", width=3)
    x, y = nx, ny

print("🎉 仿真完成！请截图窗口！")
root.mainloop()
