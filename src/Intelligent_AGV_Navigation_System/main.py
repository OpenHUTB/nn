import tkinter as tk
import math
import random

# =============== 【主窗口】 ===============
root = tk.Tk()
root.title("3D智能无人小车避障导航系统")
root.geometry("1100x800")
canvas = tk.Canvas(root, bg="#000000")
canvas.pack(fill=tk.BOTH, expand=True)

# =============== 【全局参数】 ===============
car_x, car_y = 0, 0
car_angle = 0
speed = 0.32
target_x, target_y = 13, 0

# 障碍物
obstacles = [
    (2, 2), (5, -2), (4, 0.5), (7, 1.5), (9, -1)
]

particles = []

# =============== 【坐标投影】 ===============
def project(px, py):
    scale = 60
    ox = 200
    oy = 380
    return (ox + px * scale, oy + py * scale)

def add_particle(x, y):
    particles.append({
        "x": x, "y": y,
        "life": 20,
        "size": random.randint(3,8)
    })

def update_particles():
    for p in particles[:]:
        p["life"] -= 1
        p["size"] -= 0.2
        if p["life"] <= 0:
            particles.remove(p)

# =============== 【主更新逻辑】 ===============
def update():
    global car_x, car_y, car_angle, speed
    canvas.delete("all")
    update_particles()

    # 距离目标
    dx = target_x - car_x
    dy = target_y - car_y
    dist = math.hypot(dx, dy)

    # 避障力
    avoid = 0
    avoiding = False
    for ox, oy in obstacles:
        d = math.hypot(car_x-ox, car_y-oy)
        if d < 3.0:
            avoid += (car_y - oy) * 3.2
            avoiding = True
            add_particle(car_x, car_y)  # 避障特效

    # 行驶逻辑
    if dist > 1.2:
        des_angle = math.atan2(dx, dy)
        car_angle += (des_angle - car_angle + avoid) * 0.12
        car_x += math.sin(car_angle) * speed
        car_y += math.cos(car_angle) * speed

    # 边界保护
    if abs(car_x) > 14: car_x = 14 * (1 if car_x>0 else -1)
    if abs(car_y) > 14: car_y = 14 * (1 if car_y>0 else -1)

    # =============== 【绘制：地面网格】 ===============
    for i in range(-12,13):
        x1,y1 = project(i,-12)
        x2,y2 = project(i,12)
        canvas.create_line(x1,y1,x2,y2,fill="#225522")
        x1,y1 = project(-12,i)
        x2,y2 = project(12,i)
        canvas.create_line(x1,y1,x2,y2,fill="#225522")

    # =============== 【绘制：障碍物】 ===============
    for ox, oy in obstacles:
        x,y = project(ox, oy)
        canvas.create_oval(x-30,y-30,x+30,y+30,fill="#ff2222",outline="#fff",width=2)

    # =============== 【绘制：目标点】 ===============
    tx, ty = project(target_x, target_y)
    canvas.create_oval(tx-35,ty-35,tx+35,ty+35,fill="#22ff55",outline="#fff",width=2)


    for p in particles:
        x,y = project(p["x"], p["y"])
        s = p["size"]
        canvas.create_oval(x-s,y-s,x+s,y+s,fill="#00ffff")

    # =============== 【绘制：小车】 ===============
    cx, cy = project(car_x, car_y)
    hx = cx + math.sin(car_angle) * 40
    hy = cy + math.cos(car_angle) * 40
    canvas.create_polygon(
        cx-20, cy-30,
        hx, hy,
        cx+20, cy-30,
        fill="#3399ff",
        outline="#fff",
        width=2
    )

    # =============== 【顶部信息面板】 ===============
    status = "⚠️  避障中" if avoiding else "✅ 正常行驶"
    text = (
        f"坐标 ({car_x:.1f}, {car_y:.1f}) | "
        f"目标距离 {dist:.1f}m | "
        f"{status}"
    )
    canvas.create_text(20,20,text=text,fill="#00ff00",font=("Consolas",13,"bold"),anchor="w")

    # =============== 【标题】 ===============
    canvas.create_text(550,70,text=" 智能无人小车自动导航系统",fill="#ffffff",font=("黑体",20,"bold"))

    root.after(30, update)

# 启动
update()
root.mainloop()