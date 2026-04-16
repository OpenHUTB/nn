import tkinter as tk
import threading
import time

class MapDisplay:
    def __init__(self):
        self.root = None
        self.running = False
        self.canvas = None
        self.trail_points = []
        self.waypoints = []
        
        # 地图中心坐标
        self.map_center_x = 400
        self.map_center_y = 200
        # 缩放因子
        self.map_scale = 10  # 1米 = 10像素
        
        self.initialize()
    
    def initialize(self):
        """初始化地图显示"""
        try:
            print("开始初始化地图显示...")
            # 创建主窗口
            self.root = tk.Tk()
            self.root.title("无人机地图显示")
            self.root.geometry("800x600")
            
            # 创建Canvas组件
            self.canvas = tk.Canvas(self.root, width=800, height=500, bg="white")
            self.canvas.pack(pady=20)
            
            # 绘制初始地图
            self.draw_map_grid()
            
            self.running = True
            print("地图显示初始化成功")
        except Exception as e:
            print(f"地图显示初始化失败: {e}")
            self.running = False
    
    def draw_map_grid(self):
        """绘制地图网格"""
        if not self.canvas:
            return
        
        # 清除画布
        self.canvas.delete("all")
        
        # 绘制网格线
        for i in range(-20, 21):
            # 垂直线
            x = self.map_center_x + i * self.map_scale * 2
            self.canvas.create_line(x, 0, x, 500, fill="lightgray", dash=(2, 2))
            # 水平线
            y = self.map_center_y + i * self.map_scale * 2
            self.canvas.create_line(0, y, 800, y, fill="lightgray", dash=(2, 2))
        
        # 绘制坐标轴
        self.canvas.create_line(self.map_center_x, 0, self.map_center_x, 500, fill="black", width=2)
        self.canvas.create_line(0, self.map_center_y, 800, self.map_center_y, fill="black", width=2)
        
        # 绘制原点标记
        self.canvas.create_oval(self.map_center_x-5, self.map_center_y-5, 
                              self.map_center_x+5, self.map_center_y+5, 
                              fill="red")
        self.canvas.create_text(self.map_center_x+10, self.map_center_y-10, 
                              text="原点", font=("Arial", 10))
    
    def set_waypoints(self, waypoints):
        """设置巡航路线航点"""
        self.waypoints = waypoints
        # 触发地图重绘
        if self.canvas and self.running:
            self.root.after(0, self.draw_map_grid)
    
    def _update_position(self, x, y):
        """在主线程中更新无人机位置"""
        if not self.canvas or not self.running:
            return
        
        # 将 AirSim 局部坐标转换为地图坐标
        map_x = self.map_center_x + x * self.map_scale
        map_y = self.map_center_y - y * self.map_scale  # 注意y轴方向相反
        
        # 更新航迹点
        self.trail_points.append((map_x, map_y))
        # 限制航迹点数量，避免内存占用过大
        if len(self.trail_points) > 1000:
            self.trail_points.pop(0)
        
        # 重绘地图
        self.draw_map_grid()
        
        # 绘制巡航路线
        if len(self.waypoints) > 1:
            for i in range(1, len(self.waypoints)):
                # 将航点坐标转换为地图坐标
                wp1_x, wp1_y, _ = self.waypoints[i-1]
                wp2_x, wp2_y, _ = self.waypoints[i]
                map_wp1_x = self.map_center_x + wp1_x * self.map_scale
                map_wp1_y = self.map_center_y - wp1_y * self.map_scale
                map_wp2_x = self.map_center_x + wp2_x * self.map_scale
                map_wp2_y = self.map_center_y - wp2_y * self.map_scale
                # 绘制巡航路线
                self.canvas.create_line(map_wp1_x, map_wp1_y, map_wp2_x, map_wp2_y, fill="red", width=2, dash=(5, 2))
        
        # 绘制航点
        for i, (wp_x, wp_y, _) in enumerate(self.waypoints):
            map_wp_x = self.map_center_x + wp_x * self.map_scale
            map_wp_y = self.map_center_y - wp_y * self.map_scale
            # 绘制航点标记
            self.canvas.create_oval(map_wp_x-6, map_wp_y-6, map_wp_x+6, map_wp_y+6, fill="yellow")
            self.canvas.create_text(map_wp_x+10, map_wp_y-10, text=f"{i+1}", font=("Arial", 10))
        
        # 绘制航迹
        if len(self.trail_points) > 1:
            for i in range(1, len(self.trail_points)):
                self.canvas.create_line(self.trail_points[i-1][0], self.trail_points[i-1][1],
                                      self.trail_points[i][0], self.trail_points[i][1],
                                      fill="blue", width=2)
        
        # 绘制无人机
        self.canvas.create_oval(map_x-8, map_y-8, map_x+8, map_y+8, fill="green")
        self.canvas.create_text(map_x+15, map_y-10, text="无人机", font=("Arial", 10))
    
    def update_position(self, x, y):
        """更新无人机位置"""
        if not self.canvas or not self.running:
            return
        
        # 使用after方法在主线程中执行更新操作
        self.root.after(0, self._update_position, x, y)
    
    def run(self):
        """运行地图显示"""
        if self.root and self.running:
            try:
                self.root.mainloop()
            finally:
                self.running = False
                print("地图显示已关闭")
    
    def stop(self):
        """停止地图显示"""
        if self.root:
            self.running = False
            self.root.quit()
            print("地图显示已停止")

if __name__ == "__main__":
    # 测试地图显示
    print("测试地图显示...")
    map_display = MapDisplay()
    if map_display.running:
        print("地图显示初始化成功，启动主循环...")
        map_display.run()
    else:
        print("地图显示初始化失败")