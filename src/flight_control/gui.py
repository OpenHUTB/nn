import tkinter as tk
from tkinter import ttk
import threading
import time

class FlightControlGUI:
    def __init__(self, client):
        self.client = client
        self.root = None
        self.running = False
        
        try:
            print("开始初始化 GUI...")
            # 尝试创建 tkinter 窗口
            self.root = tk.Tk()
            print("创建窗口成功")
            self.root.title("无人机控制系统")
            self.root.geometry("800x600")
            
            # 创建主框架
            self.main_frame = ttk.Frame(self.root, padding="10")
            self.main_frame.pack(fill=tk.BOTH, expand=True)
            print("创建主框架成功")
            
            # 创建状态显示区域
            self.status_frame = ttk.LabelFrame(self.main_frame, text="无人机状态", padding="10")
            self.status_frame.pack(fill=tk.X, pady=5)
            
            # 状态标签
            self.status_labels = {
                "连接状态": tk.StringVar(value="未连接"),
                "飞行状态": tk.StringVar(value="未起飞"),
                "位置": tk.StringVar(value="(0, 0, 0)"),
                "速度": tk.StringVar(value="(0, 0, 0)"),
                "高度": tk.StringVar(value="0.0 m")
            }
            
            for i, (label, var) in enumerate(self.status_labels.items()):
                ttk.Label(self.status_frame, text=label + ":").grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
                ttk.Label(self.status_frame, textvariable=var).grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
            print("创建状态显示区域成功")
            
            # 创建传感器数据区域
            self.sensor_frame = ttk.LabelFrame(self.main_frame, text="传感器数据", padding="10")
            self.sensor_frame.pack(fill=tk.X, pady=5)
            
            # 传感器标签
            self.sensor_labels = {
                "气压": tk.StringVar(value="0.0 hPa"),
                "温度": tk.StringVar(value="0.0 °C"),
                "湿度": tk.StringVar(value="0.0 %")
            }
            
            for i, (label, var) in enumerate(self.sensor_labels.items()):
                ttk.Label(self.sensor_frame, text=label + ":").grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
                ttk.Label(self.sensor_frame, textvariable=var).grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
            print("创建传感器数据区域成功")
            
            # 创建控制按钮区域
            self.control_frame = ttk.LabelFrame(self.main_frame, text="控制", padding="10")
            self.control_frame.pack(fill=tk.X, pady=5)
            
            # 控制按钮
            control_buttons = [
                ("起飞", self.takeoff),
                ("降落", self.land),
                ("悬停", self.hover),
                ("返航", self.return_home)
            ]
            
            for i, (text, command) in enumerate(control_buttons):
                ttk.Button(self.control_frame, text=text, command=command).grid(row=0, column=i, padx=5, pady=5)
            print("创建控制按钮区域成功")
            
            # 创建参数设置区域
            self.params_frame = ttk.LabelFrame(self.main_frame, text="参数设置", padding="10")
            self.params_frame.pack(fill=tk.X, pady=5)
            
            # 速度设置
            ttk.Label(self.params_frame, text="飞行速度:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.speed_var = tk.DoubleVar(value=2.0)
            ttk.Entry(self.params_frame, textvariable=self.speed_var, width=10).grid(row=0, column=1, padx=5, pady=2)
            ttk.Label(self.params_frame, text="m/s").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
            
            # 高度设置
            ttk.Label(self.params_frame, text="飞行高度:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            self.height_var = tk.DoubleVar(value=-3.0)
            ttk.Entry(self.params_frame, textvariable=self.height_var, width=10).grid(row=1, column=1, padx=5, pady=2)
            ttk.Label(self.params_frame, text="m").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
            print("创建参数设置区域成功")
            

            
            # 创建日志区域
            self.log_frame = ttk.LabelFrame(self.main_frame, text="日志", padding="10")
            self.log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
            
            self.log_text = tk.Text(self.log_frame, wrap=tk.WORD)
            self.log_text.pack(fill=tk.BOTH, expand=True)
            
            # 滚动条
            scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.log_text.config(yscrollcommand=scrollbar.set)
            print("创建日志区域成功")
            
            # 开始更新数据
            self.running = True
            self.update_thread = threading.Thread(target=self.update_data)
            self.update_thread.daemon = True
            self.update_thread.start()
            
            print("GUI 初始化成功")
        except Exception as e:
            print(f"GUI 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.running = False
        
    def takeoff(self):
        def _takeoff():
            try:
                self.log("起飞中...")
                self.client.takeoffAsync().join()
                self.log("起飞完成")
            except Exception as e:
                self.log(f"起飞失败: {e}")
        
        threading.Thread(target=_takeoff).start()
    
    def land(self):
        def _land():
            try:
                self.log("降落中...")
                self.client.landAsync().join()
                self.log("降落完成")
            except Exception as e:
                self.log(f"降落失败: {e}")
        
        threading.Thread(target=_land).start()
    
    def hover(self):
        def _hover():
            try:
                self.log("悬停")
                self.client.hoverAsync().join()
            except Exception as e:
                self.log(f"悬停失败: {e}")
        
        threading.Thread(target=_hover).start()
    
    def return_home(self):
        def _return_home():
            try:
                self.log("返航原点")
                height = self.height_var.get()
                self.client.moveToPositionAsync(0, 0, height, 2).join()
            except Exception as e:
                self.log(f"返航失败: {e}")
        
        threading.Thread(target=_return_home).start()
    
    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def update_data(self):
        def _update_data():
            while self.running:
                try:
                    # 检查客户端是否连接
                    if not self.client:
                        self.status_labels["连接状态"].set("未连接")
                        self.status_labels["位置"].set("(0, 0, 0)")
                        self.status_labels["速度"].set("(0, 0, 0)")
                        self.status_labels["高度"].set("0.0 m")
                        self.status_labels["飞行状态"].set("未起飞")
                        time.sleep(0.5)
                        continue
                    
                    # 更新状态信息
                    state = self.client.getMultirotorState()
                    
                    # 位置
                    pos = state.kinematics_estimated.position
                    self.status_labels["位置"].set(f"({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
                    
                    # 速度
                    vel = state.kinematics_estimated.linear_velocity
                    self.status_labels["速度"].set(f"({vel.x_val:.2f}, {vel.y_val:.2f}, {vel.z_val:.2f})")
                    
                    # 高度
                    self.status_labels["高度"].set(f"{abs(pos.z_val):.2f} m")
                    
                    # 飞行状态
                    if state.landed_state == 1:
                        self.status_labels["飞行状态"].set("已降落")
                    else:
                        self.status_labels["飞行状态"].set("飞行中")
                    
                    # 更新传感器数据
                    # 注意：AirSim 可能需要额外配置才能获取这些数据
                    try:
                        barometer = self.client.getBarometerData()
                        self.sensor_labels["气压"].set(f"{barometer.pressure:.2f} hPa")
                    except:
                        pass
                    
                    # 连接状态
                    self.status_labels["连接状态"].set("已连接")
                    
                except Exception as e:
                    self.status_labels["连接状态"].set("连接失败")
                    self.log(f"更新数据失败: {e}")
                
                time.sleep(0.5)
        
        threading.Thread(target=_update_data).start()
    

    
    def stop(self):
        """停止 GUI"""
        if self.root:
            self.running = False
            self.root.quit()
            print("GUI 已停止")
    
    def run(self):
        if self.root is None:
            print("GUI 未初始化，无法运行")
            return
        
        try:
            self.root.mainloop()
        finally:
            self.running = False
            if hasattr(self, 'update_thread') and self.update_thread.is_alive():
                self.update_thread.join(timeout=1.0)
            print("GUI 已关闭")
