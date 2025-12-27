import sys
import math
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph.opengl as gl

class ArmLink:
    """机械臂连杆类，定义单个连杆的属性和绘制方法"""
    def __init__(self, length, radius=0.1):
        self.length = length  # 连杆长度
        self.radius = radius  # 连杆半径
        self.angle = 0        # 关节角度
        self.pos = np.array([0, 0, 0])  # 连杆起始位置
        self.vertices = None  # 连杆顶点数据
        self.color = [0.2, 0.6, 0.8, 1.0]  # 连杆颜色（蓝绿色）
        
    def update_position(self, parent_pos, parent_angle):
        """根据父连杆的位置和角度更新当前连杆的位置"""
        # 计算当前连杆的绝对角度
        self.angle = parent_angle + self.angle
        
        # 计算连杆末端位置
        end_x = parent_pos[0] + self.length * math.cos(math.radians(self.angle))
        end_y = parent_pos[1] + self.length * math.sin(math.radians(self.angle))
        end_z = parent_pos[2]
        
        # 生成连杆的圆柱顶点
        self._create_cylinder(parent_pos, [end_x, end_y, end_z])
        
        return np.array([end_x, end_y, end_z])
    
    def _create_cylinder(self, start, end):
        """创建圆柱几何体"""
        # 生成圆柱的圆周点
        theta = np.linspace(0, 2*np.pi, 20)
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        
        # 创建圆柱侧面
        points = []
        for i in range(len(theta)):
            points.append([start[0] + x[i], start[1] + y[i], start[2]])
            points.append([end[0] + x[i], end[1] + y[i], end[2]])
        
        self.vertices = np.array(points)

class RoboticArmSimulator(QMainWindow):
    """机械臂仿真主窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D机械臂仿真")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建机械臂连杆（3个关节的机械臂）
        self.arm_links = [
            ArmLink(length=1.0),  # 第一连杆
            ArmLink(length=0.8),  # 第二连杆
            ArmLink(length=0.6)   # 第三连杆
        ]
        
        # 初始化UI
        self._init_ui()
        
        # 设置定时器更新仿真画面
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_arm)
        self.timer.start(30)  # 约30fps
        
        # 动画参数
        self.animation_angle = 0

    def _init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建3D绘图区域
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=5, elevation=30, azimuth=45)
        main_layout.addWidget(self.gl_widget, 3)
        
        # 添加坐标系
        axis = gl.GLAxisItem()
        axis.setSize(2, 2, 2)
        self.gl_widget.addItem(axis)
        
        # 创建控制面板
        control_panel = QGroupBox("关节控制")
        control_layout = QGridLayout(control_panel)
        
        # 创建关节角度滑块
        self.sliders = []
        for i, link in enumerate(self.arm_links):
            label = QLabel(f"关节 {i+1} 角度:")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-180, 180)
            slider.setValue(0)
            slider.valueChanged.connect(self.on_slider_changed)
            
            value_label = QLabel("0°")
            self.sliders.append((slider, value_label))
            
            control_layout.addWidget(label, i, 0)
            control_layout.addWidget(slider, i, 1)
            control_layout.addWidget(value_label, i, 2)
        
        main_layout.addWidget(control_panel, 1)
        
        # 创建机械臂的3D绘制对象
        self.arm_items = []
        for link in self.arm_links:
            item = gl.GLLinePlotItem(pos=link.vertices, color=link.color, width=2, antialias=True)
            self.arm_items.append(item)
            self.gl_widget.addItem(item)
        
        # 添加机械臂末端点
        self.end_effector = gl.GLScatterPlotItem(pos=np.array([[0,0,0]]), size=10, color=[1,0,0,1])
        self.gl_widget.addItem(self.end_effector)

    def on_slider_changed(self):
        """滑块值变化时更新关节角度"""
        for i, (slider, label) in enumerate(self.sliders):
            angle = slider.value()
            label.setText(f"{angle}°")
            self.arm_links[i].angle = angle

    def update_arm(self):
        """更新机械臂的位置和显示"""
        # 基础位置（机械臂底座）
        current_pos = np.array([0, 0, 0])
        current_angle = 0
        
        # 如果滑块没有被手动调整，自动动画演示
        if all(slider[0].value() == 0 for slider in self.sliders):
            self.animation_angle += 1
            for i, link in enumerate(self.arm_links):
                link.angle = 45 * math.sin(math.radians(self.animation_angle + i * 120))
                self.sliders[i][0].setValue(int(link.angle))
                self.sliders[i][1].setText(f"{int(link.angle)}°")
        
        # 更新每个连杆的位置
        for i, (link, item) in enumerate(zip(self.arm_links, self.arm_items)):
            current_pos = link.update_position(current_pos, current_angle)
            current_angle = link.angle
            item.setData(pos=link.vertices)
        
        # 更新末端执行器位置
        self.end_effector.setData(pos=np.array([current_pos]))

if __name__ == "__main__":
    # 设置高DPI支持（解决Windows下界面模糊问题）
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    app = QApplication(sys.argv)
    simulator = RoboticArmSimulator()
    simulator.show()
    sys.exit(app.exec_())