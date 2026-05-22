# collision_data_collector.py
"""碰撞检测数据采集模块

采集前视深度图像，标注是否有障碍物。
用于训练碰撞预测模型。
"""

import os
import time
import airsim
import numpy as np
import cv2
from datetime import datetime


class CollisionDataCollector:
    """碰撞数据采集器"""

    def __init__(self, output_dir=None):
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, "collision_dataset")
        
        self.output_dir = output_dir
        self.depth_dir = os.path.join(output_dir, "depth")
        self.labels_file = os.path.join(output_dir, "labels.csv")

        self._create_directories()
        
        # 如果 labels.csv 不存在，则创建并写入表头
        if not os.path.exists(self.labels_file):
            with open(self.labels_file, 'w') as f:
                f.write("filename,label,risk,min_depth,mean_depth,x,y,z\n")

        
        
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("✅ AirSim 连接成功")
        
        self.sample_count = 0
        self.current_label = 0  # 0=安全, 1=危险

    def _create_directories(self):
        if not os.path.exists(self.depth_dir):
            os.makedirs(self.depth_dir)

    def set_label(self, label):
        """设置标签: 0=安全, 1=危险"""
        self.current_label = label
        name = "安全(无障碍)" if label == 0 else "危险(有障碍)"
        print(f"🏷️ 当前标签: {label} - {name}")

    def capture_sample(self):
        """采集一个深度图像样本"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
            ])
            
            if not responses or len(responses) == 0:
                print("❌ 获取深度图像失败")
                return False
            
            depth_data = np.array(responses[0].image_data_float, dtype=np.float32)
            depth_image = depth_data.reshape(responses[0].height, responses[0].width)
            
            pos = self.client.getMultirotorState().kinematics_estimated.position
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"collision_{timestamp}_{pos.x_val:.1f}_{pos.y_val:.1f}"
            
            min_depth = np.min(depth_image)
            mean_depth = np.mean(depth_image)
            
            # 保存伪彩色深度图
            depth_norm = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            depth_path = os.path.join(self.depth_dir, f"{filename}_depth.png")
            cv2.imwrite(depth_path, depth_colored)
            
            # 保存标签
            risk_name = "safe" if self.current_label == 0 else "danger"
            with open(self.labels_file, 'a') as f:
                f.write(f"{filename},{self.current_label},{risk_name},"
                        f"{min_depth:.2f},{mean_depth:.2f},"
                        f"{pos.x_val:.1f},{pos.y_val:.1f},{pos.z_val:.1f}\n")
            self.sample_count += 1
            print(f"✅ 样本 {self.sample_count}: {filename} (深度:{min_depth:.1f}m)")
            return True
            
        except Exception as e:
            print(f"❌ 采集失败: {e}")
            return False

    def get_stats(self):
        """获取数据集统计"""
        if not os.path.exists(self.labels_file):
            return None
        
        safe_count = 0
        danger_count = 0
        with open(self.labels_file, 'r') as f:
            next(f)  # 跳过表头
            for line in f:
                if ',0,' in line:
                    safe_count += 1
                elif ',1,' in line:
                    danger_count += 1
        return {'safe': safe_count, 'danger': danger_count}


if __name__ == "__main__":
    from pynput import keyboard
    from pynput.keyboard import Key
    
    collector = CollisionDataCollector()
    
    # 起飞
    print("\n🚀 正在起飞...")
    collector.client.enableApiControl(True)
    collector.client.armDisarm(True)
    collector.client.takeoffAsync().join()
    time.sleep(1)
    collector.client.moveToZAsync(-5, 2).join()  # 飞到5米高度
    print("✅ 起飞成功！")
    
    print("""
    ============================================
         碰撞检测数据采集
    ============================================
    
    【飞行控制】
      W - 前进    S - 后退
      A - 左移    D - 右移
      Q - 上升    E - 下降
      空格 - 悬停  L - 降落
    
    【标签设置】
      0 - 安全（前方无障碍物）
      1 - 危险（前方有障碍物）
    
    【采集】
      C - 采集当前样本
      P - 自动采集10个（间隔2秒）
    
    【退出】
      ESC - 退出
    
    ============================================
    """)
    
    keys_pressed = set()
    
    def on_press(key):
        try:
            keys_pressed.add(key.char)
            
            # 标签设置
            if key.char == '0':
                collector.set_label(0)
            elif key.char == '1':
                collector.set_label(1)
            elif key.char == 'c':
                collector.capture_sample()
            elif key.char == 'p':  # 自动采集
                print("自动采集中...")
                for i in range(10):
                    print(f"[{i+1}/10]")
                    collector.capture_sample()
                    time.sleep(2)
            # 飞行控制 (AirSim坐标系: X=前后, Y=左右, Z=上下)
            elif key.char == 'w':
                collector.client.moveByVelocityAsync(3, 0, 0, 1)  # 前进
            elif key.char == 's':
                collector.client.moveByVelocityAsync(-3, 0, 0, 1)  # 后退
            elif key.char == 'a':
                collector.client.moveByVelocityAsync(0, -3, 0, 1)  # 左移
            elif key.char == 'd':
                collector.client.moveByVelocityAsync(0, 3, 0, 1)  # 右移
            elif key.char == 'q':
                collector.client.moveByVelocityAsync(0, 0, -2, 1)  # 上升
            elif key.char == 'e':
                collector.client.moveByVelocityAsync(0, 0, 2, 1)  # 下降
                
        except AttributeError:
            pass
    
    def on_release(key):
        try:
            keys_pressed.discard(key.char)
            # 松开移动键时悬停
            if key.char in ['w', 's', 'a', 'd', 'q', 'e']:
                collector.client.hoverAsync()
            # 降落
            if key.char == 'l':
                print("🛬 收到降落指令...")
                collector.client.landAsync()
                return False
        except AttributeError:
            pass
        
        if key == Key.esc:
            return False
    
    print("⏳ 开始监听键盘...")
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    while listener.is_alive():
        time.sleep(0.1)
    
    stats = collector.get_stats()
    if stats:
        print(f"\n📊 数据集统计: 安全={stats['safe']}, 危险={stats['danger']}")
    
    print("👋 退出")
