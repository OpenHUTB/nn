import time
import random

class Robot:
    def __init__(self, name="RunnerBot"):
        self.name = name
        self.is_running = False
        self.speed = 5  # 跑步速度，范围1-10
        self.energy = 100  # 能量值，跑步会消耗能量
        self.position = 0  # 当前位置
        
    def start_running(self):
        """开始跑步"""
        if self.energy <= 0:
            print(f"{self.name}能量耗尽，无法跑步！")
            return
            
        if not self.is_running:
            self.is_running = True
            print(f"{self.name}开始跑步！速度: {self.speed}")
            self._running_loop()
            
    def stop_running(self):
        """停止跑步"""
        if self.is_running:
            self.is_running = False
            print(f"\n{self.name}停止跑步。当前位置: {self.position:.2f}米，剩余能量: {self.energy}%")
            
    def set_speed(self, new_speed):
        """设置跑步速度"""
        if 1 <= new_speed <= 10:
            self.speed = new_speed
            print(f"速度已调整为: {self.speed}")
        else:
            print("速度必须在1-10之间！")
            
    def _consume_energy(self):
        """消耗能量"""
        energy_loss = self.speed * 0.5
        self.energy = max(0, self.energy - energy_loss)
        if self.energy <= 0:
            self.stop_running()
            print(f"{self.name}能量耗尽！")
            
    def _update_position(self):
        """更新位置"""
        distance = self.speed * 0.1  # 每0.1秒移动的距离
        self.position += distance
        return distance
            
    def _running_loop(self):
        """跑步循环逻辑"""
        while self.is_running and self.energy > 0:
            distance = self._update_position()
            self._consume_energy()
            
            # 模拟跑步动作反馈
            actions = ["迈左腿", "迈右腿", "摆左臂", "摆右臂", "身体前倾"]
            action = random.choice(actions)
            
            print(f"{self.name}: {action} | 已跑: {self.position:.2f}米 | 能量: {self.energy:.1f}%", end="\r")
            time.sleep(0.1)  # 控制循环速度

# 使用示例
if __name__ == "__main__":
    # 创建机器人实例
    robot = Robot("运动机器人")
    
    # 开始跑步
    robot.start_running()
    
    # 运行2秒后调整速度
    time.sleep(2)
    robot.set_speed(8)
    
    # 再运行3秒后停止
    time.sleep(3)
    robot.stop_running()
    
    # 尝试再次跑步
    time.sleep(1)
    robot.start_running()
    