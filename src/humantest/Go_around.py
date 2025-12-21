import time
import random

class Robot:
    def __init__(self, name="RunnerBot"):
        self.name = name
        self.is_running = False
        self.speed = 5  # 跑步速度，范围1-10
        self.energy = 100  # 能量值，跑步/绕障会消耗能量
        self.position = 0  # 当前纵向位置（前进方向，单位：米）
        self.lateral_offset = 0  # 横向偏移量（左右方向，0=默认路径，正数=右偏，负数=左偏）
        
        # 障碍相关属性
        self.obstacle_detected = False  # 是否检测到障碍
        self.current_obstacle = None    # 当前障碍信息（字典：位置、宽度）
        self.avoiding_obstacle = False  # 是否处于绕障状态
        self.avoid_direction = 0        # 绕障方向（1=向右绕，-1=向左绕）
        self.avoid_progress = 0         # 绕障进度（0=未开始，1=完成）

    def start_running(self):
        """开始跑步"""
        if self.energy <= 0:
            print(f"\n{self.name}能量耗尽，无法跑步！")
            return
            
        if not self.is_running:
            self.is_running = True
            print(f"\n{self.name}开始跑步！速度: {self.speed} | 初始路径：中间")
            self._running_loop()

    def stop_running(self):
        """停止跑步"""
        if self.is_running:
            self.is_running = False
            self.avoiding_obstacle = False  # 停止时重置绕障状态
            print(f"\n{self.name}停止跑步。")
            print(f"最终位置: 纵向{self.position:.2f}米 | 横向偏移{self.lateral_offset:.2f}米")
            print(f"剩余能量: {self.energy:.1f}%")

    def set_speed(self, new_speed):
        """设置跑步速度"""
        if 1 <= new_speed <= 10:
            self.speed = new_speed
            print(f"\n{self.name}：速度已调整为: {self.speed}")
        else:
            print(f"\n{self.name}：速度必须在1-10之间！")

    def _consume_energy(self, is_avoiding=False):
        """消耗能量（绕障时消耗更多）"""
        # 绕障时能量消耗增加30%（模拟额外动作消耗）
        energy_loss = self.speed * 0.5 * (1.3 if is_avoiding else 1.0)
        self.energy = max(0, self.energy - energy_loss)
        
        if self.energy <= 0:
            self.stop_running()
            print(f"\n{self.name}能量耗尽！")

    def _simulate_obstacle_detection(self):
        """模拟障碍检测（随机生成前方障碍，仅在无障碍时检测）"""
        if not self.obstacle_detected and not self.avoiding_obstacle:
            # 每20次循环（约2秒）有30%概率生成前方障碍
            if random.randint(1, 20) == 1 and random.random() < 0.3:
                # 障碍生成规则：在机器人前方5-10米处，宽度2-4米
                obstacle_pos = self.position + random.uniform(5, 10)
                obstacle_width = random.uniform(2, 4)
                self.obstacle_detected = True
                self.current_obstacle = {
                    "pos": obstacle_pos,    # 障碍纵向位置
                    "width": obstacle_width # 障碍宽度
                }
                print(f"\n{self.name}：检测到前方障碍！位置：{obstacle_pos:.2f}米，宽度：{obstacle_width:.2f}米")

    def _start_avoid_obstacle(self):
        """启动绕障流程（随机选择左右方向）"""
        self.avoiding_obstacle = True
        self.avoid_direction = random.choice([1, -1])  # 1=右绕，-1=左绕
        self.avoid_progress = 0
        direction_str = "向右" if self.avoid_direction == 1 else "向左"
        print(f"{self.name}：开始{direction_str}绕过障碍，横向偏移中...")

    def _update_avoidance(self):
        """更新绕障状态（横向偏移+前进，直到绕过障碍）"""
        # 绕障参数：横向偏移最大距离（2米，确保避开障碍）、每步偏移量
        max_lateral_offset = 2.0
        lateral_step = 0.05  # 每0.1秒横向偏移0.05米
        obstacle = self.current_obstacle

        # 阶段1：横向偏移到目标位置（避开障碍）
        if abs(self.lateral_offset) < max_lateral_offset and self.avoid_progress < 0.5:
            self.lateral_offset += lateral_step * self.avoid_direction
            # 偏移到目标位置后，进入阶段2（前进绕过障碍）
            if abs(self.lateral_offset) >= max_lateral_offset:
                self.avoid_progress = 0.5
                print(f"{self.name}：横向偏移完成，继续前进绕过障碍...")

        # 阶段2：前进至障碍后方（纵向位置超过障碍位置+宽度）
        elif self.avoid_progress == 0.5:
            # 前进逻辑不变，直到纵向位置超过障碍末端
            if self.position > obstacle["pos"] + obstacle["width"]:
                self.avoid_progress = 0.8
                direction_str = "向左" if self.avoid_direction == 1 else "向右"
                print(f"{self.name}：已绕过障碍末端，开始{direction_str}回正路径...")

        # 阶段3：横向回正（恢复到默认路径）
        elif self.avoid_progress == 0.8:
            if abs(self.lateral_offset) > 0:
                # 反向偏移回正
                self.lateral_offset -= lateral_step * self.avoid_direction
            else:
                # 绕障完成，重置状态
                self.avoiding_obstacle = False
                self.obstacle_detected = False
                self.current_obstacle = None
                self.avoid_progress = 0
                print(f"{self.name}：绕障完成！恢复默认路径前进")

    def _update_position(self):
        """更新位置（正常前进/绕障前进）"""
        # 纵向前进距离（每0.1秒）
        forward_distance = self.speed * 0.1
        
        # 若处于绕障状态，优先执行绕障逻辑
        if self.avoiding_obstacle:
            self._update_avoidance()
        # 若检测到障碍且未绕障，启动绕障
        elif self.obstacle_detected:
            # 当机器人接近障碍（距离<2米）时开始绕障
            if self.position >= self.current_obstacle["pos"] - 2.0:
                self._start_avoid_obstacle()
        
        # 更新纵向位置
        self.position += forward_distance
        return forward_distance

    def _running_loop(self):
        """跑步主循环（整合障碍检测、绕障、位置更新）"""
        while self.is_running and self.energy > 0:
            # 1. 模拟障碍检测
            self._simulate_obstacle_detection()
            
            # 2. 更新位置（含绕障逻辑）
            self._update_position()
            
            # 3. 消耗能量（绕障时消耗更多）
            self._consume_energy(is_avoiding=self.avoiding_obstacle)
            
            # 4. 生成跑步动作反馈（含绕障状态）
            base_actions = ["迈左腿", "迈右腿", "摆左臂", "摆右臂", "身体前倾"]
            if self.avoiding_obstacle:
                # 绕障时添加方向动作
                dir_action = "向右偏移" if self.avoid_direction == 1 else "向左偏移"
                action = random.choice(base_actions + [dir_action, dir_action])  # 增加方向动作概率
            else:
                action = random.choice(base_actions)
            
            # 5. 实时打印状态（覆盖当前行，保持界面简洁）
            path_info = f"路径：中间" if self.lateral_offset == 0 else f"路径：偏移{self.lateral_offset:.2f}米"
            avoid_info = "【绕障中】" if self.avoiding_obstacle else ""
            print(f"{self.name}{avoid_info}：{action} | 纵向位置: {self.position:.2f}米 | {path_info} | 能量: {self.energy:.1f}%", end="\r")
            
            # 控制循环速度（每0.1秒一次）
            time.sleep(0.1)

# 使用示例
if __name__ == "__main__":
    # 创建机器人实例
    sports_robot = Robot("运动先锋号")
    
    # 开始跑步
    sports_robot.start_running()
    
    # 运行5秒后调整速度
    time.sleep(5)
    sports_robot.set_speed(7)
    
    # 再运行8秒后停止
    time.sleep(8)
    sports_robot.stop_running()
    
    # 1秒后尝试再次跑步（能量可能不足）
    time.sleep(1)
    sports_robot.start_running()