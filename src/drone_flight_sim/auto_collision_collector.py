# auto_collision_collector.py
"""自动碰撞数据采集模块

利用仿真器内置的碰撞检测，让无人机自动飞行并采集数据：
- 安全样本(label=0)：飞行过程中定期采集
- 危险样本(label=1)：碰撞即将发生时自动采集

无需手动操作，短时间内可采集大量样本。
"""

import os
import time
import random
import airsim
import numpy as np
import cv2
from datetime import datetime


class AutoCollisionCollector:
    """自动碰撞数据采集器"""

    def __init__(self, output_dir=None, target_samples=100):
        """初始化自动采集器

        Args:
            output_dir: 数据保存目录
            target_samples: 目标样本数量（每类）
        """
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, "collision_dataset")

        self.output_dir = output_dir
        self.depth_dir = os.path.join(output_dir, "depth")
        self.labels_file = os.path.join(output_dir, "labels.csv")

        self._ensure_directories()

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("✅ AirSim 连接成功")

        self.target_samples = target_samples
        self.safe_samples = 0
        self.danger_samples = 0
        self.last_safe_capture_time = 0
        self.safe_capture_interval = 2.0  # 安全样本采集间隔(秒)

        self.running = True
        self.collision_cooldown = 2.0  # 碰撞冷却时间(秒)
        self.last_collision_time = 0
        self.was_colliding = False  # 记录上次是否在碰撞状态

        # 飞行控制参数
        self.target_altitude = -5  # 固定高度 -5米 (NED坐标系)
        self.fly_speed = 3.0       # 飞行速度 m/s

    def _ensure_directories(self):
        """确保目录存在"""
        if not os.path.exists(self.depth_dir):
            os.makedirs(self.depth_dir)

        # 如果 labels.csv 不存在，创建表头
        if not os.path.exists(self.labels_file):
            with open(self.labels_file, 'w') as f:
                f.write("filename,label,risk,min_depth,mean_depth,pos_x,pos_y\n")

    def connect(self):
        """连接并初始化无人机"""
        print("🔌 正在连接 AirSim...")
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("✅ 控制权获取成功")

    def takeoff(self, height=5):
        """起飞到指定高度"""
        print(f"🚀 起飞到 {height} 米...")
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-height, 2).join()
        self.target_altitude = -height  # 记录目标高度
        time.sleep(1)
        print(f"✅ 起飞完成，高度: {height}m")

    def capture_sample(self, label, prefix="auto"):
        """采集一个深度图像样本

        Args:
            label: 0=安全, 1=危险
            prefix: 文件名前缀

        Returns:
            bool: 是否成功采集
        """
        try:
            # 获取深度图像
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
            ])

            if not responses or len(responses) == 0:
                return False

            depth_data = np.array(responses[0].image_data_float, dtype=np.float32)
            depth_image = depth_data.reshape(responses[0].height, responses[0].width)

            # 获取位置
            pos = self.client.getMultirotorState().kinematics_estimated.position

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            risk_name = "safe" if label == 0 else "danger"
            filename = f"{prefix}_{timestamp}_{pos.x_val:.1f}_{pos.y_val:.1f}"

            # 计算深度统计
            min_depth = np.min(depth_image)
            mean_depth = np.mean(depth_image)

            # 保存伪彩色深度图
            depth_norm = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            depth_path = os.path.join(self.depth_dir, f"{filename}_depth.png")
            cv2.imwrite(depth_path, depth_colored)

            # 保存标签到 CSV
            with open(self.labels_file, 'a') as f:
                f.write(f"{filename},{label},{risk_name},"
                        f"{min_depth:.2f},{mean_depth:.2f},{pos.x_val:.1f},{pos.y_val:.1f}\n")

            if label == 0:
                self.safe_samples += 1
            else:
                self.danger_samples += 1

            return True

        except Exception as e:
            print(f"❌ 采集失败: {e}")
            return False

    def check_collision_event(self):
        """检测新的碰撞事件（带冷却机制）

        Returns:
            bool: 是否检测到新的碰撞事件
        """
        collision_info = self.client.simGetCollisionInfo()
        is_colliding = collision_info.has_collided
        current_time = time.time()

        # 检测从非碰撞 -> 碰撞的转变
        if is_colliding and not self.was_colliding:
            # 检查冷却时间
            if current_time - self.last_collision_time >= self.collision_cooldown:
                self.last_collision_time = current_time
                self.was_colliding = True
                return True

        # 如果当前不在碰撞状态，更新状态
        if not is_colliding:
            self.was_colliding = False

        return False

    def get_position(self):
        """获取当前位置"""
        return self.client.getMultirotorState().kinematics_estimated.position

    def _hover_and_maintain_altitude(self):
        """悬停并保持固定高度"""
        pos = self.get_position()
        current_z = pos.z_val
        # 如果高度偏离目标，自动调整
        if abs(current_z - self.target_altitude) > 0.5:
            # 用速度控制调整高度
            if current_z < self.target_altitude:
                # 太低了，上升
                self.client.moveByVelocityAsync(0, 0, -2, 0.5)
            else:
                # 太高了，下降
                self.client.moveByVelocityAsync(0, 0, 2, 0.5)
            time.sleep(0.5)
        self.client.hoverAsync()

    def fly_spiral(self, duration=120):
        """螺旋飞行采集 - 使用速度控制

        Args:
            duration: 飞行时长(秒)
        """
        print(f"\n🌀 开始螺旋飞行采集 (时长: {duration}秒)")

        start_time = time.time()
        start_pos = self.get_position()
        center_x, center_y = start_pos.x_val, start_pos.y_val
        angle = 0
        radius = 2

        # 先采集一个初始安全样本
        self.capture_sample(0, "init")
        self.last_safe_capture_time = time.time()

        while time.time() - start_time < duration and self.running:
            elapsed = time.time() - start_time

            # 检查目标是否达成
            if self.safe_samples >= self.target_samples and self.danger_samples >= self.target_samples:
                print("\n✅ 目标样本数达成！")
                break

            # 检测新的碰撞事件
            if self.check_collision_event():
                if self.danger_samples < self.target_samples:
                    print(f"\n💥 碰撞！采集危险样本 ({self.danger_samples + 1}/{self.target_samples})")
                    self.capture_sample(1, "collision")
                self._recover_from_collision()
                time.sleep(1.5)
                continue

            # 定期采集安全样本
            current_time = time.time()
            if current_time - self.last_safe_capture_time >= self.safe_capture_interval:
                if self.safe_samples < self.target_samples:
                    self.capture_sample(0, "safe")
                    self.last_safe_capture_time = current_time

            # 螺旋运动 - 使用速度控制
            angle += 0.1
            radius += 0.03
            # 计算速度方向
            vx = self.fly_speed * np.cos(angle)
            vy = self.fly_speed * np.sin(angle)
            vz = 0  # 保持高度

            # 检查是否超出边界，如果超出就反弹
            pos = self.get_position()
            if abs(pos.x_val - center_x) > 20 or abs(pos.y_val - center_y) > 20:
                # 反向螺旋
                angle += np.pi

            self.client.moveByVelocityAsync(vx, vy, vz, 0.2)

            # 显示进度
            if int(elapsed) % 2 == 0:
                pos = self.get_position()
                print(f"\r⏱ {int(elapsed)}s | 位置:({pos.x_val:.1f},{pos.y_val:.1f}) | "
                      f"✅ 安全: {self.safe_samples}/{self.target_samples} | "
                      f"💥 危险: {self.danger_samples}/{self.target_samples}", end="")

            time.sleep(0.1)

        self.client.hoverAsync()

    def fly_random(self, duration=120):
        """随机飞行采集 - 使用速度控制

        Args:
            duration: 飞行时长(秒)
        """
        print(f"\n🎲 开始随机飞行采集 (时长: {duration}秒)")

        start_time = time.time()
        pos = self.get_position()
        current_x, current_y = pos.x_val, pos.y_val
        target_x, target_y = random.uniform(-20, 20), random.uniform(-20, 20)

        # 先采集一个初始安全样本
        self.capture_sample(0, "init")
        self.last_safe_capture_time = time.time()

        while time.time() - start_time < duration and self.running:
            elapsed = time.time() - start_time

            # 检查目标是否达成
            if self.safe_samples >= self.target_samples and self.danger_samples >= self.target_samples:
                print("\n✅ 目标样本数达成！")
                break

            # 检测新的碰撞事件
            if self.check_collision_event():
                if self.danger_samples < self.target_samples:
                    print(f"\n💥 碰撞！采集危险样本 ({self.danger_samples + 1}/{self.target_samples})")
                    self.capture_sample(1, "collision")
                self._recover_from_collision()
                # 设置新目标
                target_x, target_y = random.uniform(-20, 20), random.uniform(-20, 20)
                time.sleep(1.5)
                continue

            # 定期采集安全样本
            current_time = time.time()
            if current_time - self.last_safe_capture_time >= self.safe_capture_interval:
                if self.safe_samples < self.target_samples:
                    self.capture_sample(0, "safe")
                    self.last_safe_capture_time = current_time

            # 计算到目标的方向并移动
            current_pos = self.get_position()
            dx = target_x - current_pos.x_val
            dy = target_y - current_pos.y_val
            dist = np.sqrt(dx*dx + dy*dy)

            if dist < 3:
                # 到达目标，设置新目标
                target_x, target_y = random.uniform(-20, 20), random.uniform(-20, 20)
            else:
                # 朝目标方向飞
                vx = (dx / dist) * self.fly_speed
                vy = (dy / dist) * self.fly_speed
                self.client.moveByVelocityAsync(vx, vy, 0, 0.2)

            # 显示进度
            if int(elapsed) % 2 == 0:
                print(f"\r⏱ {int(elapsed)}s | 位置:({current_pos.x_val:.1f},{current_pos.y_val:.1f}) | "
                      f"✅ 安全: {self.safe_samples}/{self.target_samples} | "
                      f"💥 危险: {self.danger_samples}/{self.target_samples}", end="")

            time.sleep(0.1)

        self.client.hoverAsync()

    def fly_zigzag(self, duration=120):
        """折线飞行采集 - 高速折返，高碰撞率

        Args:
            duration: 飞行时长(秒)
        """
        print(f"\n⚡ 开始折线飞行采集 (时长: {duration}秒)")

        start_time = time.time()
        pos = self.get_position()
        start_x, start_y = pos.x_val, pos.y_val

        direction = 1  # 1=正向, -1=反向
        fly_distance = 20
        self.fly_speed = 5.0  # 加速

        # 先采集一个初始安全样本
        self.capture_sample(0, "init")
        self.last_safe_capture_time = time.time()

        while time.time() - start_time < duration and self.running:
            elapsed = time.time() - start_time

            # 检查目标是否达成
            if self.safe_samples >= self.target_samples and self.danger_samples >= self.target_samples:
                print("\n✅ 目标样本数达成！")
                break

            # 检测新的碰撞事件
            if self.check_collision_event():
                if self.danger_samples < self.target_samples:
                    print(f"\n💥 碰撞！采集危险样本 ({self.danger_samples + 1}/{self.target_samples})")
                    self.capture_sample(1, "collision")
                self._recover_from_collision()
                direction *= -1  # 反弹方向
                time.sleep(1.5)
                continue

            # 定期采集安全样本
            current_time = time.time()
            if current_time - self.last_safe_capture_time >= self.safe_capture_interval:
                if self.safe_samples < self.target_samples:
                    self.capture_sample(0, "safe")
                    self.last_safe_capture_time = current_time

            # 直线高速飞行
            current_pos = self.get_position()
            target_x = start_x + direction * fly_distance

            dx = target_x - current_pos.x_val
            dist = abs(dx)

            if dist < 3:
                # 到达端点，换方向
                direction *= -1
                start_y = current_pos.y_val  # 更新起点
                start_x = current_pos.x_val
            else:
                # 直线飞向目标
                vx = np.sign(dx) * self.fly_speed
                self.client.moveByVelocityAsync(vx, 0, 0, 0.2)

            # 显示进度
            if int(elapsed) % 2 == 0:
                print(f"\r⏱ {int(elapsed)}s | 位置:({current_pos.x_val:.1f},{current_pos.y_val:.1f}) | "
                      f"✅ 安全: {self.safe_samples}/{self.target_samples} | "
                      f"💥 危险: {self.danger_samples}/{self.target_samples}", end="")

            time.sleep(0.1)

        self.client.hoverAsync()

    def fly_toward_obstacles(self, duration=120):
        """朝障碍物飞行采集 - 最高碰撞率

        Args:
            duration: 飞行时长(秒)
        """
        print(f"\n🎯 开始朝障碍物飞行采集 (时长: {duration}秒)")

        start_time = time.time()
        self.fly_speed = 4.0

        # 预设危险方向 - 朝向仿真环境中已知障碍物
        danger_directions = [
            (1, 0),   # 北(+X)
            (-1, 0),  # 南(-X)
            (0, 1),   # 东(+Y)
            (0, -1),  # 西(-Y)
            (0.7, 0.7),   # 东北
            (-0.7, 0.7),  # 西北
            (0.7, -0.7),  # 东南
            (-0.7, -0.7), # 西南
        ]

        direction_index = 0
        change_interval = 8  # 每8秒换方向

        # 先采集一个初始安全样本
        self.capture_sample(0, "init")
        self.last_safe_capture_time = time.time()

        while time.time() - start_time < duration and self.running:
            elapsed = time.time() - start_time

            # 检查目标是否达成
            if self.safe_samples >= self.target_samples and self.danger_samples >= self.target_samples:
                print("\n✅ 目标样本数达成！")
                break

            # 检测新的碰撞事件
            if self.check_collision_event():
                if self.danger_samples < self.target_samples:
                    print(f"\n💥 碰撞！采集危险样本 ({self.danger_samples + 1}/{self.target_samples})")
                    self.capture_sample(1, "collision")
                self._recover_from_collision()
                direction_index = (direction_index + 1) % len(danger_directions)
                time.sleep(1.5)
                continue

            # 定期采集安全样本
            current_time = time.time()
            if current_time - self.last_safe_capture_time >= self.safe_capture_interval:
                if self.safe_samples < self.target_samples:
                    self.capture_sample(0, "safe")
                    self.last_safe_capture_time = current_time

            # 朝障碍物方向飞行
            dx, dy = danger_directions[direction_index]
            vx = dx * self.fly_speed
            vy = dy * self.fly_speed

            self.client.moveByVelocityAsync(vx, vy, 0, 0.2)

            # 定期换方向
            if int(elapsed) % change_interval == 0 and int(elapsed) > 0:
                direction_index = (direction_index + 1) % len(danger_directions)

            # 显示进度
            if int(elapsed) % 2 == 0:
                current_pos = self.get_position()
                print(f"\r⏱ {int(elapsed)}s | 位置:({current_pos.x_val:.1f},{current_pos.y_val:.1f}) | "
                      f"✅ 安全: {self.safe_samples}/{self.target_samples} | "
                      f"💥 危险: {self.danger_samples}/{self.target_samples}", end="")

            time.sleep(0.1)

        self.client.hoverAsync()

    def _recover_from_collision(self):
        """从碰撞中恢复"""
        try:
            self.client.cancelLastTask()
            time.sleep(0.3)

            pos = self.get_position()
            # 后退逃离
            backup_dist = 5
            # 随机方向后退
            angle = random.uniform(0, 2 * np.pi)
            new_x = pos.x_val - backup_dist * np.cos(angle)
            new_y = pos.y_val - backup_dist * np.sin(angle)

            # 使用位置控制飞回安全区域
            self.client.moveToPositionAsync(new_x, new_y, self.target_altitude, 3)
            time.sleep(2)
            self.client.hoverAsync()
            time.sleep(0.5)

        except Exception as e:
            print(f"⚠️ 恢复失败: {e}")

    def stop(self):
        """停止采集"""
        self.running = False
        self.client.hoverAsync()

    def land(self):
        """降落"""
        print("\n🛬 开始降落...")
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

    def get_stats(self):
        """获取采集统计"""
        return {
            'safe': self.safe_samples,
            'danger': self.danger_samples,
            'total': self.safe_samples + self.danger_samples
        }


def select_mode():
    """交互式选择飞行模式"""
    print("\n" + "=" * 50)
    print("    飞行模式选择")
    print("=" * 50)
    print("  1️⃣  螺旋飞行 - 螺旋向外，容易撞到障碍物")
    print("  2️⃣  随机飞行 - 飞向随机目标点，有碰撞风险")
    print("  3️⃣  折线飞行 - 快速折返，高碰撞率")
    print("  4️⃣  撞墙模式 - 专门朝障碍物飞行，最高碰撞率")
    print("=" * 50)

    while True:
        choice = input("\n请选择模式 (1-4): ").strip()
        if choice == '1':
            return 'spiral'
        elif choice == '2':
            return 'random'
        elif choice == '3':
            return 'zigzag'
        elif choice == '4':
            return 'obstacles'
        else:
            print("无效选择，请输入 1-4")


def main():
    """主函数"""
    print("\n" + "=" * 50)
    print("    🚁 自动碰撞数据采集")
    print("=" * 50)

    # 交互式选择模式
    mode = select_mode()

    # 设置参数
    duration = 180  # 默认3分钟
    target = 100    # 每类目标样本数

    try:
        duration_input = input(f"\n采集时长(秒) [默认{duration}]: ").strip()
        if duration_input:
            duration = int(duration_input)
    except ValueError:
        pass

    try:
        target_input = input(f"每类目标样本数 [默认{target}]: ").strip()
        if target_input:
            target = int(target_input)
    except ValueError:
        pass

    print("\n" + "=" * 50)
    print("    配置确认")
    print("=" * 50)
    mode_names = {
        'spiral': '螺旋飞行',
        'random': '随机飞行',
        'zigzag': '折线飞行',
        'obstacles': '撞墙模式'
    }
    print(f"  飞行模式: {mode_names.get(mode, mode)}")
    print(f"  采集时长: {duration}秒")
    print(f"  目标样本: 每类 {target} 个")
    print("=" * 50)

    collector = AutoCollisionCollector(target_samples=target)

    try:
        collector.connect()
        collector.takeoff(height=5)

        if mode == 'spiral':
            collector.fly_spiral(duration=duration)
        elif mode == 'random':
            collector.fly_random(duration=duration)
        elif mode == 'zigzag':
            collector.fly_zigzag(duration=duration)
        elif mode == 'obstacles':
            collector.fly_toward_obstacles(duration=duration)

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    finally:
        collector.stop()
        collector.land()

        stats = collector.get_stats()
        print(f"\n{'=' * 50}")
        print("📊 采集完成！")
        print(f"   安全样本: {stats['safe']}")
        print(f"   危险样本: {stats['danger']}")
        print(f"   总计: {stats['total']}")
        print(f"   数据保存: {collector.labels_file}")
        print("=" * 50)


if __name__ == "__main__":
    main()
