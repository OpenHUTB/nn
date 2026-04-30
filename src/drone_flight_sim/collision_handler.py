# collision_handler.py
"""碰撞检测和处理模块

本模块负责检测无人机飞行过程中的碰撞事件，
区分地面接触和严重碰撞，并记录碰撞统计信息。
提供碰撞后自动恢复和手动接管机制。
"""

# 导入 time 模块，用于获取当前时间实现碰撞冷却
import time
# 导入随机模块，用于随机避障方向
import random
# 从 config 模块导入飞行配置和地面物体列表
from config import FlightConfig, GROUND_OBJECTS


class CollisionHandler:
    """碰撞检测处理器类

    负责与 AirSim 仿真器交互，检测并处理碰撞事件。
    使用冷却机制避免短时间内重复触发同一碰撞。
    提供碰撞后自动恢复和手动接管功能。
    """

    def __init__(self, client):
        """初始化碰撞处理器

        参数:
            client: AirSim 多旋翼无人机客户端对象
        """
        # 保存 AirSim 客户端引用，用于获取碰撞信息
        self.client = client
        # 初始化碰撞计数器，记录累计碰撞次数
        self.collision_count = 0
        # 记录上一次碰撞发生的时间戳（秒）
        self.last_collision_time = 0
        # 标记当前是否处于碰撞状态
        self.is_collided = False
        # 自动恢复尝试次数
        self.auto_recovery_attempts = 0
        # 最大自动恢复尝试次数（从配置读取）
        self.max_auto_recovery_attempts = FlightConfig.MAX_AUTO_RECOVERY_ATTEMPTS

    def check_collision(self):
        """检测碰撞事件

        从 AirSim 获取碰撞信息，判断是否为严重碰撞。
        地面接触和冷却期内的碰撞会被忽略。

        返回:
            tuple: (是否严重碰撞, 碰撞信息对象)
                   严重碰撞返回 (True, collision_info)
                   非严重或无碰撞返回 (False, None)
        """
        # 调用 AirSim API 获取当前碰撞信息
        collision_info = self.client.simGetCollisionInfo()

        # 检查是否真的发生了碰撞（has_collided 为 False 表示无碰撞）
        if not collision_info.has_collided:
            # 没有碰撞，返回 False
            return False, None

        # 发生了碰撞，获取当前时间戳
        current_time = time.time()
        # 检查碰撞冷却：如果距离上次碰撞时间小于阈值，则忽略此次碰撞
        if current_time - self.last_collision_time < FlightConfig.COLLISION_COOLDOWN:
            # 处于冷却期内，忽略此次碰撞
            return False, None

        # 更新碰撞时间记录
        self.last_collision_time = current_time
        # 累加碰撞计数器
        self.collision_count += 1

        # 获取无人机当前位置
        drone_pos = self.client.getMultirotorState().kinematics_estimated.position
        # 计算当前高度（Z 轴向下为正，所以取负值）
        current_height = -drone_pos.z_val

        # 判断是否为地面接触
        # 条件1：当前高度小于地面阈值
        # 条件2：碰撞物体的名称包含地面相关关键词
        is_ground = (
            current_height < FlightConfig.GROUND_HEIGHT_THRESHOLD or
            any(keyword in collision_info.object_name for keyword in GROUND_OBJECTS)
        )

        # 如果是地面接触，打印提示信息并忽略
        if is_ground:
            print(f"⚠️  检测到与 {collision_info.object_name} 接触（高度: {current_height:.2f}m），忽略")
            return False, None

        # 判定为严重碰撞，打印详细的碰撞信息
        print(f"\n💥 严重碰撞发生！")
        # 打印碰撞位置坐标
        print(f"   碰撞位置: ({collision_info.position.x_val:.2f}, "
              f"{collision_info.position.y_val:.2f}, {collision_info.position.z_val:.2f})")
        # 打印碰撞物体的名称
        print(f"   碰撞物体: {collision_info.object_name}")
        # 打印碰撞时的高度
        print(f"   当前高度: {current_height:.2f}m")
        # 打印累计碰撞次数
        print(f"   碰撞次数: {self.collision_count}")

        # 返回严重碰撞标志和碰撞信息对象
        return True, collision_info

    def auto_recover(self):
        """自动恢复碰撞

        尝试通过后退和上升来脱离碰撞状态。
        如果自动恢复失败，返回 False 表示需要手动接管。

        返回:
            bool: 自动恢复成功返回 True，失败返回 False
        """
        self.auto_recovery_attempts += 1

        if self.auto_recovery_attempts > self.max_auto_recovery_attempts:
            # 超过最大自动恢复次数，需要手动接管
            return False

        print(f"\n🔧 尝试自动恢复 ({self.auto_recovery_attempts}/{self.max_auto_recovery_attempts})...")

        try:
            # 1. 取消当前任务
            self.client.cancelLastTask()
            time.sleep(0.5)

            # 2. 获取当前位置
            pos = self.client.getMultirotorState().kinematics_estimated.position

            # 3. 后退一段距离（随机选择后退方向）
            backward_distance = 3
            directions = [-1, 1]  # 后退方向：左后方或右后方
            direction = random.choice(directions)

            new_x = pos.x_val - backward_distance * 0.7
            new_y = pos.y_val + direction * backward_distance * 0.7
            new_z = pos.z_val - 2  # 上升 2 米

            print(f"   后退避障中...")
            self.client.moveToPositionAsync(new_x, new_y, new_z, FlightConfig.FLIGHT_VELOCITY)

            # 等待移动完成
            time.sleep(3)

            # 4. 悬停等待稳定
            self.client.hoverAsync().join()
            time.sleep(1)

            # 5. 检查是否脱离碰撞
            collision_info = self.client.simGetCollisionInfo()
            if not collision_info.has_collided:
                print(f"✅ 自动恢复成功！")
                self.auto_recovery_attempts = 0
                return True
            else:
                print(f"⚠️  自动恢复后仍处于碰撞状态")
                return False

        except Exception as e:
            print(f"❌ 自动恢复失败: {e}")
            return False

    def request_manual_control(self):
        """请求手动接管控制

        打印手动接管提示信息，提示用户如何手动解决碰撞。

        返回:
            bool: 始终返回 True，表示需要手动接管
        """
        print(f"\n{'=' * 50}")
        print(f"🚨 自动恢复失败，需要手动接管！")
        print(f"{'=' * 50}")
        print(f"""
📋 手动接管说明:
   碰撞后无人机可能处于卡住状态，请使用键盘控制：
   
   键盘控制说明:
   - W/↑     : 前进
   - S/↓     : 后退
   - A       : 向左移动
   - D       : 向右移动
   - Q       : 上升
   - E       : 下降
   - 空格    : 悬停
   - L       : 执行降落
   - ESC     : 紧急停止并退出

💡 提示: 先按 Q 上升脱离碰撞，然后按 L 降落
""")
        print(f"{'=' * 50}\n")
        return True

    def reset_collision_state(self):
        """重置碰撞状态

        将碰撞状态标记重置为 False，
        通常在任务重新开始时调用。
        """
        # 重置碰撞状态标志
        self.is_collided = False
        # 重置自动恢复计数器
        self.auto_recovery_attempts = 0
