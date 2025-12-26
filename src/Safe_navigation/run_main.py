#!/usr/bin/env python3
"""
AirSimNH 无人车仿真控制脚本 - 强力防碰撞修复版本
"""

import airsim
import time
import numpy as np
import cv2
import json
import os
from datetime import datetime
from collections import deque
import math


class AirSimNHCarSimulator:
    """AirSim无人车仿真主类"""

    def __init__(self, ip="127.0.0.1", port=41451, vehicle_name="PhysXCar"):
        self.ip = ip
        self.port = port
        self.vehicle_name = vehicle_name
        self.client = None
        self.is_connected = False
        self.is_api_control_enabled = False

        # 车辆状态跟踪
        self.initial_position = None
        self.initial_yaw = None
        self.path_history = []

        # 碰撞计数器
        self.collision_count = 0
        self.last_collision_state = False

        # 创建数据保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = f"simulation_data_{timestamp}"
        os.makedirs(self.data_dir, exist_ok=True)

        print(f"数据保存目录: {self.data_dir}")

    def connect(self):
        """连接到AirSim仿真器"""
        try:
            print(f"正在连接到AirSim仿真器 {self.ip}:{self.port}...")
            self.client = airsim.CarClient(ip=self.ip, port=self.port)
            self.client.confirmConnection()

            vehicles = self.client.listVehicles()
            if self.vehicle_name not in vehicles:
                print(f"警告: 车辆 '{self.vehicle_name}' 未找到，可用车辆: {vehicles}")
                if vehicles:
                    self.vehicle_name = vehicles[0]
                    print(f"使用车辆: {self.vehicle_name}")

            self.is_connected = True
            print("✓ 成功连接到AirSim仿真器！")

            self.initial_position = self.get_position()
            self.initial_yaw = self.get_yaw()
            print(
                f"初始位置: x={self.initial_position['x']:.3f}, y={self.initial_position['y']:.3f}, z={self.initial_position['z']:.3f}")
            print(f"初始偏航角: {self.initial_yaw:.2f}°")

            return True

        except Exception as e:
            print(f"✗ 连接失败: {e}")
            print("请确保AirSimNH环境正在运行")
            return False

    def get_position(self):
        """获取车辆位置"""
        try:
            kinematics = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
            return {
                "x": kinematics.position.x_val,
                "y": kinematics.position.y_val,
                "z": kinematics.position.z_val
            }
        except:
            return {"x": 0, "y": 0, "z": 0}

    def get_yaw(self):
        """获取车辆偏航角"""
        try:
            kinematics = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
            orientation = kinematics.orientation

            q0, q1, q2, q3 = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
            siny_cosp = 2 * (q0 * q3 + q1 * q2)
            cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            yaw_deg = math.degrees(yaw)

            if yaw_deg < 0:
                yaw_deg += 360

            return yaw_deg
        except:
            return 0.0

    def enable_api_control(self, enable=True):
        """启用/禁用API控制"""
        try:
            self.client.enableApiControl(enable, vehicle_name=self.vehicle_name)
            self.is_api_control_enabled = enable

            if enable:
                print("✓ API控制已启用")
                controls = airsim.CarControls()
                controls.throttle = 0
                controls.steering = 0
                controls.brake = 0
                self.client.setCarControls(controls, vehicle_name=self.vehicle_name)
            else:
                print("✓ API控制已禁用")

            return True
        except Exception as e:
            print(f"✗ API控制设置失败: {e}")
            return False

    def get_vehicle_state(self):
        """获取车辆状态 - 修复了collision_count错误"""
        try:
            state = self.client.getCarState(vehicle_name=self.vehicle_name)
            kinematics = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
            yaw = self.get_yaw()

            current_position = {
                "x": kinematics.position.x_val,
                "y": kinematics.position.y_val,
                "z": kinematics.position.z_val
            }

            # 检查碰撞状态并更新计数器
            current_collision = state.collision.has_collided
            if current_collision and not self.last_collision_state:
                self.collision_count += 1
                print(f"\n!!! 检测到碰撞！碰撞次数: {self.collision_count}")
            self.last_collision_state = current_collision

            # 记录路径
            self.path_history.append({
                "timestamp": time.time(),
                "position": current_position.copy(),
                "yaw": yaw,
                "speed": state.speed
            })

            if len(self.path_history) > 200:
                self.path_history.pop(0)

            state_info = {
                "timestamp": time.time(),
                "speed_kmh": state.speed,
                "speed_ms": state.speed / 3.6,
                "position": current_position,
                "yaw": yaw,
                "rpm": state.rpm,
                "max_rpm": state.maxrpm,
                "gear": state.gear,
                "handbrake": state.handbrake,
                "collision": current_collision,
                "collision_count": self.collision_count  # 使用我们自己的计数器
            }

            return state_info
        except Exception as e:
            print(f"获取车辆状态失败: {e}")
            return None

    def calculate_lateral_offset(self, current_position):
        """计算横向偏移（改进版本）"""
        if self.initial_position is None:
            return 0.0

        # 计算绝对偏移
        absolute_offset = current_position["y"] - self.initial_position["y"]

        return absolute_offset

    def safe_control_demo(self, duration=30):
        """
        安全控制演示：主动避免右侧碰撞
        使用强力左转修正策略

        参数:
            duration: 演示总时长（秒）
        """
        if not self.is_connected or not self.is_api_control_enabled:
            print("错误: 请先连接并启用API控制")
            return False

        print(f"\n开始安全控制演示 ({duration}秒)...")
        print("策略: 强力左转修正，防止向右偏移和碰撞")

        start_time = time.time()
        controls = airsim.CarControls()

        # 控制参数
        target_speed_kmh = 18
        base_throttle = 0.45

        # 偏移监控
        max_right_offset = 0
        offset_history = deque(maxlen=5)

        # 状态跟踪
        emergency_left_turn = False
        emergency_turn_start_time = 0
        last_good_position = self.initial_position.copy()

        # 强力左转参数
        strong_left_steering = 0.35  # 强力左转角度
        moderate_left_steering = 0.2  # 中等左转角度
        slight_left_steering = 0.1  # 轻微左转角度

        try:
            while time.time() - start_time < duration:
                elapsed = time.time() - start_time

                # 获取当前状态
                state = self.get_vehicle_state()
                if not state:
                    print("  ! 获取状态失败，继续尝试...")
                    time.sleep(0.1)
                    continue

                current_speed = state['speed_kmh']
                current_position = state['position']
                current_yaw = state['yaw']

                # 计算偏移
                absolute_offset = self.calculate_lateral_offset(current_position)

                # 更新历史
                offset_history.append(absolute_offset)

                # 更新最大右偏移
                if absolute_offset > max_right_offset:
                    max_right_offset = absolute_offset

                # 计算偏移趋势
                offset_trend = 0
                if len(offset_history) >= 3:
                    offset_trend = sum(offset_history) / len(offset_history)

                # 1. 紧急情况检测和处理
                collision_detected = state.get('collision', False)

                # 如果已经发生碰撞
                if collision_detected:
                    print(f"\n!!! 发生碰撞！执行紧急避障程序")
                    # 紧急刹车+强力左转
                    controls = airsim.CarControls()
                    controls.throttle = 0
                    controls.brake = 1.0
                    controls.steering = -strong_left_steering  # 强力左转摆脱
                    self.client.setCarControls(controls, vehicle_name=self.vehicle_name)
                    time.sleep(1.5)  # 紧急避障1.5秒

                    # 尝试回退到安全位置
                    print("  尝试回到安全位置...")
                    controls.brake = 0
                    controls.throttle = -0.3  # 倒车
                    controls.steering = 0.1  # 稍微右转
                    self.client.setCarControls(controls, vehicle_name=self.vehicle_name)
                    time.sleep(2.0)

                    controls.throttle = 0
                    controls.brake = 0.5
                    self.client.setCarControls(controls, vehicle_name=self.vehicle_name)
                    time.sleep(1.0)

                    # 重置初始位置为当前位置
                    self.initial_position = self.get_position()
                    print(f"  重置初始位置: y={self.initial_position['y']:.3f}")
                    continue

                # 2. 基于偏移量的强力修正逻辑
                base_steering = 0.0
                collision_risk = False

                # 强力修正逻辑：基于偏移量决定左转力度
                if absolute_offset > 0.15:  # 向右偏移超过15厘米 - 紧急情况！
                    collision_risk = True
                    base_steering = -strong_left_steering * 1.2  # 超强力左转
                    print(f"\n!!! 紧急！向右偏移{absolute_offset:.3f}米，执行超强力左转！")
                    emergency_left_turn = True
                    emergency_turn_start_time = elapsed

                elif absolute_offset > 0.10:  # 向右偏移超过10厘米
                    collision_risk = True
                    base_steering = -strong_left_steering  # 强力左转
                    print(f"  !! 危险！向右偏移{absolute_offset:.3f}米，执行强力左转")
                    if not emergency_left_turn:
                        emergency_left_turn = True
                        emergency_turn_start_time = elapsed

                elif absolute_offset > 0.05:  # 向右偏移超过5厘米
                    collision_risk = True
                    base_steering = -moderate_left_steering  # 中等左转
                    print(f"  ! 警告！向右偏移{absolute_offset:.3f}米，执行中等左转")
                    if emergency_left_turn:
                        # 检查是否可以退出紧急模式
                        if elapsed - emergency_turn_start_time > 3.0 and absolute_offset < 0.03:
                            emergency_left_turn = False
                            print("  ✓ 危险解除")

                elif absolute_offset > 0.02:  # 向右偏移超过2厘米
                    base_steering = -slight_left_steering  # 轻微左转
                    if elapsed % 2.0 < 0.1:  # 每2秒显示一次
                        print(f"  > 注意：向右偏移{absolute_offset:.3f}米，轻微左转修正")

                elif absolute_offset < -0.05:  # 向左偏移超过5厘米
                    base_steering = 0.05  # 轻微右转修正
                    if elapsed % 2.0 < 0.1:
                        print(f"  < 注意：向左偏移{abs(absolute_offset):.3f}米，轻微右转修正")

                else:  # 偏移在安全范围内
                    base_steering = -0.03  # 始终轻微左倾，预防向右偏移
                    emergency_left_turn = False

                # 3. 基于趋势的额外修正
                if offset_trend > 0.01:  # 偏移趋势向右
                    base_steering -= 0.08  # 增加左转力度
                    if elapsed % 1.0 < 0.1:
                        print(f"  ↗ 趋势向右，增加左转修正")

                # 4. 油门控制策略
                if collision_risk or emergency_left_turn:
                    # 危险情况下减速
                    controls.throttle = base_throttle * 0.3
                    controls.brake = 0.1  # 轻微刹车
                else:
                    # 正常情况下的速度控制
                    if current_speed < target_speed_kmh * 0.7:
                        controls.throttle = base_throttle
                        controls.brake = 0
                    elif current_speed < target_speed_kmh:
                        controls.throttle = base_throttle * 0.6
                        controls.brake = 0
                    else:
                        controls.throttle = base_throttle * 0.4
                        controls.brake = 0.05  # 轻微刹车控制速度

                # 5. 阶段控制（根据时间调整策略）
                if elapsed < 6.0:  # 起步阶段（6秒）
                    controls.throttle = base_throttle * 0.7
                    base_steering = -0.05  # 轻微左转起步

                elif elapsed < 18.0:  # 主要行驶阶段（12秒）
                    # 保持主动左转修正
                    pass

                elif elapsed < 24.0:  # 测试阶段（6秒） - 尝试轻微右转但受安全约束
                    # 只有在绝对安全时才允许轻微右转
                    if absolute_offset < 0.01 and not collision_risk and not emergency_left_turn:
                        test_steering = 0.04  # 非常轻微的右转
                        base_steering = test_steering
                        if elapsed % 2.0 < 0.1:
                            print("  → 安全条件下测试轻微右转")
                    else:
                        if elapsed % 2.0 < 0.1:
                            print("  × 条件不满足，取消右转测试，保持左转")

                else:  # 减速停止阶段（最后6秒）
                    # 逐渐减速
                    stop_progress = (elapsed - 24.0) / 6.0
                    controls.throttle = max(0, base_throttle * (1.0 - stop_progress))

                    if current_speed > 12:
                        controls.brake = 0.4
                    elif current_speed > 6:
                        controls.brake = 0.2
                    else:
                        controls.brake = 0.1

                    # 停止阶段更积极的左转，确保停在安全位置
                    base_steering = -0.08

                # 6. 应用控制
                steering = max(-1.0, min(1.0, base_steering))
                controls.steering = steering

                # 发送控制命令
                self.client.setCarControls(controls, vehicle_name=self.vehicle_name)

                # 7. 显示状态
                status_symbol = "✓"
                if collision_risk:
                    status_symbol = "⚠️"
                if emergency_left_turn:
                    status_symbol = "🚨"
                if collision_detected:
                    status_symbol = "💥"

                status_line = (f"{status_symbol} 速度: {current_speed:5.1f} km/h | "
                               f"转向: {controls.steering:+.3f} | "
                               f"油门: {controls.throttle:.2f} | "
                               f"刹车: {controls.brake:.2f} | "
                               f"偏航: {current_yaw:6.1f}° | "
                               f"偏移: {absolute_offset:+.3f}m | "
                               f"最大偏移: {max_right_offset:+.3f}m")

                print(f"\r{status_line}", end="")

                # 8. 慢速采集数据
                if elapsed % 0.5 < 0.05:  # 每0.5秒采集一次
                    try:
                        # 简单状态检查
                        pass
                    except:
                        pass

                time.sleep(0.08)  # 12.5Hz控制频率

                # 9. 保存最后一个好位置
                if not collision_risk and absolute_offset < 0.03:
                    last_good_position = current_position.copy()

            print("\n✓ 安全控制演示完成")

            # 最终分析
            print(f"\n最终统计:")
            print(f"最大向右偏移: {max_right_offset:.3f}米")
            print(f"碰撞次数: {self.collision_count}")
            print(f"路径点数量: {len(self.path_history)}")

            if max_right_offset > 0.15:
                print("  ⚠️⚠️⚠️  严重警告：车辆明显向右偏移，碰撞风险高！")
            elif max_right_offset > 0.08:
                print("  ⚠️⚠️  警告：车辆有向右偏移趋势")
            elif max_right_offset > 0.03:
                print("  ⚠️  注意：车辆轻微向右偏移")
            else:
                print("  ✓ 优秀：车辆保持在安全范围内")

            if self.collision_count > 0:
                print(f"  ⚠️  发生碰撞: {self.collision_count}次")
            else:
                print("  ✓ 安全：无碰撞发生")

            return True

        except KeyboardInterrupt:
            print("\n\n演示被用户中断")
            return False
        except Exception as e:
            print(f"\n✗ 控制演示出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_simulation_data(self):
        """保存仿真数据"""
        try:
            # 保存路径历史
            if self.path_history:
                path_file = f"{self.data_dir}/path_history.json"
                with open(path_file, 'w') as f:
                    json.dump(self.path_history, f, indent=2)
                print(f"✓ 路径历史已保存: {path_file}")

            # 保存统计数据
            stats = {
                "timestamp": datetime.now().isoformat(),
                "vehicle_name": self.vehicle_name,
                "collision_count": self.collision_count,
                "path_history_length": len(self.path_history),
                "initial_position": self.initial_position,
                "initial_yaw": self.initial_yaw
            }

            stats_file = f"{self.data_dir}/simulation_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)

            # 生成报告
            report_file = f"{self.data_dir}/report.txt"
            with open(report_file, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("AirSim无人车安全控制演示报告\n")
                f.write("强力防碰撞版本\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"演示时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"车辆名称: {self.vehicle_name}\n")
                f.write(f"碰撞次数: {self.collision_count}\n")
                f.write(f"路径点数量: {len(self.path_history)}\n")

                if self.path_history and len(self.path_history) > 10:
                    first_pos = self.path_history[0]['position']
                    last_pos = self.path_history[-1]['position']
                    y_offset = last_pos['y'] - first_pos['y']
                    f.write(f"最终横向偏移(Y轴): {y_offset:.3f}米\n")

                    # 分析偏移范围
                    y_values = [p['position']['y'] for p in self.path_history]
                    min_y = min(y_values)
                    max_y = max(y_values)
                    avg_y = sum(y_values) / len(y_values)

                    f.write(f"Y坐标范围: {min_y:.3f} 到 {max_y:.3f} 米\n")
                    f.write(f"平均Y坐标: {avg_y:.3f} 米\n")

                    if y_offset > 0.1:
                        f.write("结论: 车辆明显向右偏移，需要加强左转修正\n")
                    elif y_offset > 0.05:
                        f.write("结论: 车辆有向右偏移趋势\n")
                    elif y_offset > 0:
                        f.write("结论: 车辆轻微向右偏移\n")
                    elif y_offset < -0.05:
                        f.write("结论: 车辆向左偏移\n")
                    else:
                        f.write("结论: 车辆基本保持在车道中央\n")

            print(f"✓ 报告已保存: {report_file}")
            print(f"✓ 统计数据已保存: {stats_file}")
            return True

        except Exception as e:
            print(f"✗ 保存数据失败: {e}")
            return False

    def run_safe_demo(self, duration=30):
        """运行安全演示"""
        print("=" * 60)
        print("AirSimNH 无人车安全控制演示")
        print("强力防碰撞修复版本")
        print("=" * 60)

        # 连接仿真器
        if not self.connect():
            return False

        try:
            # 启用API控制
            if not self.enable_api_control(True):
                return False

            print("\n等待车辆稳定...")
            time.sleep(2)

            # 运行安全控制演示
            print("\n" + "=" * 60)
            print("开始安全控制演示")
            print("策略: 强力左转修正，主动防止向右偏移")
            print("=" * 60)

            success = self.safe_control_demo(duration)

            if success:
                print("\n" + "=" * 60)
                print("演示完成，保存数据...")
                print("=" * 60)
                self.save_simulation_data()

            return success

        finally:
            # 清理
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        print("\n正在清理资源...")

        # 停止车辆
        if self.is_api_control_enabled:
            controls = airsim.CarControls()
            controls.throttle = 0
            controls.brake = 1.0
            controls.steering = 0
            controls.handbrake = True
            try:
                self.client.setCarControls(controls, vehicle_name=self.vehicle_name)
                time.sleep(1)
            except:
                pass

            # 禁用API控制
            try:
                self.enable_api_control(False)
            except:
                pass

        print("✓ 清理完成")


def main():
    """主函数"""
    simulator = AirSimNHCarSimulator(
        ip="127.0.0.1",
        port=41451,
        vehicle_name="PhysXCar"
    )

    try:
        simulator.run_safe_demo(duration=30)

        print("\n" + "=" * 60)
        print("安全控制演示完成！")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n演示被用户中断")
        simulator.cleanup()
    except Exception as e:
        print(f"\n演示出错: {e}")
        import traceback
        traceback.print_exc()
        simulator.cleanup()


if __name__ == "__main__":
    main()