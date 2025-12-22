import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import threading


class HandDemoMujoco3:
    """兼容 MuJoCo 3.x 的手部演示"""

    def __init__(self, model_path='left_hand.xml'):
        try:
            # 加载模型
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)

            print("=" * 60)
            print("✅ 手部模型加载成功")
            print(f"📊 执行器数量: {self.model.nu}")
            print(f"📊 关节数量: {self.model.njnt}")
            print(f"📊 仿真时间步: {self.model.opt.timestep:.4f}秒")
            print("=" * 60)

            # 创建预设姿态
            self._create_preset_poses()

            # 初始化状态
            self.current_pose_idx = 0
            self.animating = False
            self.animation_start = 0
            self.animation_duration = 1.5
            self.start_values = None
            self.target_values = None

            # 添加暂停相关变量
            self.paused = False
            self.pause_start_time = 0
            self.total_pause_time = 0
            self.pause_lock = threading.Lock()

            # 添加用户控制变量
            self.manual_mode = False
            self.should_exit = False

            # 控制变量
            self.should_toggle_pause = False
            self.should_next_pose = False
            self.should_prev_pose = False
            self.should_toggle_mode = False
            self.should_restart = False

            print(f"🎭 创建了 {len(self.poses)} 种预设姿态")
            print("=" * 60)

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

    def _create_preset_poses(self):
        """创建预设姿态"""
        self.poses = {}

        # 张开手
        self.poses['张开手'] = {
            'values': np.zeros(self.model.nu),
            'emoji': '🤚',
            'description': '所有手指完全展开'
        }

        # 握拳
        self.poses['握拳'] = {
            'values': self._create_pose_fist(),
            'emoji': '✊',
            'description': '所有手指完全握紧'
        }

        # 圆柱体抓握
        self.poses['圆柱体抓握'] = {
            'values': self._create_pose_cylinder(),
            'emoji': '🫱',
            'description': '环绕抓握柱状物体'
        }

        # 剪刀手
        self.poses['剪刀手'] = {
            'values': self._create_pose_scissors(),
            'emoji': '✌️',
            'description': '食指和中指张开呈V形'
        }

        # OK手势
        self.poses['OK手势'] = {
            'values': self._create_pose_ok(),
            'emoji': '👌',
            'description': '拇指和食指形成圆圈'
        }

        # 指点
        self.poses['指点'] = {
            'values': self._create_pose_pointing(),
            'emoji': '👉',
            'description': '食指伸直，其他手指握起'
        }

        # 演示序列 (去掉了捏取动作)
        self.demo_sequence = [
            '张开手',
            '握拳',
            '圆柱体抓握',
            '剪刀手',
            'OK手势',
            '指点',
            '张开手'
        ]

    def _create_pose_fist(self):
        """创建握拳姿态"""
        values = np.zeros(self.model.nu)

        # 根据执行器数量调整姿态
        if self.model.nu >= 20:
            # 假设前20个执行器是：手腕(2) + 拇指(5) + 4个手指*3 + 小指额外(1)
            values[:20] = [
                0.0, 0.0,  # 手腕
                0.199, 0.354, 0.209, 0.698, 0.3,  # 拇指
                5, 5, 5,   # 食指
                5, 5, 5,  # 中指
                5, 5, 5,   # 无名指
                0.0, 5, 5, 5,   # 小指
            ]
        elif self.model.nu >= 10:
            # 简化的握拳姿态
            for i in range(self.model.nu):
                if i < 2:  # 前2个是手腕
                    values[i] = 0.0
                else:  # 其他是手指
                    values[i] = 0.8
        else:
            # 最小配置
            for i in range(self.model.nu):
                values[i] = 0.8 if i >= 2 else 0.0

        return values


    def _create_pose_cylinder(self):
        """创建圆柱体抓握姿态"""
        values = np.zeros(self.model.nu)

        if self.model.nu >= 20:
            values[:20] = [
                0.0, 0.0,  # 手腕
                0.3, 0.5, 0.0, 0.4, 0.6,  # 拇指
                0.1, 0.6, 0.6,  # 食指
                0.1, 0.6, 0.6,  # 中指
                0.1, 0.6, 0.6,  # 无名指
                0.1, 0.6, 0.6, 0.6  # 小指
            ]
        elif self.model.nu >= 3:
            # 所有手指中等弯曲
            for i in range(self.model.nu):
                if i < 2:  # 手腕
                    values[i] = 0.0
                else:  # 手指
                    values[i] = 0.5
        else:
            # 最小配置
            for i in range(self.model.nu):
                values[i] = 0.5 if i >= 2 else 0.0

        return values

    def _create_pose_scissors(self):
        """创建剪刀手姿态"""
        values = np.zeros(self.model.nu)

        if self.model.nu >= 20:
            values[:20] = [
                0.0, 10,  # 手腕
                0.2, 0.4, 0.0, 0.3, 0.2,  # 拇指
                0.0, 0.0, 0.0,  # 食指
                0.0, 0.0, 0.0,  # 中指
                0.7, 5, 5,  # 无名指
                0.0, 1, 5, 5  # 小指
            ]
        elif self.model.nu >= 7:
            # 简化的剪刀手：假设第3-4个是食指和中指，其他手指弯曲
            for i in range(self.model.nu):
                if i in [3, 4]:  # 食指和中指
                    values[i] = 0.3
                elif i >= 2:  # 其他手指
                    values[i] = 0.7
                else:  # 手腕
                    values[i] = 0.0
        else:
            # 最小配置
            for i in range(self.model.nu):
                values[i] = 0.3 if i in [3, 4] else 0.7 if i >= 2 else 0.0

        return values

    def _create_pose_ok(self):
        """创建OK手势"""
        values = np.zeros(self.model.nu)

        if self.model.nu >= 20:
            values[:20] = [
                0.0, 0.0,  # 手腕
                -0.178, 1.22, 0.134, 0.698, 0.361, # 拇指
                0.2, 1.57, 1.3, # 食指
                0.0, 2, 0.2,  # 中指
                0.0, 1, 0.2,  # 无名指
                0.0, -0.5, 0.2, 0.2  # 小指
            ]
        elif self.model.nu >= 5:
            # 简化的OK手势
            for i in range(self.model.nu):
                if i == 2:  # 拇指
                    values[i] = 0.6
                elif i == 3:  # 食指
                    values[i] = 0.8
                elif i >= 4:  # 其他手指
                    values[i] = 0.2
                else:  # 手腕
                    values[i] = 0.0
        else:
            # 最小配置
            for i in range(self.model.nu):
                values[i] = 0.6 if i == 2 else (0.8 if i == 3 else 0.0)

        return values

    def _create_pose_pointing(self):
        """创建指点姿态"""
        values = np.zeros(self.model.nu)

        if self.model.nu >= 20:
            values[:20] = [
                0.0, 0.0,  # 手腕
                0.2, 0.3, 0.0, 0.2, 0.3,  # 拇指
                0.0, 0.0, 0.0,  # 食指
                5, 5, 5,  # 中指
                5, 5, 5,  # 无名指
                0.0, 5, 5, 5,   # 小指
            ]
        elif self.model.nu >= 4:
            # 简化的指点：假设第3个是食指
            for i in range(self.model.nu):
                if i == 3:  # 食指
                    values[i] = 0.0
                elif i >= 2:  # 其他手指
                    values[i] = 0.8
                else:  # 手腕
                    values[i] = 0.0
        else:
            # 最小配置
            for i in range(self.model.nu):
                values[i] = 0.0 if i == 3 else (0.8 if i >= 2 else 0.0)

        return values

    def start_animation(self, pose_name):
        """开始动画到指定姿态"""
        if pose_name not in self.poses:
            print(f"❌ 未知姿态: {pose_name}")
            return False

        pose_info = self.poses[pose_name]
        self.start_values = self.data.ctrl.copy()
        self.target_values = pose_info['values']
        self.animation_start = time.time()
        self.animating = True

        # 显示姿态信息
        progress = (self.current_pose_idx + 1) / len(self.demo_sequence) * 100
        sys.stdout.write("\r")
        sys.stdout.write(f"{pose_info['emoji']} [{pose_name:10s}] ")
        sys.stdout.write(f"进度: {progress:5.1f}% - {pose_info['description']}")
        sys.stdout.flush()

        return True

    def update_animation(self):
        """更新动画状态"""
        if not self.animating:
            return False

        elapsed = time.time() - self.animation_start
        t = min(elapsed / self.animation_duration, 1.0)

        # 缓动函数（ease in-out）
        if t < 0.5:
            t_eased = 2 * t * t
        else:
            t_eased = -1 + (4 - 2 * t) * t

        # 插值计算
        current_values = self.start_values + (self.target_values - self.start_values) * t_eased
        self.data.ctrl[:] = current_values

        # 检查动画是否完成
        if elapsed >= self.animation_duration:
            self.animating = False
            return True

        return False

    def toggle_pause(self):
        """切换暂停状态"""
        with self.pause_lock:
            if self.paused:
                # 从暂停状态恢复
                self.paused = False
                pause_duration = time.time() - self.pause_start_time
                self.total_pause_time += pause_duration
                print(f"\n▶️  演示继续 (暂停了 {pause_duration:.1f} 秒)")
            else:
                # 进入暂停状态
                self.paused = True
                self.pause_start_time = time.time()
                print(f"\n⏸️  演示暂停")

                # 显示当前状态信息
                current_pose = self.demo_sequence[self.current_pose_idx]
                pose_info = self.poses[current_pose]
                progress = (self.current_pose_idx + 1) / len(self.demo_sequence) * 100
                print(f"  当前姿态: {current_pose} - {pose_info['description']}")
                print(f"  进度: {progress:.1f}%")

    def next_pose(self):
        """切换到下一个姿态"""
        if not self.paused and not self.manual_mode:
            # 只有在自动模式下才能手动切换
            return False

        with self.pause_lock:
            if self.paused:
                # 从暂停状态恢复但切换到下一个姿态
                self.paused = False
                pause_duration = time.time() - self.pause_start_time
                self.total_pause_time += pause_duration

            self.current_pose_idx = (self.current_pose_idx + 1) % len(self.demo_sequence)
            next_pose = self.demo_sequence[self.current_pose_idx]
            self.start_animation(next_pose)

            print(f"\n⏭️  切换到下一个姿态: {next_pose}")
            return True

    def previous_pose(self):
        """切换到上一个姿态"""
        if not self.paused and not self.manual_mode:
            # 只有在自动模式下才能手动切换
            return False

        with self.pause_lock:
            if self.paused:
                # 从暂停状态恢复但切换到上一个姿态
                self.paused = False
                pause_duration = time.time() - self.pause_start_time
                self.total_pause_time += pause_duration

            self.current_pose_idx = (self.current_pose_idx - 1) % len(self.demo_sequence)
            previous_pose = self.demo_sequence[self.current_pose_idx]
            self.start_animation(previous_pose)

            print(f"\n⏮️  切换到上一个姿态: {previous_pose}")
            return True

    def toggle_manual_mode(self):
        """切换手动模式"""
        self.manual_mode = not self.manual_mode
        if self.manual_mode:
            print(f"\n🎮 切换到手动模式")
            print("   使用控制台命令控制")
        else:
            print(f"\n🤖 切换到自动模式")
        return self.manual_mode

    def print_controls(self):
        """打印控制说明"""
        print("\n" + "=" * 60)
        print("🎮 控制说明 (在终端中输入命令):")
        print("  pause: 暂停/继续演示")
        print("  next: 下一个姿态")
        print("  prev: 上一个姿态")
        print("  mode: 切换手动/自动模式")
        print("  restart: 重新开始演示")
        print("  help: 显示控制说明")
        print("  quit: 退出演示")
        print("=" * 60)

    def process_command(self, command):
        """处理控制台命令"""
        command = command.strip().lower()

        if command == 'pause':
            self.should_toggle_pause = True
        elif command == 'next':
            self.should_next_pose = True
        elif command == 'prev':
            self.should_prev_pose = True
        elif command == 'mode':
            self.should_toggle_mode = True
        elif command == 'restart':
            self.should_restart = True
        elif command == 'help':
            self.print_controls()
        elif command == 'quit':
            self.should_exit = True
            print("\n👋 正在退出演示...")
        elif command:
            print(f"❌ 未知命令: {command}")
            print("输入 'help' 查看可用命令")

    def run_demo(self):
        """运行演示"""
        print("\n" + "=" * 60)
        print("🤖 手部抓握姿态全自动演示 (MuJoCo 3.x 兼容版)")
        print("=" * 60)
        print(f"🎬 演示序列: {len(self.demo_sequence)} 个姿态")
        print(f"⏱️  每个姿态保持: 5.0秒")
        print(f"🎥 动画过渡: {self.animation_duration}秒")
        self.print_controls()
        print("=" * 60)
        print("\n💡 提示: 在终端中输入命令控制演示")
        print("=" * 60)

        # 设置初始姿态
        initial_pose = self.demo_sequence[0]
        self.data.ctrl[:] = self.poses[initial_pose]['values']

        last_change = time.time()
        hold_duration = 5.0  # 每个姿态保持5秒

        # 启动一个线程来处理用户输入
        def input_thread():
            """处理用户输入的线程"""
            while not self.should_exit:
                try:
                    command = input("\n> ").strip()
                    if command:
                        self.process_command(command)
                except (EOFError, KeyboardInterrupt):
                    self.should_exit = True
                    break

        # 启动输入线程
        input_handler = threading.Thread(target=input_thread, daemon=True)
        input_handler.start()

        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # 设置视角
                viewer.cam.azimuth = 45
                viewer.cam.elevation = -20
                viewer.cam.distance = 0.8
                viewer.cam.lookat[:] = [0.0, 0.0, 0.1]

                print("\n演示开始...\n")

                # 显示第一个姿态
                pose_info = self.poses[initial_pose]
                sys.stdout.write(f"\r{pose_info['emoji']} [{initial_pose:10s}] ")
                sys.stdout.write(f"进度: {0.0:5.1f}% - {pose_info['description']}")
                sys.stdout.flush()

                while viewer.is_running() and not self.should_exit:
                    current_time = time.time()

                    # 处理命令
                    if self.should_toggle_pause:
                        self.toggle_pause()
                        self.should_toggle_pause = False

                    if self.should_next_pose:
                        self.next_pose()
                        if not self.manual_mode:
                            self.toggle_manual_mode()
                        last_change = current_time
                        self.should_next_pose = False

                    if self.should_prev_pose:
                        self.previous_pose()
                        if not self.manual_mode:
                            self.toggle_manual_mode()
                        last_change = current_time
                        self.should_prev_pose = False

                    if self.should_toggle_mode:
                        self.toggle_manual_mode()
                        self.should_toggle_mode = False

                    if self.should_restart:
                        self.current_pose_idx = 0
                        self.start_animation(self.demo_sequence[0])
                        self.paused = False
                        self.total_pause_time = 0
                        last_change = current_time
                        print(f"\n🔄 重新开始演示")
                        self.should_restart = False

                    # 如果暂停，跳过更新
                    if self.paused:
                        viewer.sync()
                        time.sleep(0.01)  # 减少CPU使用率
                        continue

                    # 在手动模式下，不自动切换姿态
                    if not self.manual_mode:
                        # 更新动画
                        self.update_animation()

                        # 检查是否需要切换到下一个姿态
                        if not self.animating and (current_time - last_change > hold_duration):
                            self.current_pose_idx = (self.current_pose_idx + 1) % len(self.demo_sequence)
                            next_pose = self.demo_sequence[self.current_pose_idx]

                            if self.start_animation(next_pose):
                                last_change = current_time
                    else:
                        # 手动模式下，只更新动画
                        self.update_animation()

                    # 运行仿真
                    mujoco.mj_step(self.model, self.data)

                    # 同步可视化
                    viewer.sync()

                    # 帧率控制
                    time.sleep(self.model.opt.timestep)

        except KeyboardInterrupt:
            print("\n\n👋 演示被用户中断")
        except Exception as e:
            print(f"\n❌ 运行时错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.should_exit = True
            print("\n" + "=" * 60)
            print("🎉 演示结束")
            print(f"⏱️  总暂停时间: {self.total_pause_time:.1f}秒")
            print("=" * 60)


def main():
    """主函数"""
    print("正在初始化手部模型演示...")

    try:
        demo = HandDemoMujoco3('left_hand.xml')
        demo.run_demo()
    except FileNotFoundError:
        print("❌ 找不到模型文件 'left_hand.xml'")
        print("请确保文件在当前目录中")
        print("当前目录内容:")
        import os
        for file in os.listdir('.'):
            if file.endswith('.xml'):
                print(f"  - {file}")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()