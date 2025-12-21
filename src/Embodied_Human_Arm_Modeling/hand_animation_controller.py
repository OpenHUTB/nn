# 保存为 hand_demo_mujoco3.py
import mujoco
import mujoco.viewer
import numpy as np
import time
import sys


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

            # 在 MuJoCo 3.x 中获取执行器名称的替代方法
            print("📋 执行器信息:")
            # 注意：MuJoCo 3.x 中获取执行器名称的方式不同
            # 这里我们只显示数量，不尝试获取名称

            # 创建预设姿态
            self._create_preset_poses()

            # 初始化状态
            self.current_pose_idx = 0
            self.animating = False
            self.animation_start = 0
            self.animation_duration = 1.5
            self.start_values = None
            self.target_values = None

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

        # 捏取
        self.poses['捏取'] = {
            'values': self._create_pose_pinch(),
            'emoji': '🤏',
            'description': '拇指和食指对捏'
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

        # 演示序列
        self.demo_sequence = [
            '张开手',
            '握拳',
            '捏取',
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
                0.0, 1.57, 0.0, 0.5, 0.3,  # 拇指
                1.57, 1.57, 1.57,  # 食指
                1.57, 1.57, 1.57,  # 中指
                1.57, 1.57, 1.57,  # 无名指
                0.0, 1.57, 1.57, 1.57  # 小指
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

    def _create_pose_pinch(self):
        """创建捏取姿态"""
        values = np.zeros(self.model.nu)

        if self.model.nu >= 20:
            values[:20] = [
                0.0, 0.0,  # 手腕
                0.5, 0.6, 0.0, 0.5, 0.8,  # 拇指
                0.2, 0.7, 0.7,  # 食指
                0.0, 0.2, 0.2,  # 中指
                0.0, 0.1, 0.1,  # 无名指
                0.0, 0.1, 0.1, 0.1  # 小指
            ]
        elif self.model.nu >= 5:
            # 简化的捏取：假设前5个执行器中，第2个是拇指，第3个是食指
            for i in range(self.model.nu):
                if i == 2:  # 拇指
                    values[i] = 0.5
                elif i == 3:  # 食指
                    values[i] = 0.7
                elif i >= 4:  # 其他手指
                    values[i] = 0.2
                else:  # 手腕
                    values[i] = 0.0
        else:
            # 最小配置
            for i in range(self.model.nu):
                values[i] = 0.5 if i == 2 else 0.0

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
                0.0, 0.0,  # 手腕
                0.2, 0.4, 0.0, 0.3, 0.2,  # 拇指
                0.0, 0.0, 0.0,  # 食指
                0.0, 0.0, 0.0,  # 中指
                0.7, 1.57, 1.57,  # 无名指
                0.0, 1.57, 1.57, 1.57  # 小指
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
                0.4, 0.6, 0.0, 0.5, 0.7,  # 拇指
                0.3, 0.7, 0.9,  # 食指
                0.0, 0.1, 0.2,  # 中指
                0.0, 0.1, 0.2,  # 无名指
                0.0, 0.1, 0.2, 0.2  # 小指
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
                1.57, 1.57, 1.57,  # 中指
                1.57, 1.57, 1.57,  # 无名指
                0.0, 1.57, 1.57, 1.57  # 小指
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

    def run_demo(self):
        """运行演示"""
        print("\n" + "=" * 60)
        print("🤖 手部抓握姿态全自动演示 (MuJoCo 3.x 兼容版)")
        print("=" * 60)
        print(f"🎬 演示序列: {len(self.demo_sequence)} 个姿态")
        print(f"⏱️  每个姿态保持: 3.0秒")
        print(f"🎥 动画过渡: {self.animation_duration}秒")
        print("按 Ctrl+C 退出演示")
        print("=" * 60)

        # 设置初始姿态
        initial_pose = self.demo_sequence[0]
        self.data.ctrl[:] = self.poses[initial_pose]['values']

        last_change = time.time()
        hold_duration = 3.0  # 每个姿态保持3秒

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

                while viewer.is_running():
                    current_time = time.time()

                    # 更新动画
                    self.update_animation()

                    # 检查是否需要切换到下一个姿态
                    if not self.animating and (current_time - last_change > hold_duration):
                        self.current_pose_idx = (self.current_pose_idx + 1) % len(self.demo_sequence)
                        next_pose = self.demo_sequence[self.current_pose_idx]

                        if self.start_animation(next_pose):
                            last_change = current_time

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
            print("\n" + "=" * 60)
            print("🎉 演示结束")
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