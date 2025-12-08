#!/usr/bin/env python3
import rospy
import mujoco
from mujoco import viewer
import threading
import sys
import tty
import termios
import rospkg  # 用于获取ROS功能包路径


class HumanoidStandupController:
    def __init__(self):
        # 1. 初始化ROS节点与参数
        rospy.init_node("humanoid_standup_node", anonymous=True)
        rospy.loginfo("人形机器人起身控制器启动...")

        # 从ROS参数服务器获取配置（默认值为launch文件中设置）
        self.kp_gain = rospy.get_param("kp_gain", 5.0)  # 比例增益
        self.model_path = rospy.get_param("model_path", "xml/humanoid.xml")  # 模型相对路径

        # 2. 解析模型绝对路径（通过rospkg获取功能包路径）
        self.rospack = rospkg.RosPack()
        try:
            pkg_path = self.rospack.get_path("humanoid_motion")  # 功能包路径
            self.full_model_path = f"{pkg_path}/{self.model_path}"  # 拼接绝对路径
            rospy.loginfo(f"模型路径: {self.full_model_path}")
        except rospkg.ResourceNotFound:
            rospy.logerr("功能包'humanoid_motion'未找到，请确认包已安装")
            sys.exit(1)

        # 3. 加载MuJoCo模型与初始化仿真数据
        try:
            self.model = mujoco.MjModel.from_xml_path(self.full_model_path)  # 模型对象
            self.data = mujoco.MjData(self.model)  # 仿真数据（关节角度、力矩等）
            self.target_data = mujoco.MjData(self.model)  # 目标状态数据（用于存储目标姿势）
        except Exception as e:
            rospy.logerr(f"模型加载失败: {e}")
            sys.exit(1)

        # 4. 设置初始姿势与目标姿势（通过关键帧，需模型xml中定义keyframe）
        # 关键帧0: 初始姿势（如深蹲）；关键帧1: 目标姿势（如站立）
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # 初始姿势
        mujoco.mj_resetDataKeyframe(self.model, self.target_data, 1)  # 目标姿势
        rospy.loginfo("初始姿势与目标姿势加载完成")

        # 打印关键信息（用于调试：确认关节数和控制信号数）
        rospy.loginfo(f"总关节数(njnt): {self.model.njnt} | 可控制信号数(ctrl_size): {len(self.data.ctrl)}")

        # 5. 键盘控制状态变量
        self.running = False  # 是否执行起身控制
        self.exit_flag = False  # 是否退出程序
        self.last_log_time = rospy.Time.now()  # 用于控制日志输出频率

        # 6. 启动键盘监听线程
        self.keyboard_thread = threading.Thread(target=self._keyboard_listener)
        self.keyboard_thread.daemon = True  # 主线程退出时自动结束
        self.keyboard_thread.start()

        # 7. 打印操作说明
        self._print_help()

    def _print_help(self):
        """打印键盘控制指令说明"""
        print("\n===== 键盘控制指令 =====")
        print("  s: 开始/继续起身控制")
        print("  p: 暂停起身控制（保持当前姿势）")
        print("  +: 增大比例增益KP (+0.5)")
        print("  -: 减小比例增益KP (-0.5)")
        print("  q: 退出程序")
        print("=======================")

    def _keyboard_listener(self):
        """独立线程：监听键盘输入并更新控制状态"""
        # 配置终端为非阻塞模式（无需按回车即可捕获按键）
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            while not self.exit_flag:
                key = sys.stdin.read(1)  # 读取单个字符
                if key == 's':
                    self.running = True
                    rospy.loginfo("▶️ 开始起身控制")
                elif key == 'p':
                    self.running = False
                    rospy.loginfo("⏸️ 已暂停起身控制")
                elif key == '+':
                    self.kp_gain += 0.5
                    rospy.loginfo(f"📈 KP增益调整为: {self.kp_gain:.1f}")
                elif key == '-':
                    self.kp_gain = max(0.5, self.kp_gain - 0.5)  # 限制最小增益为0.5
                    rospy.loginfo(f"📉 KP增益调整为: {self.kp_gain:.1f}")
                elif key == 'q':
                    self.exit_flag = True
                    rospy.loginfo("❌ 准备退出程序...")
        finally:
            # 恢复终端默认设置（避免程序退出后终端异常）
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _log_state(self):
        """定时输出机器人状态（避免日志刷屏）"""
        current_time = rospy.Time.now()
        if (current_time - self.last_log_time).to_sec() > 1.0:  # 每1秒输出一次
            # 躯干高度（假设模型root关节的z坐标为躯干高度）
            torso_height = self.data.qpos[2]  # 需根据模型结构调整索引
            #rospy.loginfo(f"当前躯干高度: {torso_height:.2f}m | KP增益: {self.kp_gain:.1f}")
            self.last_log_time = current_time

    def run(self):
        """主控制循环：执行仿真与控制逻辑"""
        # 启动MuJoCo可视化窗口（被动模式，由主循环驱动）
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        rospy.loginfo("可视化窗口启动，等待指令...")

        # 控制频率：200Hz（物理仿真步长与控制频率匹配）
        rate = rospy.Rate(200)

        try:
            while not rospy.is_shutdown() and not self.exit_flag:
                # 1. 推进物理仿真（单步）
                mujoco.mj_step(self.model, self.data)

                # 2. 若处于运行状态，执行起身控制（比例控制）
                if self.running:
                    # 关键修改：按可控制信号数遍历（而非总关节数）
                    # 假设可控制关节对应原关节索引的[7, 7+ctrl_size)，需与模型结构匹配
                    ctrl_size = len(self.data.ctrl)
                    for ctrl_idx in range(ctrl_size):
                        joint_idx = 7 + ctrl_idx  # 跳过前7个根关节，映射到可控制关节
                        # 确保joint_idx不超出qpos的索引范围（双重保险）
                        if joint_idx >= len(self.data.qpos) or joint_idx >= len(self.target_data.qpos):
                            rospy.logwarn(f"关节索引{joint_idx}超出qpos范围，跳过该关节")
                            continue
                        # 计算关节角度误差（目标-当前）
                        error = self.target_data.qpos[joint_idx] - self.data.qpos[joint_idx]
                        # 比例控制：力矩 = KP * 误差（赋值给对应控制信号索引）
                        self.data.ctrl[ctrl_idx] = self.kp_gain * error

                # 3. 定时输出状态日志
                self._log_state()

                # 4. 刷新可视化窗口
                viewer.sync()

                # 5. 控制循环频率
                rate.sleep()

        except Exception as e:
            rospy.logerr(f"主循环异常: {e}")
            # 输出详细调试信息
            rospy.logerr(f"当前ctrl索引范围: 0~{len(self.data.ctrl)-1} | 映射后的关节索引: 7~{7+len(self.data.ctrl)-1}")
        finally:
            # 关闭可视化窗口
            viewer.close()
            rospy.loginfo("程序已退出")


if __name__ == "__main__":
    try:
        controller = HumanoidStandupController()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS节点被中断")
    except Exception as e:
        rospy.logerr(f"程序异常: {e}")