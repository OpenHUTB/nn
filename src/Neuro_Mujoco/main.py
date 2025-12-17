import os
import sys
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List, Dict
import numpy as np
import mujoco
from mujoco import viewer

# ===================== 机器学习（PyTorch）相关导入 =====================
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("未检测到PyTorch，策略控制功能已禁用（安装：pip install torch）")

# ===================== ROS 1 相关导入 =====================
try:
    import rospy
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float32MultiArray
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logging.warning("未检测到 ROS 环境，ROS 功能已禁用（如需启用，请安装 ROS 1 Noetic）")

# ===================== 日志系统配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("mujoco_utils")

# ===================== 强化学习策略网络 =====================
class PolicyNetwork(nn.Module):
    """轻量级策略网络（适用于MuJoCo机器人控制）
    输入：观测（关节位置+速度），输出：归一化控制指令（[-1,1]）
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出范围[-1,1]，后续映射到实际控制范围
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向推理（禁用梯度计算以提升效率）"""
        with torch.no_grad():
            return self.net(x)

# ===================== 核心功能函数 =====================
def load_model(model_path: str) -> Tuple[Optional[mujoco.MjModel], Optional[mujoco.MjData]]:
    """
    加载MuJoCo模型（支持XML/MJB格式）
    """
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return None, None

    try:
        if model_path.endswith('.mjb'):
            model = mujoco.MjModel.from_binary_path(model_path)
        else:
            model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        logger.info(f"成功加载模型: {model_path}")
        logger.info(f"模型信息：控制维度(nu)={model.nu} | 关节数(njnt)={model.njnt} | 自由度(nq)={model.nq}")
        return model, data
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}", exc_info=True)
        return None, None


def convert_model(input_path: str, output_path: str) -> bool:
    """
    转换模型格式（XML ↔ MJB）
    """
    model, data = load_model(input_path)
    if not model or not data:
        return False

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"创建输出目录: {output_dir}")
        except Exception as e:
            logger.error(f"无法创建输出目录: {str(e)}")
            return False

    try:
        if output_path.endswith('.mjb'):
            mujoco.save_model(model, output_path)
            logger.info(f"二进制模型已保存至: {output_path}")
        else:
            xml_content = mujoco.mj_saveLastXMLToString(data)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            logger.info(f"XML模型已保存至: {output_path}")
        return True
    except Exception as e:
        logger.error(f"模型转换失败: {str(e)}", exc_info=True)
        return False


def test_speed(
    model_path: str,
    nstep: int = 10000,
    nthread: int = 1,
    ctrlnoise: float = 0.01
) -> None:
    """
    测试模型模拟速度（多线程）
    """
    model, _ = load_model(model_path)
    if not model:
        return

    # 参数验证
    if nstep <= 0:
        logger.error("步数必须为正数")
        return
    if nthread <= 0:
        logger.error("线程数必须为正数")
        return

    logger.info(f"开始速度测试: 线程数={nthread}, 每线程步数={nstep}")

    def simulate_thread(thread_id: int) -> float:
        """单线程模拟函数（优化：线程内逐步生成噪声，降低内存占用）"""
        mj_data = mujoco.MjData(model)
        start = time.perf_counter()
        for i in range(nstep):
            # 优化点：线程内逐步生成控制噪声，避免主线程预生成大数组
            if model.nu > 0:
                mj_data.ctrl[:] = ctrlnoise * np.random.randn(model.nu)
            mujoco.mj_step(model, mj_data)
        end = time.perf_counter()
        duration = end - start
        logger.debug(f"线程 {thread_id} 完成，耗时: {duration:.2f}秒")
        return duration

    # 执行多线程测试
    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=nthread) as executor:
        thread_durations: List[float] = list(executor.map(simulate_thread, range(nthread)))
    total_time = time.perf_counter() - start_time

    # 计算性能指标
    total_steps = nstep * nthread
    steps_per_sec = total_steps / total_time
    realtime_factor = (total_steps * model.opt.timestep) / total_time

    logger.info("\n===== 速度测试结果 =====")
    logger.info(f"总步数: {total_steps:,}")
    logger.info(f"总耗时: {total_time:.2f}秒")
    logger.info(f"每秒步数: {steps_per_sec:.0f}")
    logger.info(f"实时因子: {realtime_factor:.2f}x")
    logger.info(f"线程平均耗时: {np.mean(thread_durations):.2f}秒 (±{np.std(thread_durations):.2f})")


def visualize(model_path: str, use_ros: bool = False, policy_path: Optional[str] = None) -> None:
    """
    可视化模型并运行模拟（支持ROS/策略控制）
    :param model_path: 模型文件路径/目录路径
    :param use_ros: 是否启用ROS模式
    :param policy_path: 预训练策略模型路径（.pth）
    """
    # 模型路径智能校验（支持目录自动找XML/MJB文件）
    if os.path.isdir(model_path):
        model_files = []
        for file in os.listdir(model_path):
            if file.endswith(('.xml', '.mjb')):
                model_files.append(os.path.join(model_path, file))
        if not model_files:
            logger.error(f"目录 {model_path} 中未找到.xml/.mjb模型文件")
            return
        model_path = model_files[0]
        logger.info(f"自动选择目录中的模型文件: {model_path}")
    
    model, data = load_model(model_path)
    if not model:
        return

    # ===================== 策略网络初始化 =====================
    policy = None
    obs_dim = model.nq + model.nv  # 观测维度：关节位置 + 关节速度
    action_dim = model.nu
    ctrl_range = None  # 控制指令实际范围

    if policy_path and TORCH_AVAILABLE and action_dim > 0:
        try:
            # 加载预训练策略模型
            policy = PolicyNetwork(obs_dim, action_dim)
            policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))
            policy.eval()  # 推理模式
            logger.info(f"成功加载策略模型: {policy_path}")

            # 获取控制指令范围（映射[-1,1]到实际范围）
            ctrl_range = []
            for i in range(action_dim):
                if model.actuator_ctrllimited[i]:
                    ctrl_range.append(model.actuator_ctrlrange[i])
                else:
                    ctrl_range.append((-1.0, 1.0))
            ctrl_range = np.array(ctrl_range)
        except Exception as e:
            logger.error(f"策略模型加载失败: {str(e)}", exc_info=True)
            policy = None
    elif policy_path:
        logger.warning("策略功能需满足：PyTorch已安装 + 模型有控制维度(nu>0)")

    # ===================== ROS 初始化 =====================
    ros_publishers = None
    ros_subscribers = None
    ros_rate = None
    ctrl_cmd = None
    joint_msg = None
    joint_ids = []
    joint_qpos_idxs = []
    joint_qvel_idxs = []

    if use_ros:
        if not ROS_AVAILABLE:
            logger.error("ROS环境未就绪，无法启用ROS模式")
            return
        
        rospy.init_node("mujoco_ros_node", anonymous=True)
        ros_rate = rospy.Rate(100)  # 100Hz匹配MuJoCo默认步长
        logger.info("="*60)
        logger.info("ROS 1 模式已启用！")
        logger.info(f"发布话题：/mujoco/joint_states、/mujoco/pose")
        logger.info(f"订阅话题：/mujoco/ctrl_cmd（长度={model.nu}）")
        logger.info("="*60)

        # 创建ROS发布者
        joint_state_pub = rospy.Publisher("/mujoco/joint_states", JointState, queue_size=10)
        pose_pub = rospy.Publisher("/mujoco/pose", PoseStamped, queue_size=10)
        ros_publishers = (joint_state_pub, pose_pub)

        # 初始化关节状态消息（精准映射非自由关节）
        joint_msg = JointState()
        joint_msg.name = []
        for i in range(model.njnt):
            joint_type = model.joint(i).type
            if joint_type != mujoco.mjtJoint.mjJNT_FREE:
                joint_msg.name.append(model.joint(i).name)
                joint_ids.append(i)
                joint_qpos_idxs.append(model.jnt_qposadr[i])
                joint_qvel_idxs.append(model.jnt_dofadr[i])
        
        njnt = len(joint_msg.name)
        logger.info(f"ROS将发布 {njnt} 个非自由关节状态：{joint_msg.name}")

        # 创建ROS订阅者（接收控制指令）
        ctrl_cmd = np.zeros(model.nu) if model.nu > 0 else None
        def ctrl_callback(msg: Float32MultiArray):
            nonlocal ctrl_cmd
            if model.nu == len(msg.data):
                ctrl_cmd = np.array(msg.data)
                logger.debug(f"收到ROS控制指令：{ctrl_cmd[:5]}...")
            else:
                logger.warning(f"控制指令长度不匹配！期望 {model.nu}，实际 {len(msg.data)}")
        
        if model.nu > 0:
            ros_subscribers = rospy.Subscriber(
                "/mujoco/ctrl_cmd", Float32MultiArray, ctrl_callback, queue_size=5
            )
        else:
            logger.warning("模型无控制输入（nu=0），不订阅控制指令话题")

    # ===================== 可视化主循环 =====================
    logger.info("启动可视化窗口（按ESC键退出 | 鼠标交互：拖拽旋转、滚轮缩放）")  # 修正缩进
    try:
        with viewer.launch_passive(model, data) as v:
            # 预分配推理张量（复用，避免每次创建）
            obs_tensor = None if policy is None else torch.zeros(1, obs_dim, dtype=torch.float32)
            
            while v.is_running() and (not use_ros or not rospy.is_shutdown()):
                # 控制指令优先级：ROS指令 > 策略推理 > 无控制
                if use_ros and ctrl_cmd is not None:
                    data.ctrl[:] = ctrl_cmd
                elif policy is not None:
                    # 提取观测：关节位置 + 关节速度
                    obs = np.concatenate([data.qpos, data.qvel])
                    
                    # 优化1：复用张量，避免重复内存分配
                    if obs_tensor is None:
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    else:
                        obs_tensor[0] = torch.from_numpy(obs)
                    
                    # 策略推理
                    action = policy(obs_tensor).squeeze().numpy()
                    
                    # 映射到实际控制范围 + 优化2：强制裁剪到物理极限（避免超限）
                    if ctrl_range is not None:
                        action = ctrl_range[:, 0] + (ctrl_range[:, 1] - ctrl_range[:, 0]) * (action + 1) / 2  # 核心映射：[-1,1]→[ctrl_min,ctrl_max] 线性缩放
                        action = np.clip(action, ctrl_range[:, 0], ctrl_range[:, 1])  # 强制裁剪，保证指令符合执行器物理极限
                        action = ctrl_range[:, 0] + (ctrl_range[:, 1] - ctrl_range[:, 0]) * (action + 1) / 2  # 核心映射：[-1,1]→[ctrl_min,ctrl_max] 线性缩放，保证指令符合执行器物理极限
                    
                    data.ctrl[:] = action

                # 执行模拟步
                mujoco.mj_step(model, data)
                v.sync()

                # ROS消息发布（修正缩进：从policy分支移出，作为while循环顶层逻辑）
                if use_ros and ros_publishers is not None:
                    joint_state_pub, pose_pub = ros_publishers

                    # 发布关节状态
                    joint_msg.header.stamp = rospy.Time.now()
                    joint_msg.position = []
                    joint_msg.velocity = []
                    for joint_id, qpos_idx, qvel_idx in zip(joint_ids, joint_qpos_idxs, joint_qvel_idxs):
                        joint_type = model.joint(joint_id).type
                        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
                            joint_msg.position.extend(data.qpos[qpos_idx:qpos_idx+3])
                            joint_msg.velocity.extend(data.qvel[qvel_idx:qvel_idx+3])
                        elif joint_type in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                            joint_msg.position.append(data.qpos[qpos_idx])
                            joint_msg.velocity.append(data.qvel[qvel_idx])
                    
                    joint_state_pub.publish(joint_msg)

                    # 发布基座姿态
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = rospy.Time.now()
                    pose_msg.header.frame_id = "world"
                    
                    if model.nq >= 1:
                        pose_msg.pose.position.x = data.qpos[0]
                    if model.nq >= 2:
                        pose_msg.pose.position.y = data.qpos[1]
                    if model.nq >= 3:
                        pose_msg.pose.position.z = data.qpos[2]
                    
                    if model.nq >= 4:
                        pose_msg.pose.orientation.x = data.qpos[3]
                    if model.nq >= 5:
                        pose_msg.pose.orientation.y = data.qpos[4]
                    if model.nq >= 6:
                        pose_msg.pose.orientation.z = data.qpos[5]
                    if model.nq >= 7:
                        pose_msg.pose.orientation.w = data.qpos[6]
                    
                    pose_pub.publish(pose_msg)
                    ros_rate.sleep()

        logger.info("可视化窗口已关闭")
    except Exception as e:
        logger.error(f"可视化过程出错: {str(e)}", exc_info=True)

# ===================== 主函数（命令行入口） =====================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="MuJoCo功能整合工具（支持强化学习策略控制 + ROS 1通信）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. 可视化命令（修复重复添加model参数问题）
    viz_parser = subparsers.add_parser("visualize", help="可视化模型并运行模拟")
    viz_parser.add_argument(
        "model", 
        help="模型文件路径/目录（示例：/home/lan/桌面/nn/mujoco_menagerie/anybotics_anymal_b）"
    )
    viz_parser.add_argument("--ros", action="store_true", help="启用ROS 1模式（发布/订阅关节控制话题）")
    viz_parser.add_argument("--policy", help="预训练策略模型路径（.pth文件）")

    # 2. 速度测试命令
    speed_parser = subparsers.add_parser("testspeed", help="测试模型模拟速度（多线程）")
    speed_parser.add_argument("model", help="模型文件路径")
    speed_parser.add_argument("--nstep", type=int, default=10000, help="每线程模拟步数")
    speed_parser.add_argument("--nthread", type=int, default=1, help="测试线程数")
    speed_parser.add_argument("--ctrlnoise", type=float, default=0.01, help="控制噪声强度")

    # 3. 模型转换命令
    convert_parser = subparsers.add_parser("convert", help="转换模型格式（XML ↔ MJB）")
    convert_parser.add_argument("input", help="输入模型路径")
    convert_parser.add_argument("output", help="输出模型路径（指定.xml/.mjb）")

    # 解析命令行参数
    args, unknown = parser.parse_known_args()

    # 命令映射
    command_handlers: Dict[str, callable] = {
        "visualize": lambda: visualize(args.model, use_ros=args.ros, policy_path=args.policy),
        "testspeed": lambda: test_speed(args.model, args.nstep, args.nthread, args.ctrlnoise),
        "convert": lambda: convert_model(args.input, args.output)
    }

    # 执行命令
    try:
        command_handlers[args.command]()
    except KeyError:
        logger.error(f"未知命令: {args.command}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"程序执行失败: {str(e)}", exc_info=True)
        sys.exit(1)

# ===================== 程序入口 =====================
if __name__ == "__main__":
    main()