#! /usr/bin/env python
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

# 新增：机器学习相关库
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("未检测到PyTorch，策略控制功能已禁用")

# ===================== ROS 1 相关导入 =====================
try:
    import rospy
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float32MultiArray
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logging.warning("未检测到 ROS 环境，ROS 功能已禁用")


# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("mujoco_utils")


# 新增：简单的策略网络结构（示例）
class PolicyNetwork(nn.Module):
    """强化学习策略网络，用于生成控制指令"""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出范围[-1,1]，后续需映射到实际控制范围
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_model(model_path: str) -> Tuple[Optional[mujoco.MjModel], Optional[mujoco.MjData]]:
    """加载MuJoCo模型（支持XML和MJB格式）"""
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
    """转换模型格式（XML↔MJB）"""
    model, data = load_model(input_path)
    if not model or not data:
        return False

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
    """测试模型模拟速度"""
    model, _ = load_model(model_path)
    if not model:
        return

    if nstep <= 0:
        logger.error("步数必须为正数")
        return
    if nthread <= 0:
        logger.error("线程数必须为正数")
        return

    if model.nu == 0:
        ctrl = None
        logger.warning("模型无控制输入（nu=0），将跳过控制噪声")
    else:
        ctrl = ctrlnoise * np.random.randn(nstep, model.nu)
    
    logger.info(f"开始速度测试: 线程数={nthread}, 每线程步数={nstep}")

    def simulate_thread(thread_id: int) -> float:
        mj_data = mujoco.MjData(model)
        start = time.perf_counter()
        for i in range(nstep):
            if ctrl is not None:
                mj_data.ctrl[:] = ctrl[i]
            mujoco.mj_step(model, mj_data)
        end = time.perf_counter()
        duration = end - start
        logger.debug(f"线程 {thread_id} 完成，耗时: {duration:.2f}秒")
        return duration

    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=nthread) as executor:
        thread_durations: List[float] = list(executor.map(simulate_thread, range(nthread)))
    total_time = time.perf_counter() - start_time

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
    可视化模型并运行模拟（新增策略控制功能）
    
    参数:
        policy_path: 预训练策略模型路径（.pth文件）
    """
    if os.path.isdir(model_path):
        model_files = []
        for file in os.listdir(model_path):
            if file.endswith(('.xml', '.mjb')):
                model_files.append(os.path.join(model_path, file))
        if not model_files:
            logger.error(f"目录 {model_path} 中未找到模型文件")
            return
        model_path = model_files[0]
        logger.info(f"自动选择模型文件: {model_path}")
    
    model, data = load_model(model_path)
    if not model:
        return

    # 策略加载与初始化
    policy = None
    obs_dim = model.nq + model.nv  # 观测维度：关节位置+速度
    action_dim = model.nu
    ctrl_range = None  # 存储控制指令范围用于映射

    if policy_path and TORCH_AVAILABLE and action_dim > 0:
        try:
            # 加载策略网络
            policy = PolicyNetwork(obs_dim, action_dim)
            policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))
            policy.eval()
            logger.info(f"成功加载策略模型: {policy_path}")

            # 获取控制指令范围（用于将[-1,1]输出映射到实际范围）
            ctrl_range = []
            for i in range(action_dim):
                if model.actuator_ctrllimited[i]:
                    ctrl_range.append(model.actuator_ctrlrange[i])
                else:
                    ctrl_range.append((-1.0, 1.0))  # 无限制时默认范围
            ctrl_range = np.array(ctrl_range)
        except Exception as e:
            logger.error(f"策略模型加载失败: {str(e)}", exc_info=True)
            policy = None
    elif policy_path:
        logger.warning("策略功能需同时满足：PyTorch已安装且模型有控制维度(nu>0)")

    # ROS相关初始化
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
        ros_rate = rospy.Rate(100)
        logger.info("="*60)
        logger.info("ROS 1 模式已启用！")
        logger.info(f"发布话题：/mujoco/joint_states、/mujoco/pose")
        logger.info(f"订阅话题：/mujoco/ctrl_cmd（长度={model.nu}）")
        logger.info("="*60)

        joint_state_pub = rospy.Publisher("/mujoco/joint_states", JointState, queue_size=10)
        pose_pub = rospy.Publisher("/mujoco/pose", PoseStamped, queue_size=10)
        ros_publishers = (joint_state_pub, pose_pub)

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

    # 可视化主循环
    logger.info("启动可视化窗口（按ESC键退出）")
    try:
        with viewer.launch_passive(model, data) as v:
            while v.is_running() and (not use_ros or not rospy.is_shutdown()):
                # 1. 生成控制指令（优先级：ROS指令 > 策略 > 无控制）
                if use_ros and ctrl_cmd is not None:
                    data.ctrl[:] = ctrl_cmd
                elif policy is not None:
                    # 提取观测：关节位置+速度
                    obs = np.concatenate([data.qpos, data.qvel])
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    
                    # 策略推理（禁用梯度计算提高效率）
                    with torch.no_grad():
                        action = policy(obs_tensor).squeeze().numpy()
                    
                    # 将策略输出映射到实际控制范围
                    if ctrl_range is not None:
                        # 从[-1,1]映射到[min, max]：min + (max-min)*(action+1)/2
                        action = ctrl_range[:, 0] + (ctrl_range[:, 1] - ctrl_range[:, 0]) * (action + 1) / 2
                    
                    data.ctrl[:] = action

                # 2. 执行模拟步
                mujoco.mj_step(model, data)
                v.sync()

                # 3. ROS消息发布
                if use_ros and ros_publishers is not None:
                    joint_state_pub, pose_pub = ros_publishers

                    joint_msg.header.stamp = rospy.Time.now()
                    joint_msg.position = []
                    joint_msg.velocity = []
                    for idx, (joint_id, qpos_idx, qvel_idx) in enumerate(zip(joint_ids, joint_qpos_idxs, joint_qvel_idxs)):
                        joint_type = model.joint(joint_id).type
                        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
                            joint_msg.position.extend(data.qpos[qpos_idx:qpos_idx+3])
                            joint_msg.velocity.extend(data.qvel[qvel_idx:qvel_idx+3])
                        elif joint_type in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                            joint_msg.position.append(data.qpos[qpos_idx])
                            joint_msg.velocity.append(data.qvel[qvel_idx])
                    
                    joint_state_pub.publish(joint_msg)

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MuJoCo功能整合工具（支持策略控制与ROS）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 可视化命令（新增策略参数）
    viz_parser = subparsers.add_parser("visualize", help="可视化模型并运行模拟")
    viz_parser.add_argument("model", help="模型文件路径或目录")
    viz_parser.add_argument("--ros", action="store_true", help="启用ROS模式")
    viz_parser.add_argument("--policy", help="预训练策略模型路径（.pth文件）")  # 新增参数

    # 速度测试命令
    speed_parser = subparsers.add_parser("testspeed", help="测试模型模拟速度")
    speed_parser.add_argument("model", help="模型文件路径")
    speed_parser.add_argument("--nstep", type=int, default=10000, help="每线程步数")
    speed_parser.add_argument("--nthread", type=int, default=1, help="线程数量")
    speed_parser.add_argument("--ctrlnoise", type=float, default=0.01, help="控制噪声强度")

    # 模型转换命令
    convert_parser = subparsers.add_parser("convert", help="转换模型格式")
    convert_parser.add_argument("input", help="输入模型路径")
    convert_parser.add_argument("output", help="输出模型路径")

    args, unknown = parser.parse_known_args()

    # 命令映射（更新可视化函数参数）
    command_handlers: Dict[str, callable] = {
        "visualize": lambda: visualize(args.model, use_ros=args.ros, policy_path=args.policy),
        "testspeed": lambda: test_speed(args.model, args.nstep, args.nthread, args.ctrlnoise),
        "convert": lambda: convert_model(args.input, args.output)
    }

    try:
        command_handlers[args.command]()
    except KeyError:
        logger.error(f"未知命令: {args.command}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"程序执行失败: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()