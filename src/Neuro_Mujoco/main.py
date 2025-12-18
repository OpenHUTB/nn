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

# ===================== 依赖导入 - 机器学习（PyTorch） =====================
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch未检测到，策略控制功能禁用（安装命令：pip install torch）")

# ===================== 依赖导入 - ROS 1 =====================
try:
    import rospy
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float32MultiArray
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logging.warning("ROS环境未检测到，ROS功能禁用（需安装ROS 1 Noetic）")

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("mujoco_control_tool")

# ===================== 强化学习策略网络定义 =====================
class PolicyNetwork(nn.Module):
    """
    轻量级策略网络（适配MuJoCo机器人控制场景）
    输入：观测向量（关节位置 + 关节速度）
    输出：归一化控制指令（范围[-1, 1]）
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出约束在[-1,1]，后续映射到实际控制范围
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """前向推理（禁用梯度计算提升效率）"""
        with torch.no_grad():
            return self.backbone(obs)

# ===================== 核心功能函数 =====================
def load_mujoco_model(model_path: str) -> Tuple[Optional[mujoco.MjModel], Optional[mujoco.MjData]]:
    """
    加载MuJoCo模型（支持XML/MJB格式）
    :param model_path: 模型文件路径
    :return: (MjModel实例, MjData实例)，加载失败返回(None, None)
    """
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在：{model_path}")
        return None, None

    try:
        # 区分二进制/XML格式加载
        if model_path.endswith('.mjb'):
            model = mujoco.MjModel.from_binary_path(model_path)
        else:
            model = mujoco.MjModel.from_xml_path(model_path)
        
        data = mujoco.MjData(model)
        logger.info(f"模型加载成功：{model_path}")
        logger.info(f"模型参数：控制维度(nu)={model.nu} | 关节数(njnt)={model.njnt} | 自由度(nq)={model.nq}")
        return model, data
    except Exception as e:
        logger.error(f"模型加载失败：{str(e)}", exc_info=True)
        return None, None


def convert_mujoco_model(input_path: str, output_path: str) -> bool:
    """
    转换MuJoCo模型格式（XML ↔ MJB）
    :param input_path: 输入模型路径
    :param output_path: 输出模型路径（指定.xml/.mjb后缀）
    :return: 转换成功返回True，失败返回False
    """
    model, data = load_mujoco_model(input_path)
    if not model or not data:
        return False

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"创建输出目录：{output_dir}")
        except Exception as e:
            logger.error(f"创建输出目录失败：{str(e)}")
            return False

    try:
        if output_path.endswith('.mjb'):
            # 保存为二进制格式
            mujoco.save_model(model, output_path)
            logger.info(f"二进制模型已保存：{output_path}")
        else:
            # 保存为XML格式
            xml_content = mujoco.mj_saveLastXMLToString(data)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            logger.info(f"XML模型已保存：{output_path}")
        return True
    except Exception as e:
        logger.error(f"模型转换失败：{str(e)}", exc_info=True)
        return False


def test_simulation_speed(
    model_path: str,
    step_num: int = 10000,
    thread_num: int = 1,
    ctrl_noise: float = 0.01
) -> None:
    """
    测试MuJoCo模型模拟速度（支持多线程）
    :param model_path: 模型文件路径
    :param step_num: 每线程模拟步数
    :param thread_num: 测试线程数
    :param ctrl_noise: 控制指令噪声强度
    """
    model, _ = load_mujoco_model(model_path)
    if not model:
        return

    # 参数合法性校验
    if step_num <= 0:
        logger.error("模拟步数必须为正整数")
        return
    if thread_num <= 0:
        logger.error("线程数必须为正整数")
        return

    logger.info(f"开始模拟速度测试 | 线程数：{thread_num} | 每线程步数：{step_num}")

    def single_thread_simulation(thread_id: int) -> float:
        """单线程模拟函数（优化内存占用：逐步生成控制噪声）"""
        mj_data = mujoco.MjData(model)
        start_time = time.perf_counter()
        
        for _ in range(step_num):
            # 逐步生成控制噪声，避免预分配大数组
            if model.nu > 0:
                mj_data.ctrl[:] = ctrl_noise * np.random.randn(model.nu)
            mujoco.mj_step(model, mj_data)
        
        duration = time.perf_counter() - start_time
        logger.debug(f"线程 {thread_id} 完成 | 耗时：{duration:.2f}秒")
        return duration

    # 多线程执行模拟
    total_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        thread_durations = list(executor.map(single_thread_simulation, range(thread_num)))
    total_duration = time.perf_counter() - total_start

    # 计算性能指标
    total_steps = step_num * thread_num
    steps_per_second = total_steps / total_duration
    realtime_factor = (total_steps * model.opt.timestep) / total_duration

    logger.info("\n===== 模拟速度测试结果 =====")
    logger.info(f"总模拟步数：{total_steps:,}")
    logger.info(f"总耗时：{total_duration:.2f}秒")
    logger.info(f"每秒步数：{steps_per_second:.0f} step/s")
    logger.info(f"实时因子：{realtime_factor:.2f}x")
    logger.info(f"线程平均耗时：{np.mean(thread_durations):.2f}秒（±{np.std(thread_durations):.2f}）")


def visualize_model(model_path: str, use_ros_mode: bool = False, policy_model_path: Optional[str] = None) -> None:
    """
    可视化MuJoCo模型并运行模拟（支持ROS通信/策略控制）
    :param model_path: 模型文件路径/目录路径
    :param use_ros_mode: 是否启用ROS 1通信模式
    :param policy_model_path: 预训练策略模型路径（.pth文件）
    """
    # 路径智能匹配：如果是目录，自动查找XML/MJB文件
    if os.path.isdir(model_path):
        model_files = [
            os.path.join(model_path, f) 
            for f in os.listdir(model_path) 
            if f.endswith(('.xml', '.mjb'))
        ]
        if not model_files:
            logger.error(f"目录 {model_path} 中未找到.xml/.mjb模型文件")
            return
        model_path = model_files[0]
        logger.info(f"自动选择模型文件：{model_path}")
    
    # 加载模型
    model, data = load_mujoco_model(model_path)
    if not model:
        return

    # ===================== 策略网络初始化 =====================
    policy_net = None
    obs_dimension = model.nq + model.nv  # 观测维度 = 关节位置数 + 关节速度数
    action_dimension = model.nu
    control_range = None  # 执行器控制范围（min, max）

    if policy_model_path and TORCH_AVAILABLE and action_dimension > 0:
        try:
            # 初始化并加载预训练策略
            policy_net = PolicyNetwork(obs_dimension, action_dimension)
            policy_net.load_state_dict(torch.load(policy_model_path, map_location=torch.device('cpu')))
            policy_net.eval()  # 推理模式（禁用Dropout/BatchNorm）
            logger.info(f"预训练策略模型加载成功：{policy_model_path}")

            # 获取执行器控制范围（用于映射归一化输出）
            control_range = []
            for idx in range(action_dimension):
                if model.actuator_ctrllimited[idx]:
                    control_range.append(model.actuator_ctrlrange[idx])
                else:
                    control_range.append((-1.0, 1.0))
            control_range = np.array(control_range)
        except Exception as e:
            logger.error(f"策略模型加载失败：{str(e)}", exc_info=True)
            policy_net = None
    elif policy_model_path:
        logger.warning("策略控制需满足：PyTorch已安装 + 模型有控制维度（nu>0）")

    # ===================== ROS 初始化 =====================
    ros_pubs = None
    ros_subs = None
    ros_loop_rate = None
    ros_ctrl_cmd = None
    joint_state_msg = None
    joint_ids_list = []
    joint_qpos_indices = []
    joint_qvel_indices = []

    if use_ros_mode:
        if not ROS_AVAILABLE:
            logger.error("ROS环境未就绪，无法启用ROS模式")
            return
        
        rospy.init_node("mujoco_visualizer_node", anonymous=True)
        ros_loop_rate = rospy.Rate(100)  # 100Hz（匹配MuJoCo默认步长）
        logger.info("="*60)
        logger.info("ROS 1 通信模式已启用")
        logger.info(f"发布话题：/mujoco/joint_states、/mujoco/base_pose")
        logger.info(f"订阅话题：/mujoco/control_command（长度={model.nu}）")
        logger.info("="*60)

        # 创建ROS发布者
        joint_state_pub = rospy.Publisher("/mujoco/joint_states", JointState, queue_size=10)
        base_pose_pub = rospy.Publisher("/mujoco/base_pose", PoseStamped, queue_size=10)
        ros_pubs = (joint_state_pub, base_pose_pub)

        # 初始化关节状态消息（仅包含非自由关节）
        joint_state_msg = JointState()
        joint_state_msg.name = []
        for joint_idx in range(model.njnt):
            joint_type = model.joint(joint_idx).type
            if joint_type != mujoco.mjtJoint.mjJNT_FREE:
                joint_state_msg.name.append(model.joint(joint_idx).name)
                joint_ids_list.append(joint_idx)
                joint_qpos_indices.append(model.jnt_qposadr[joint_idx])
                joint_qvel_indices.append(model.jnt_dofadr[joint_idx])
        
        valid_joint_num = len(joint_state_msg.name)
        logger.info(f"ROS将发布 {valid_joint_num} 个非自由关节状态：{joint_state_msg.name}")

        # 创建ROS订阅者（接收外部控制指令）
        ros_ctrl_cmd = np.zeros(model.nu) if model.nu > 0 else None
        def control_cmd_callback(msg: Float32MultiArray):
            nonlocal ros_ctrl_cmd
            if len(msg.data) == model.nu:
                ros_ctrl_cmd = np.array(msg.data)
                logger.debug(f"收到ROS控制指令：{ros_ctrl_cmd[:5]}...")
            else:
                logger.warning(f"控制指令长度不匹配 | 期望：{model.nu} | 实际：{len(msg.data)}")
        
        if model.nu > 0:
            ros_subs = rospy.Subscriber(
                "/mujoco/control_command", Float32MultiArray, control_cmd_callback, queue_size=5
            )
        else:
            logger.warning("模型无控制输入（nu=0），跳过控制指令订阅")

    # ===================== 可视化主循环 =====================
    logger.info("启动MuJoCo可视化窗口 | 退出：ESC键 | 交互：鼠标拖拽（旋转）/滚轮（缩放）")
    try:
        with viewer.launch_passive(model, data) as viewer_instance:
            # 预分配观测张量（复用避免重复内存分配）
            obs_tensor = torch.zeros(1, obs_dimension, dtype=torch.float32) if policy_net else None
            
            while viewer_instance.is_running() and (not use_ros_mode or not rospy.is_shutdown()):
                # 控制指令优先级：ROS指令 > 策略推理 > 无控制
                if use_ros_mode and ros_ctrl_cmd is not None:
                    data.ctrl[:] = ros_ctrl_cmd
                elif policy_net is not None:
                    # 提取观测向量：关节位置 + 关节速度
                    observation = np.concatenate([data.qpos, data.qvel])
                    
                    # 复用张量，避免重复创建
                    if obs_tensor is None:
                        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                    else:
                        obs_tensor[0] = torch.from_numpy(observation)
                    
                    # 策略推理得到归一化动作（[-1,1]）
                    normalized_action = policy_net(obs_tensor).squeeze().numpy()
                    
                    # 映射到执行器实际控制范围 + 裁剪（核心优化：删除重复映射）
                    if control_range is not None:
                        # 线性映射：[-1,1] → [ctrl_min, ctrl_max]
                        normalized_action = control_range[:, 0] + (control_range[:, 1] - control_range[:, 0]) * (normalized_action + 1) / 2
                        # 强制裁剪到物理极限（避免超限损坏模拟）
                        normalized_action = np.clip(normalized_action, control_range[:, 0], control_range[:, 1])
                    
                    data.ctrl[:] = normalized_action

                # 执行单步模拟
                mujoco.mj_step(model, data)
                viewer_instance.sync()

                # ROS消息发布逻辑
                if use_ros_mode and ros_pubs is not None:
                    joint_state_pub, base_pose_pub = ros_pubs

                    # 填充并发布关节状态
                    joint_state_msg.header.stamp = rospy.Time.now()
                    joint_state_msg.position = []
                    joint_state_msg.velocity = []
                    
                    for j_id, qp_idx, qv_idx in zip(joint_ids_list, joint_qpos_indices, joint_qvel_indices):
                        j_type = model.joint(j_id).type
                        if j_type == mujoco.mjtJoint.mjJNT_BALL:
                            # 球关节：3个位置/速度维度
                            joint_state_msg.position.extend(data.qpos[qp_idx:qp_idx+3])
                            joint_state_msg.velocity.extend(data.qvel[qv_idx:qv_idx+3])
                        elif j_type in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                            # 铰链/滑动关节：1个位置/速度维度
                            joint_state_msg.position.append(data.qpos[qp_idx])
                            joint_state_msg.velocity.append(data.qvel[qv_idx])
                    
                    joint_state_pub.publish(joint_state_msg)

                    # 填充并发布基座姿态
                    base_pose_msg = PoseStamped()
                    base_pose_msg.header.stamp = rospy.Time.now()
                    base_pose_msg.header.frame_id = "world"
                    
                    # 位置信息（前3维）
                    if model.nq >= 1:
                        base_pose_msg.pose.position.x = data.qpos[0]
                    if model.nq >= 2:
                        base_pose_msg.pose.position.y = data.qpos[1]
                    if model.nq >= 3:
                        base_pose_msg.pose.position.z = data.qpos[2]
                    
                    # 姿态四元数（后4维）
                    if model.nq >= 4:
                        base_pose_msg.pose.orientation.x = data.qpos[3]
                    if model.nq >= 5:
                        base_pose_msg.pose.orientation.y = data.qpos[4]
                    if model.nq >= 6:
                        base_pose_msg.pose.orientation.z = data.qpos[5]
                    if model.nq >= 7:
                        base_pose_msg.pose.orientation.w = data.qpos[6]
                    
                    base_pose_pub.publish(base_pose_msg)
                    ros_loop_rate.sleep()

        logger.info("MuJoCo可视化窗口已关闭")
    except Exception as e:
        logger.error(f"可视化过程异常：{str(e)}", exc_info=True)

# ===================== 命令行入口 =====================
def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo模型工具集（支持可视化/速度测试/格式转换 + ROS/策略控制）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True, help="子命令列表")

    # 子命令1：模型可视化
    viz_parser = subparsers.add_parser("visualize", help="可视化模型并运行模拟")
    viz_parser.add_argument("model_path", help="模型文件/目录路径（示例：./models/anymal.xml）")
    viz_parser.add_argument("--ros", action="store_true", help="启用ROS 1通信模式")
    viz_parser.add_argument("--policy", help="预训练策略模型路径（.pth文件）")

    # 子命令2：模拟速度测试
    speed_parser = subparsers.add_parser("test_speed", help="测试模型模拟速度（多线程）")
    speed_parser.add_argument("model_path", help="模型文件路径")
    speed_parser.add_argument("--step_num", type=int, default=10000, help="每线程模拟步数")
    speed_parser.add_argument("--thread_num", type=int, default=1, help="测试线程数")
    speed_parser.add_argument("--ctrl_noise", type=float, default=0.01, help="控制指令噪声强度")

    # 子命令3：模型格式转换
    convert_parser = subparsers.add_parser("convert", help="转换模型格式（XML ↔ MJB）")
    convert_parser.add_argument("input_path", help="输入模型路径")
    convert_parser.add_argument("output_path", help="输出模型路径（指定.xml/.mjb后缀）")

    # 解析参数
    args = parser.parse_args()

    # 子命令映射
    subcommand_handlers = {
        "visualize": lambda: visualize_model(args.model_path, args.ros, args.policy),
        "test_speed": lambda: test_simulation_speed(args.model_path, args.step_num, args.thread_num, args.ctrl_noise),
        "convert": lambda: convert_mujoco_model(args.input_path, args.output_path)
    }

    # 执行子命令
    try:
        subcommand_handlers[args.subcommand]()
    except KeyError:
        logger.error(f"未知子命令：{args.subcommand}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"程序执行失败：{str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()