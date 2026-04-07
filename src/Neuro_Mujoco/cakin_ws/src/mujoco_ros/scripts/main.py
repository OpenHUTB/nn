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

# ===================== ROS 1 相关导入（新增）=====================
try:
    import rospy
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float32MultiArray
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logging.warning("未检测到 ROS 环境，ROS 功能已禁用（如需启用，请安装 ROS 1 Noetic 并配置环境）")


# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("mujoco_utils")


def load_model(model_path: str) -> Tuple[Optional[mujoco.MjModel], Optional[mujoco.MjData]]:
    """
    加载MuJoCo模型（支持XML和MJB格式）
    
    参数:
        model_path: 模型文件路径
        
    返回:
        加载成功返回(model, data)元组，失败返回(None, None)
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
    转换模型格式（XML↔MJB）
    参数:
        input_path: 输入模型路径
        output_path: 输出模型路径（需指定扩展名.xml或.mjb）
    返回:
        转换成功返回True，失败返回False
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
    测试模型模拟速度
    
    参数:
        model_path: 模型文件路径
        nstep: 每线程模拟步数
        nthread: 测试线程数
        ctrlnoise: 控制噪声强度
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

    # 生成控制噪声（处理nu=0的情况）
    if model.nu == 0:
        ctrl = None
        logger.warning("模型无控制输入（nu=0），将跳过控制噪声")
    else:
        ctrl = ctrlnoise * np.random.randn(nstep, model.nu)
    
    logger.info(f"开始速度测试: 线程数={nthread}, 每线程步数={nstep}")

    def simulate_thread(thread_id: int) -> float:
        """单线程模拟函数"""
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

# ===================== 可视化函数（仅优化ROS关节状态发布逻辑）=====================
def visualize(model_path: str, use_ros: bool = False) -> None:
    """
    可视化模型并运行模拟（支持ROS 1模式）
    
    参数:
        model_path: 模型文件路径
        use_ros: 是否启用ROS模式（默认False）
    """
    model, data = load_model(model_path)
    if not model:
        return

    # ===================== ROS 1 初始化（新增）=====================
    ros_publishers = None
    ros_subscribers = None
    ros_rate = None
    ctrl_cmd = None
    joint_msg = None
    # 新增：存储非自由关节的ID和对应的qpos/qvel索引（解决索引错位+支持多自由度）
    joint_ids = []
    joint_qpos_idxs = []
    joint_qvel_idxs = []

    if use_ros:
        if not ROS_AVAILABLE:
            logger.error("ROS 环境未就绪，无法启用 ROS 模式（请检查ROS安装和环境配置）")
            return
        
        # 初始化ROS节点
        rospy.init_node("mujoco_ros_node", anonymous=True)
        ros_rate = rospy.Rate(100)  # 100Hz发布频率（与MuJoCo默认步长0.01s匹配）
        logger.info("="*60)
        logger.info("ROS 1 模式已启用！")
        logger.info(f"发布话题：/mujoco/joint_states（关节状态）、/mujoco/pose（基座姿态）")
        logger.info(f"订阅话题：/mujoco/ctrl_cmd（控制指令，长度={model.nu}）")
        logger.info("="*60)

        # 1. 创建ROS发布者
        joint_state_pub = rospy.Publisher(
            "/mujoco/joint_states",
            JointState,
            queue_size=10  # 消息队列大小
        )
        pose_pub = rospy.Publisher(
            "/mujoco/pose",
            PoseStamped,
            queue_size=10
        )
        ros_publishers = (joint_state_pub, pose_pub)

        # 2. 初始化关节状态消息（精准映射非自由关节的索引，支持多自由度）
        joint_msg = JointState()
        joint_msg.name = []
        for i in range(model.njnt):
            joint_type = model.joint(i).type
            if joint_type != mujoco.mjtJoint.mjJNT_FREE:
                joint_msg.name.append(model.joint(i).name)
                joint_ids.append(i)
                # 获取该关节在qpos中的起始索引（mjJNT_FREE=7维, mjJNT_BALL=3维, mjJNT_HINGE/SLIDE=1维）
                joint_qpos_idxs.append(model.jnt_qposadr[i])
                # 获取该关节在qvel中的起始索引
                joint_qvel_idxs.append(model.jnt_dofadr[i])
        
        njnt = len(joint_msg.name)
        logger.info(f"ROS将发布 {njnt} 个非自由关节状态：{joint_msg.name}")
        if njnt > 0:
            logger.debug(f"关节qpos索引映射：{dict(zip(joint_msg.name, joint_qpos_idxs))}")

        # 3. 创建ROS订阅者（接收控制指令）
        ctrl_cmd = np.zeros(model.nu) if model.nu > 0 else None
        def ctrl_callback(msg: Float32MultiArray):
            nonlocal ctrl_cmd
            if model.nu == len(msg.data):
                ctrl_cmd = np.array(msg.data)
                logger.debug(f"收到ROS控制指令：{ctrl_cmd[:5]}...")  # 只打印前5个值，避免日志冗余
            else:
                logger.warning(f"控制指令长度不匹配！期望 {model.nu} 个值，实际收到 {len(msg.data)} 个")
        
        if model.nu > 0:
            ros_subscribers = rospy.Subscriber(
                "/mujoco/ctrl_cmd",
                Float32MultiArray,
                ctrl_callback,
                queue_size=5
            )
        else:
            logger.warning("模型无控制输入（nu=0），不订阅控制指令话题")

    # ===================== 可视化主循环（原有逻辑+优化ROS发布）=====================
    logger.info("启动可视化窗口（按ESC键退出，鼠标可交互：拖拽旋转、滚轮缩放）")
    try:
        with viewer.launch_passive(model, data) as v:
            while v.is_running() and (not use_ros or not rospy.is_shutdown()):
                # ROS模式：应用控制指令（新增）
                if use_ros and ctrl_cmd is not None:
                    data.ctrl[:] = ctrl_cmd

                # 执行MuJoCo模拟步（原有逻辑）
                mujoco.mj_step(model, data)
                v.sync()

                # ===================== ROS 消息发布（仅优化关节状态部分）=====================
                if use_ros and ros_publishers is not None:
                    joint_state_pub, pose_pub = ros_publishers

                    # 1. 发布关节状态（位置、速度）- 优化后：精准映射+支持多自由度
                    joint_msg.header.stamp = rospy.Time.now()
                    joint_msg.position = []
                    joint_msg.velocity = []
                    for idx, (joint_id, qpos_idx, qvel_idx) in enumerate(zip(joint_ids, joint_qpos_idxs, joint_qvel_idxs)):
                        joint_type = model.joint(joint_id).type
                        # 球关节（3自由度）：补充3维位置/速度
                        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
                            joint_msg.position.extend(data.qpos[qpos_idx:qpos_idx+3])
                            joint_msg.velocity.extend(data.qvel[qvel_idx:qvel_idx+3])
                        # 铰链/滑动关节（1自由度）：仅取1维
                        elif joint_type in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                            joint_msg.position.append(data.qpos[qpos_idx])
                            joint_msg.velocity.append(data.qvel[qvel_idx])
                    
                    joint_state_pub.publish(joint_msg)

                    # 2. 发布基座姿态（原有逻辑不变）
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = rospy.Time.now()
                    pose_msg.header.frame_id = "world"  # 坐标系名称（可自定义）
                    
                    # 位置信息（x,y,z）
                    if model.nq >= 1:
                        pose_msg.pose.position.x = data.qpos[0]
                    if model.nq >= 2:
                        pose_msg.pose.position.y = data.qpos[1]
                    if model.nq >= 3:
                        pose_msg.pose.position.z = data.qpos[2]
                    
                    # 姿态信息（四元数 qx,qy,qz,qw）
                    if model.nq >= 4:
                        pose_msg.pose.orientation.x = data.qpos[3]
                    if model.nq >= 5:
                        pose_msg.pose.orientation.y = data.qpos[4]
                    if model.nq >= 6:
                        pose_msg.pose.orientation.z = data.qpos[5]
                    if model.nq >= 7:
                        pose_msg.pose.orientation.w = data.qpos[6]
                    
                    pose_pub.publish(pose_msg)

                    # 按ROS频率休眠，确保消息发布稳定
                    ros_rate.sleep()

        logger.info("可视化窗口已关闭")
    except Exception as e:
        logger.error(f"可视化过程出错: {str(e)}", exc_info=True)

# ===================== 主函数（完全保持原有逻辑不变）=====================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="MuJoCo功能整合工具（支持ROS 1消息封装）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. 可视化命令（新增--ros选项）
    viz_parser = subparsers.add_parser("visualize", help="可视化模型并运行模拟")
    viz_parser.add_argument("model", help="/home/lan/桌面/nn/mujoco_menagerie/anybotics_anymal_b")
    viz_parser.add_argument(
        "--ros",
        action="store_true",
        help="启用ROS模式（发布关节状态/基座姿态，订阅控制指令）"
    )

    # 2. 速度测试命令（原有功能不变）
    speed_parser = subparsers.add_parser("testspeed", help="测试模型模拟速度")
    speed_parser.add_argument("model", help="模型文件路径")
    speed_parser.add_argument("--nstep", type=int, default=10000, help="每线程模拟步数")
    speed_parser.add_argument("--nthread", type=int, default=1, help="测试线程数量")
    speed_parser.add_argument("--ctrlnoise", type=float, default=0.01, help="控制噪声强度")

    # 3. 模型转换命令（原有功能不变）
    convert_parser = subparsers.add_parser("convert", help="转换模型格式（XML↔MJB）")
    convert_parser.add_argument("input", help="输入模型路径")
    convert_parser.add_argument("output", help="输出模型路径（需指定.xml或.mjb扩展名）")

    args, unknown = parser.parse_known_args()


    # 命令映射（更新visualize，支持use_ros参数）
    command_handlers: Dict[str, callable] = {
        "visualize": lambda: visualize(args.model, use_ros=args.ros),
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



if __name__ == "__main__":
    main()