import sys
import os
import time
import logging
import argparse
import threading
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import mujoco
from mujoco import viewer

# ===================== 核心路径配置（相对路径）=====================
# 获取当前脚本所在目录（所有相对路径的基准）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 模型文件夹相对路径（相对于当前脚本）
MODEL_ROOT = os.path.join(SCRIPT_DIR, "mujoco_menagerie")

# ===================== 依赖导入 - ROS 1（保留但禁用）=====================
ROS_AVAILABLE = False
try:
    import rospy
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float32MultiArray
    ROS_AVAILABLE = True
except ImportError:
    logging.warning("ROS环境未检测到，ROS功能禁用")

# ===================== 多模型配置（修正预设指令 + 相对路径）=====================
MODEL_CONFIGS = {
    1: {
        "name": "Franka Panda（机械臂）",
        "key": "franka",
        "path": os.path.join(MODEL_ROOT, "franka_emika_panda/panda.xml"),
        "joint_num": 7,
        "pd_params": {"KP": 800.0, "KD": 60.0},
        "presets": {
            "home": np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0]),
            "up": np.array([0.2, 0.1, 0.0, -1.57, 0.2, 1.57, 0.0]),
            "left": np.array([0.0, 0.2, 0.0, -1.2, 0.0, 1.8, 0.0]),
            "right": np.array([0.0, -0.2, 0.0, -1.8, 0.0, 1.2, 0.0])
        },
        "default_preset": "home",
        "ee_site_name": "end_effector"
    },
    2: {
        "name": "UR5 机械臂",
        "key": "ur5",
        "path": os.path.join(MODEL_ROOT, "universal_robots_ur5e/ur5e.xml"),
        "joint_num": 6,
        "pd_params": {"KP": 700.0, "KD": 50.0},
        "presets": {
            "home": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "up": np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.0]),
            "forward": np.array([0.0, -0.5, 0.5, 0.0, 0.0, 0.0])
        },
        "default_preset": "home",
        "ee_site_name": "ee_link"
    },
    3: {
        "name": "Franka Panda（带手爪）",
        "key": "franka_gripper",
        "path": os.path.join(MODEL_ROOT, "franka_emika_panda/panda_gripper.xml"),
        "joint_num": 8,
        "pd_params": {"KP": 800.0, "KD": 60.0},
        "presets": {
            "home": np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0, 0.0]),
            "open_gripper": np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0, 1.0]),
            "up_open": np.array([0.2, 0.1, 0.0, -1.57, 0.2, 1.57, 0.0, 1.0])
        },
        "default_preset": "home",
        "ee_site_name": "end_effector"
    },
    4: {
        "name": "Walker2d 机器人",
        "key": "walker2d",
        "path": os.path.join(MODEL_ROOT, "walker2d/walker2d.xml"),
        "joint_num": 6,
        "pd_params": {"KP": 1000.0, "KD": 80.0},
        "presets": {
            "stand": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "walk_left": np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1]),
            "walk_right": np.array([-0.1, 0.1, -0.1, 0.1, -0.1, 0.1])
        },
        "default_preset": "stand",
        "ee_site_name": "torso"
    }
}

# ===================== 全局变量（精简版）=====================
CURRENT_CONFIG = None
TARGET_JOINT_POS = None
KP = None
KD = None
SIMULATION_PAUSE = False
SIMULATION_RUNNING = False
CMD_LOCK = threading.Lock()

# ===================== 日志配置（精简输出）=====================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("mujoco_control_tool")

# ===================== 核心功能函数（精简版 + 路径调试）=====================
def load_mujoco_model(model_path: str) -> Tuple[Optional[mujoco.MjModel], Optional[mujoco.MjData]]:
    """加载MuJoCo模型（增加路径调试信息）"""
    # 路径合法性检查 + 调试信息
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在：{model_path}")
        logger.info(f"💡 调试信息 - 当前脚本目录：{SCRIPT_DIR}")
        logger.info(f"💡 调试信息 - 模型根目录：{MODEL_ROOT}")
        return None, None

    try:
        if model_path.endswith('.mjb'):
            model = mujoco.MjModel.from_binary_path(model_path)
        else:
            model = mujoco.MjModel.from_xml_path(model_path)
        
        data = mujoco.MjData(model)
        logger.info(f"模型加载成功：{model_path}")
        logger.info(f"模型参数：控制维度={model.nu} | 关节数={model.njnt} | 自由度={model.nq}")
        return model, data
    except Exception as e:
        logger.error(f"模型加载失败：{str(e)}", exc_info=True)
        return None, None

def load_selected_model() -> Tuple[Optional[mujoco.MjModel], Optional[mujoco.MjData]]:
    """加载选中的模型"""
    if not CURRENT_CONFIG:
        logger.error("❌ 未选择模型！")
        return None, None
    
    model_path = CURRENT_CONFIG["path"]
    model, data = load_mujoco_model(model_path)
    if model and data:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        logger.info(f"\n✅ 成功加载模型：{CURRENT_CONFIG['name']}")
        logger.info(f"📂 模型路径：{model_path}")
        logger.info(f"🔧 控制关节数：{CURRENT_CONFIG['joint_num']} | PD参数：KP={KP}, KD={KD}")
    return model, data

def pd_controller(model: mujoco.MjModel, data: mujoco.MjData):
    """PD控制器"""
    joint_num = CURRENT_CONFIG["joint_num"]
    if model.nq < joint_num or model.nu < joint_num:
        logger.warning(f"⚠️  模型自由度/控制维度不足")
        return
    
    global TARGET_JOINT_POS
    current_joint_pos = data.qpos[:joint_num].copy()
    current_joint_vel = data.qvel[:joint_num].copy()
    
    # PD控制核心计算
    pos_error = TARGET_JOINT_POS - current_joint_pos
    vel_error = -current_joint_vel
    joint_torque = KP * pos_error + KD * vel_error
    
    # 力矩限位
    if model.actuator_forcerange.size >= joint_num:
        torque_limit = model.actuator_forcerange[:, 1].copy()[:joint_num]
        joint_torque = np.clip(joint_torque, -torque_limit, torque_limit)
    
    data.ctrl[:joint_num] = joint_torque

def simulation_worker(model, data, viewer_instance):
    """仿真工作线程"""
    global SIMULATION_RUNNING, SIMULATION_PAUSE
    step_counter = 0
    sim_frequency = 50
    sim_interval = 1.0 / sim_frequency
    
    # 适配相机视角
    cam_distance = 2.0 if "机械臂" in CURRENT_CONFIG["name"] else 3.0
    viewer_instance.cam.distance = cam_distance
    viewer_instance.cam.azimuth = 90
    viewer_instance.cam.elevation = -20
    viewer_instance.cam.lookat = [0.0, 0.0, 0.5]
    
    while SIMULATION_RUNNING and viewer_instance.is_running():
        if SIMULATION_PAUSE:
            time.sleep(0.1)
            continue
        
        loop_start = time.time()
        
        # 执行PD控制
        pd_controller(model, data)
        
        # 步进仿真
        mujoco.mj_step(model, data)
        viewer_instance.sync()

        # 大幅降低打印频率（从200步→500步），减少刷屏
        if step_counter % 500 == 0:
            print_simulation_status(data, step_counter)
        step_counter += 1
        
        # 控制仿真频率
        loop_duration = time.time() - loop_start
        if loop_duration < sim_interval:
            time.sleep(sim_interval - loop_duration)

def cmd_listener_main():
    """终端指令监听（主线程，移除save指令）"""
    global TARGET_JOINT_POS, SIMULATION_PAUSE, SIMULATION_RUNNING
    joint_num = CURRENT_CONFIG["joint_num"]
    preset_keys = list(CURRENT_CONFIG["presets"].keys())
    
    # 修正指令提示，只显示当前模型支持的预设
    logger.info("\n" + "="*50)
    logger.info(f"📢 {CURRENT_CONFIG['name']} 控制指令说明")
    logger.info(f"  1. 预设位置：{' / '.join(preset_keys)}（直接输入即可切换）")
    logger.info(f"  2. 自定义关节：set 关节号 角度（示例：set 0 0.5，单位rad）")
    logger.info(f"  3. 暂停/继续仿真：pause / resume")
    logger.info(f"  4. 退出仿真：exit")
    logger.info("="*50 + "\n")
    
    while SIMULATION_RUNNING:
        try:
            # 输入提示符单独显示，更清晰
            cmd = input("\n👉 请输入控制指令：").strip().lower()
            
            with CMD_LOCK:
                pass  # 移除指令存储，直接处理
            
            # 核心指令逻辑（移除save指令）
            if cmd == "exit":
                logger.info("📤 收到退出指令，即将关闭仿真...")
                SIMULATION_RUNNING = False
                break
            elif cmd == "pause":
                SIMULATION_PAUSE = True
                logger.info("⏸️  仿真已暂停（输入resume继续）")
            elif cmd == "resume":
                SIMULATION_PAUSE = False
                logger.info("▶️  仿真已继续")
            elif cmd in preset_keys:
                TARGET_JOINT_POS = CURRENT_CONFIG["presets"][cmd]
                logger.info(f"\n🎯 切换到预设位置：{cmd}")
                logger.info(f"🎯 目标关节位置：{TARGET_JOINT_POS.round(3)}")
            elif cmd.startswith("set"):
                parts = cmd.split()
                if len(parts) != 3:
                    logger.error("❌ set指令格式错误！正确示例：set 0 0.5")
                    continue
                try:
                    joint_idx = int(parts[1])
                    joint_angle = float(parts[2])
                    if 0 <= joint_idx < joint_num:
                        TARGET_JOINT_POS[joint_idx] = joint_angle
                        logger.info(f"\n🔧 关节{joint_idx}目标角度设为：{joint_angle} rad")
                        logger.info(f"🔍 当前完整目标位置：{TARGET_JOINT_POS.round(3)}")
                    else:
                        logger.error(f"❌ 关节号无效！必须是0-{joint_num-1}之间的整数")
                except ValueError:
                    logger.error("❌ 关节号/角度必须是数字！示例：set 0 0.5")
            else:
                logger.error(f"❌ 未知指令！支持的指令：{' / '.join(preset_keys)}、set、pause、resume、exit")
        
        except KeyboardInterrupt:
            logger.info("\n⚠️  检测到键盘中断，即将退出...")
            SIMULATION_RUNNING = False
            break
        except Exception as e:
            logger.error(f"❌ 指令解析失败：{str(e)}", exc_info=True)

def print_simulation_status(data: mujoco.MjData, step: int):
    """打印仿真状态（精简输出）"""
    joint_num = CURRENT_CONFIG["joint_num"]
    current_pos = data.qpos[:joint_num].round(4)
    pos_error = TARGET_JOINT_POS - current_pos
    avg_error = np.mean(np.abs(pos_error))
    max_error = np.max(np.abs(pos_error))
    
    logger.info(f"\n===== 仿真进度：{data.time:.2f}秒 | 第{step}步 =====")
    logger.info(f"🎯 目标关节位置：{TARGET_JOINT_POS.round(3)}")
    logger.info(f"📝 当前关节位置：{current_pos}")
    logger.info(f"📊 控制精度：平均误差={avg_error:.6f}rad | 最大误差={max_error:.6f}rad")

def run_selected_model():
    """启动选中模型的可视化与控制（精简版）"""
    # 加载模型
    model, data = load_selected_model()
    if not model or not data:
        input("\n按回车键返回模型选择菜单...")
        return
    
    # 初始化全局标志
    global SIMULATION_PAUSE, SIMULATION_RUNNING
    SIMULATION_PAUSE = False
    SIMULATION_RUNNING = True
    
    logger.info(f"\n🖥️  正在启动 {CURRENT_CONFIG['name']} 可视化窗口...")
    logger.info("💡 提示：窗口弹出后，终端会显示输入提示符，可输入指令控制模型！")
    
    try:
        with viewer.launch_passive(model, data) as viewer_instance:
            # 启动仿真线程
            sim_thread = threading.Thread(
                target=simulation_worker,
                args=(model, data, viewer_instance),
                daemon=True
            )
            sim_thread.start()
            
            # 主线程运行指令监听
            cmd_listener_main()
            
            # 等待仿真线程结束
            sim_thread.join(timeout=2)
        
        logger.info(f"\n✅ {CURRENT_CONFIG['name']} 仿真已关闭")
            
    except Exception as e:
        logger.error(f"\n❌ 可视化出错：{str(e)}", exc_info=True)
        SIMULATION_RUNNING = False
    
    input("\n按回车键返回模型选择菜单...")

def show_menu():
    """显示模型选择菜单"""
    os.system("clear") if os.name != "nt" else os.system("cls")
    print("="*60)
    print("          🚀 MuJoCo 多模型控制平台 🚀          ")
    print("="*60)
    print("请选择要运行的模型（输入数字序号）：")
    for idx, config in MODEL_CONFIGS.items():
        print(f"  {idx}. {config['name']}")
    print("  0. 退出程序")
    print("="*60)

def main_menu():
    """交互式菜单主函数"""
    global CURRENT_CONFIG, TARGET_JOINT_POS, KP, KD
    
    while True:
        show_menu()
        try:
            choice = int(input("\n👉 请输入选择（0-4）：").strip())
        except ValueError:
            logger.error("❌ 输入无效！请输入数字0-4")
            input("按回车键重新选择...")
            continue
        
        if choice == 0:
            logger.info("\n👋 感谢使用，程序已退出！")
            sys.exit(0)
        elif choice in MODEL_CONFIGS:
            CURRENT_CONFIG = MODEL_CONFIGS[choice]
            TARGET_JOINT_POS = CURRENT_CONFIG["presets"][CURRENT_CONFIG["default_preset"]]
            KP = CURRENT_CONFIG["pd_params"]["KP"]
            KD = CURRENT_CONFIG["pd_params"]["KD"]
            run_selected_model()
        else:
            logger.error("❌ 选择无效！请输入0-4之间的数字")
            input("按回车键重新选择...")

# ===================== 命令行入口（精简版）=====================
def main():
    # 启动时打印路径信息，方便调试
    logger.info(f"📌 当前脚本目录：{SCRIPT_DIR}")
    logger.info(f"📌 模型根目录：{MODEL_ROOT}")
    
    parser = argparse.ArgumentParser(
        description="MuJoCo模型控制工具（精简版，无数据保存）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True, help="子命令列表")

    # 只保留menu子命令（核心控制功能）
    menu_parser = subparsers.add_parser("menu", help="启动交互式模型选择菜单（PD控制）")

    args = parser.parse_args()

    subcommand_handlers = {
        "menu": main_menu
    }

    try:
        subcommand_handlers[args.subcommand]()
    except KeyError:
        logger.error(f"未知子命令：{args.subcommand}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"程序执行失败：{str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n👋 程序被用户中断，已退出！")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"\n❌ 程序运行出错：{str(e)}", exc_info=True)
        sys.exit(1)