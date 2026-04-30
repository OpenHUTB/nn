"""游戏手柄相关函数和状态模块"""
import pygame
import numpy as np
import json
import os
from .mode_utils import MODE_WALK, MODE_RUN, MODE_DISTURBANCE, mode_names, get_cmd_max_for_mode
from .viewer_utils import (
    track_pelvis, show_contacts, show_forces, reset_requested,
    camera_angle, camera_elevation, camera_distance,
    camera_angle_speed, camera_distance_speed
)
from . import viewer_utils  # 用于修改全局变量

# 游戏手柄相关全局变量
joystick = None
deadzone = 0.1  # 摇杆死区，小于此值视为无输入
button_states = {}  # 按钮状态缓存，用于检测按钮按下事件（从释放到按下）
gamepad_calibration = None  # 手柄校准数据
# 轴映射：根据手柄类型确定（左摇杆X, 左摇杆Y, 右摇杆X, 右摇杆Y）
axis_mapping = [0, 1, 2, 3]  # 默认罗技映射

# 模式控制（从mode_utils导入，但需要本地状态）
current_mode = MODE_WALK  # 当前模式

# 抗扰动测试相关（从disturbance_utils导入）
disturbance_force_scale = 100.0  # 扰动力缩放系数（N）
disturbance_body_name = "torso_link"  # 默认施加扰动力的body


def set_axis_mapping_from_type(gamepad_type, axis_mapping_custom=None):
    """根据手柄类型设置轴映射
    
    Args:
        gamepad_type: 手柄类型 ('logitech', 'betop', 'custom')
        axis_mapping_custom: 自定义轴映射 [左摇杆X, 左摇杆Y, 右摇杆X, 右摇杆Y]，仅在gamepad_type='custom'时使用
    """
    global axis_mapping
    
    if gamepad_type == 'betop':
        axis_mapping = [0, 1, 3, 4]  # 北通：左摇杆0,1，右摇杆3,4
        print(f"  手柄类型: 北通 (Betop) - 轴映射: 左摇杆({axis_mapping[0]},{axis_mapping[1]}), 右摇杆({axis_mapping[2]},{axis_mapping[3]})")
    elif gamepad_type == 'custom':
        if axis_mapping_custom is not None and len(axis_mapping_custom) == 4:
            axis_mapping = axis_mapping_custom
            print(f"  手柄类型: 自定义 - 轴映射: 左摇杆({axis_mapping[0]},{axis_mapping[1]}), 右摇杆({axis_mapping[2]},{axis_mapping[3]})")
        else:
            print(f"  警告: 自定义手柄类型但未提供轴映射，使用默认（罗技）")
            axis_mapping = [0, 1, 2, 3]
    else:
        axis_mapping = [0, 1, 2, 3]  # 默认罗技
        print(f"  手柄类型: 罗技 (Logitech) - 轴映射: 左摇杆({axis_mapping[0]},{axis_mapping[1]}), 右摇杆({axis_mapping[2]},{axis_mapping[3]})")


def get_calibration_file_path(gamepad_type=None, custom_path=None):
    """获取校准文件路径
    
    Args:
        gamepad_type: 手柄类型，如果提供则生成对应的文件名
        custom_path: 自定义校准文件路径（从yaml配置中读取）
    
    Returns:
        校准文件路径
    """
    # 获取deploy_mujoco目录（utils的父目录）
    deploy_dir = os.path.dirname(os.path.dirname(__file__))
    
    # 如果指定了自定义路径，直接使用
    if custom_path:
        # 如果是相对路径，相对于deploy_mujoco目录
        if not os.path.isabs(custom_path):
            return os.path.join(deploy_dir, custom_path)
        return custom_path
    
    # 如果指定了手柄类型，生成对应的文件名
    if gamepad_type:
        base_name = f"gamepad_calibration_{gamepad_type}.json"
        return os.path.join(deploy_dir, base_name)
    
    # 默认返回None，让调用者处理
    return None


def load_gamepad_calibration(config=None):
    """加载游戏手柄校准文件和配置
    
    Args:
        config: 配置文件字典，可能包含：
            - gamepad_type: 手柄类型
            - axis_mapping: 自定义轴映射
            - gamepad_calibration_file: 校准文件路径（可选，如果指定则使用此文件）
    """
    global gamepad_calibration, axis_mapping
    
    # 优先级：配置文件中的校准文件 > 配置文件中的gamepad_type > 校准文件 > 默认
    gamepad_type = None
    axis_mapping_custom = None
    calibration_file_path = None
    
    # 1. 首先从配置文件中读取
    if config is not None:
        # 优先读取指定的校准文件路径
        calibration_file_path = config.get('gamepad_calibration_file')
        gamepad_type = config.get('gamepad_type')
        axis_mapping_custom = config.get('axis_mapping')
        
        # 如果指定了校准文件路径，直接加载
        if calibration_file_path:
            calibration_file = get_calibration_file_path(custom_path=calibration_file_path)
            if os.path.exists(calibration_file):
                try:
                    with open(calibration_file, 'r', encoding='utf-8') as f:
                        gamepad_calibration = json.load(f)
                    print(f"从配置文件中指定的路径加载校准文件: {calibration_file}")
                    print(f"  手柄名称: {gamepad_calibration.get('joystick_name', 'Unknown')}")
                    print(f"  校准日期: {gamepad_calibration.get('calibration_date', 'Unknown')}")
                    
                    # 从校准文件中获取手柄类型（如果配置文件中没有指定）
                    if not gamepad_type:
                        gamepad_type = gamepad_calibration.get('gamepad_type', 'logitech')
                    
                    # 处理自定义映射
                    if gamepad_type == 'custom':
                        axes = gamepad_calibration.get('axes', {})
                        if len(axes) >= 4 and not axis_mapping_custom:
                            axis_mapping_custom = [None, None, None, None]
                            for axis_id_str, axis_cal in axes.items():
                                axis_id = int(axis_id_str)
                                name = axis_cal.get('name', '')
                                if '左摇杆X' in name or '角速度' in name:
                                    axis_mapping_custom[0] = axis_id
                                elif '左摇杆Y' in name or '扰动力' in name:
                                    axis_mapping_custom[1] = axis_id
                                elif '右摇杆X' in name or '左右速度' in name:
                                    axis_mapping_custom[2] = axis_id
                                elif '右摇杆Y' in name or '前后速度' in name:
                                    axis_mapping_custom[3] = axis_id
                            
                            if None in axis_mapping_custom:
                                axis_ids = sorted([int(k) for k in axes.keys()])
                                if len(axis_ids) >= 4:
                                    axis_mapping_custom = axis_ids[:4]
                    
                    set_axis_mapping_from_type(gamepad_type, axis_mapping_custom)
                    return gamepad_calibration
                except Exception as e:
                    print(f"警告: 加载校准文件失败: {e}")
                    gamepad_calibration = None
        
        # 如果配置文件中指定了gamepad_type，但没有指定校准文件，则根据类型生成文件名
        if gamepad_type and not calibration_file_path:
            print(f"从配置文件中读取手柄类型: {gamepad_type}")
            if axis_mapping_custom:
                print(f"从配置文件中读取自定义轴映射: {axis_mapping_custom}")
            set_axis_mapping_from_type(gamepad_type, axis_mapping_custom)
            
            # 根据gamepad_type生成校准文件路径
            calibration_file = get_calibration_file_path(gamepad_type=gamepad_type)
            if os.path.exists(calibration_file):
                try:
                    with open(calibration_file, 'r', encoding='utf-8') as f:
                        gamepad_calibration = json.load(f)
                    print(f"已加载手柄校准文件（用于按钮和轴中位值校准）: {calibration_file}")
                except Exception as e:
                    print(f"警告: 加载校准文件失败: {e}")
            else:
                print(f"提示: 未找到校准文件 {calibration_file}")
                print("      可以运行 calibrate_gamepad.py 进行校准")
            return gamepad_calibration
    
    # 2. 如果没有配置文件或配置文件中没有，尝试从校准文件读取（按优先级尝试）
    if gamepad_type is None:
        # 按优先级尝试加载：betop > logitech > custom
        for gp_type in ['betop', 'logitech', 'custom']:
            calibration_file = get_calibration_file_path(gamepad_type=gp_type)
            if os.path.exists(calibration_file):
                try:
                    with open(calibration_file, 'r', encoding='utf-8') as f:
                        gamepad_calibration = json.load(f)
                    print(f"已加载手柄校准文件: {calibration_file}")
                    print(f"  手柄名称: {gamepad_calibration.get('joystick_name', 'Unknown')}")
                    print(f"  校准日期: {gamepad_calibration.get('calibration_date', 'Unknown')}")
                    
                    # 从校准文件中获取手柄类型
                    gamepad_type = gamepad_calibration.get('gamepad_type', 'logitech')
                    
                    if gamepad_type == 'custom':
                        # 自定义映射，从校准的轴中按功能顺序提取
                        axes = gamepad_calibration.get('axes', {})
                        if len(axes) >= 4:
                            # 按照功能名称匹配：左摇杆X, 左摇杆Y, 右摇杆X, 右摇杆Y
                            axis_mapping_custom = [None, None, None, None]
                            for axis_id_str, axis_cal in axes.items():
                                axis_id = int(axis_id_str)
                                name = axis_cal.get('name', '')
                                if '左摇杆X' in name or '角速度' in name:
                                    axis_mapping_custom[0] = axis_id
                                elif '左摇杆Y' in name or '扰动力' in name:
                                    axis_mapping_custom[1] = axis_id
                                elif '右摇杆X' in name or '左右速度' in name:
                                    axis_mapping_custom[2] = axis_id
                                elif '右摇杆Y' in name or '前后速度' in name:
                                    axis_mapping_custom[3] = axis_id
                            
                            # 如果匹配失败，按轴ID排序作为后备
                            if None in axis_mapping_custom:
                                axis_ids = sorted([int(k) for k in axes.keys()])
                                if len(axis_ids) >= 4:
                                    axis_mapping_custom = axis_ids[:4]
                    
                    set_axis_mapping_from_type(gamepad_type, axis_mapping_custom)
                    break
                except Exception as e:
                    print(f"警告: 加载校准文件失败: {e}")
                    gamepad_calibration = None
                    continue
        
        if gamepad_calibration is None:
            print(f"提示: 未找到校准文件")
            print("      可以运行 calibrate_gamepad.py 进行校准")
    
    # 3. 如果都没有，使用默认
    if gamepad_type is None:
        print("使用默认轴映射（罗技：左摇杆0,1，右摇杆2,3）")
        set_axis_mapping_from_type('logitech')
    
    return gamepad_calibration


def apply_axis_calibration(axis_id, raw_value):
    """应用轴校准：减去中位值（不再反转方向）"""
    global gamepad_calibration
    
    if gamepad_calibration is None:
        return raw_value
    
    axes = gamepad_calibration.get('axes', {})
    # 尝试字符串键和整数键
    axis_cal = axes.get(str(axis_id)) or axes.get(axis_id)
    
    if axis_cal is None:
        return raw_value
    
    # 减去中位值（不再反转方向）
    calibrated_value = raw_value - axis_cal.get('center', 0.0)
    
    return calibrated_value


def get_calibrated_button_id(button_name, default_ids=None):
    """获取校准后的按钮ID
    
    Args:
        button_name: 按钮名称 ('LB', 'RB', 'X', 'Y', 'B', 'A')
        default_ids: 默认按钮ID列表（如果未校准，尝试这些ID）
    
    Returns:
        按钮ID，如果未找到返回None
    """
    global gamepad_calibration
    
    if gamepad_calibration is None:
        # 未校准，使用默认ID
        if default_ids is not None:
            return default_ids
        return None
    
    buttons = gamepad_calibration.get('buttons', {})
    button_cal = buttons.get(button_name)
    
    if button_cal is not None:
        return button_cal.get('button_id')
    
    # 如果未找到校准，使用默认ID
    if default_ids is not None:
        return default_ids
    return None


def init_gamepad(config=None):
    """初始化游戏手柄
    
    Args:
        config: 配置文件字典，可能包含 gamepad_type 和 axis_mapping 配置
    """
    global joystick
    pygame.init()
    pygame.joystick.init()
    
    # 检查是否有连接的游戏手柄
    if pygame.joystick.get_count() == 0:
        print("警告: 未检测到游戏手柄，将使用默认速度命令")
        return None
    
    # 初始化第一个游戏手柄
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"游戏手柄已连接: {joystick.get_name()}")
    
    # 加载校准文件和配置
    load_gamepad_calibration(config)
    
    print_gamepad_controls()
    return joystick


def print_gamepad_controls():
    """打印游戏手柄控制说明"""
    print("=" * 50)
    print("游戏手柄控制说明:")
    print("  右摇杆上下: 控制前后速度 (cmd[0])")
    print("  右摇杆左右: 控制左右速度 (cmd[1])")
    print("  左摇杆左右: 控制角速度 (cmd[2])，抗扰动模式下不使用")
    print("  左摇杆: 抗扰动模式下控制扰动力（X轴=左右，Y轴=前后）")
    print("  方向键左右: 围绕机器人旋转视角（跟踪pelvis时）")
    print("  方向键上下: 控制视角远近（跟踪pelvis时）")
    print("  LB按钮: 切换模式（行走/奔跑/抗扰动测试）")
    print("  X按钮: 切换显示足底力")
    print("  Y按钮: 切换是否跟踪pelvis")
    print("  B按钮: 重置机器人状态")
    print("  A按钮: 切换显示足端接触状态")
    print("=" * 50)


def handle_mode_switch():
    """处理模式切换（LB按钮）"""
    global current_mode
    if joystick is None:
        return
    
    # 获取LB按钮ID（校准后或默认值）
    lb_button_ids = get_calibrated_button_id('LB', default_ids=[4, 6])
    if lb_button_ids is None:
        return
    
    # 支持单个ID或ID列表
    if isinstance(lb_button_ids, list):
        lb_pressed = False
        for button_id in lb_button_ids:
            if joystick.get_numbuttons() > button_id:
                if joystick.get_button(button_id):
                    lb_pressed = True
                    break
    else:
        button_id = lb_button_ids
        lb_pressed = joystick.get_button(button_id) if joystick.get_numbuttons() > button_id else False
    
    if lb_pressed and not button_states.get('lb', False):
        current_mode = (current_mode + 1) % 3  # 循环切换：0->1->2->0
        print(f"模式已切换: {mode_names[current_mode]} (模式 {current_mode})")
        if current_mode == MODE_WALK:
            print("  - x前进最大速度: 1.2 m/s")
            print("  - x后退最大速度: 0.8 m/s")
        elif current_mode == MODE_RUN:
            print("  - x前进最大速度: 2.0 m/s")
            print("  - x后退最大速度: 0.8 m/s")
        elif current_mode == MODE_DISTURBANCE:
            print("  - 抗扰动测试模式：左手摇杆控制扰动力")
            print(f"  - 扰动力施加在: {disturbance_body_name}")
    
    # 更新按钮状态
    button_states['lb'] = lb_pressed


def handle_viewer_controls():
    """处理viewer相关的按钮控制"""
    if joystick is None:
        return
    
    # 如果跟踪pelvis，使用方向键（D-pad/hat）调节视角
    if viewer_utils.track_pelvis and joystick.get_numhats() > 0:
        hat = joystick.get_hat(0)  # 读取第一个 hat switch
        hat_x, hat_y = hat
        
        # 方向键左右：围绕机器人旋转视角（调整水平角度）
        if hat_x != 0:
            viewer_utils.camera_angle += hat_x * viewer_utils.camera_angle_speed
            # 将角度限制在 [0, 2π] 范围内
            viewer_utils.camera_angle = viewer_utils.camera_angle % (2 * np.pi)
        
        # 方向键上下：控制视角远近（调整距离）
        if hat_y != 0:
            viewer_utils.camera_distance += hat_y * viewer_utils.camera_distance_speed
            # 限制距离范围
            viewer_utils.camera_distance = np.clip(viewer_utils.camera_distance, 0.5, 10.0)
    
    # X按钮: 切换显示足底力
    x_button_id = get_calibrated_button_id('X', default_ids=[0])
    if x_button_id is not None and joystick.get_numbuttons() > x_button_id:
        button_x_pressed = joystick.get_button(x_button_id)
        if button_x_pressed and not button_states.get('x', False):
            viewer_utils.show_forces = not viewer_utils.show_forces
            print(f"{'已启用' if viewer_utils.show_forces else '已禁用'}足底力显示")
        button_states['x'] = button_x_pressed
    
    # Y按钮: 切换是否跟踪pelvis
    y_button_id = get_calibrated_button_id('Y', default_ids=[3])
    if y_button_id is not None and joystick.get_numbuttons() > y_button_id:
        button_y_pressed = joystick.get_button(y_button_id)
        if button_y_pressed and not button_states.get('y', False):
            viewer_utils.track_pelvis = not viewer_utils.track_pelvis
            if viewer_utils.track_pelvis:
                print("已启用pelvis跟踪（相机以机器人为中心）")
                # 重置为初始视角（正对机器人）
                viewer_utils.camera_angle = 0.0
                viewer_utils.camera_elevation = -0.3
                viewer_utils.camera_distance = 3.0
            else:
                print("已禁用pelvis跟踪")
        button_states['y'] = button_y_pressed
    
    # B按钮: 重置机器人状态
    b_button_id = get_calibrated_button_id('B', default_ids=[2])
    if b_button_id is not None and joystick.get_numbuttons() > b_button_id:
        button_b_pressed = joystick.get_button(b_button_id)
        if button_b_pressed and not button_states.get('b', False):
            viewer_utils.reset_requested = True
            print("请求重置机器人状态")
        button_states['b'] = button_b_pressed
    
    # A按钮: 切换显示足端接触状态
    a_button_id = get_calibrated_button_id('A', default_ids=[1])
    if a_button_id is not None and joystick.get_numbuttons() > a_button_id:
        button_a_pressed = joystick.get_button(a_button_id)
        if button_a_pressed and not button_states.get('a', False):
            viewer_utils.show_contacts = not viewer_utils.show_contacts
            print(f"{'已启用' if viewer_utils.show_contacts else '已禁用'}足端接触状态显示")
        button_states['a'] = button_a_pressed


def update_cmd_from_gamepad(cmd):
    """从游戏手柄读取输入并更新速度命令
    
    轴映射（根据手柄类型）:
    - 右摇杆 Y轴: 前后速度 (cmd[0]), 向上为负值，向下为正值
    - 右摇杆 X轴: 左右速度 (cmd[1]), 向左为负值，向右为正值
    - 左摇杆 X轴: 角速度 (cmd[2]), 向左为负值，向右为正值
    - 左摇杆 Y轴: 抗扰动模式下控制扰动力（向上为负，向下为正）
    """
    global current_mode
    
    if joystick is None:
        return cmd
    
    # 处理 pygame 事件（必须调用以更新游戏手柄状态）
    pygame.event.pump()
    
    # 处理模式切换
    handle_mode_switch()
    
    # 处理viewer控制
    handle_viewer_controls()
    
    # 根据模式获取速度限制
    cmd_max = get_cmd_max_for_mode(current_mode)
    
    # 读取摇杆输入（使用校准后的轴映射）
    # axis_mapping: [左摇杆X, 左摇杆Y, 右摇杆X, 右摇杆Y]
    raw_right_stick_y = -joystick.get_axis(axis_mapping[3])  # 右摇杆Y轴
    raw_right_stick_x = -joystick.get_axis(axis_mapping[2])  # 右摇杆X轴
    raw_left_stick_x = -joystick.get_axis(axis_mapping[0])   # 左摇杆X轴
    
    # 应用校准（使用实际的轴ID）
    right_stick_y = apply_axis_calibration(axis_mapping[3], raw_right_stick_y)
    right_stick_x = apply_axis_calibration(axis_mapping[2], raw_right_stick_x)
    left_stick_x = apply_axis_calibration(axis_mapping[0], raw_left_stick_x)
    
    # 应用默认方向（如果未校准，保持原有逻辑）
    if gamepad_calibration is None:
        # 未校准时的默认处理
        right_stick_y = -right_stick_y  # 取反，因为向上推时轴值为负
        right_stick_x = -right_stick_x
        left_stick_x = -left_stick_x
    # 如果已校准，方向已在 apply_axis_calibration 中处理
    
    # 应用死区
    if abs(right_stick_y) < deadzone:
        right_stick_y = 0.0
    if abs(right_stick_x) < deadzone:
        right_stick_x = 0.0
    if abs(left_stick_x) < deadzone:
        left_stick_x = 0.0
    
    # 根据模式更新速度命令
    # 所有模式都允许控制速度（包括抗扰动模式）
    # 对于x方向，需要区分前进和后退
    if right_stick_y >= 0:
        # 前进：使用前进最大速度
        cmd[0] = right_stick_y * cmd_max[0]
    else:
        # 后退：使用后退最大速度
        cmd[0] = right_stick_y * cmd_max[1]
    
    cmd[1] = right_stick_x * cmd_max[2]  # y速度
    
    # 角速度控制：抗扰动模式下不使用角速度
    if current_mode == MODE_DISTURBANCE:
        cmd[2] = 0.0  # 抗扰动模式下不控制角速度
    else:
        cmd[2] = left_stick_x * cmd_max[3]  # 其他模式下使用左摇杆X轴控制角速度
    
    return cmd


def get_disturbance_force_base():
    """获取抗扰动模式下的扰动力（基于左摇杆输入），返回base坐标系下的力"""
    global current_mode
    
    if joystick is None or current_mode != MODE_DISTURBANCE:
        return np.zeros(3)
    
    # 在抗扰动模式下：
    # - 左摇杆X轴：控制扰动力左右方向
    # - 左摇杆Y轴：控制扰动力前后方向
    # - 右摇杆：控制速度命令（x, y方向）
    # - 不使用角速度
    # 使用校准后的轴映射
    raw_left_stick_x = joystick.get_axis(axis_mapping[0])  # 左摇杆X轴：左右方向的扰动力
    raw_left_stick_y = joystick.get_axis(axis_mapping[1])  # 左摇杆Y轴：前后方向的扰动力
    
    # 应用校准（使用实际的轴ID）
    left_stick_x = apply_axis_calibration(axis_mapping[0], raw_left_stick_x)
    left_stick_y = apply_axis_calibration(axis_mapping[1], raw_left_stick_y)
    
    # 应用默认方向（如果未校准，保持原有逻辑）
    left_stick_x = -left_stick_x
    left_stick_y = -left_stick_y
    # 如果已校准，方向已在 apply_axis_calibration 中处理
    
    # 应用死区
    if abs(left_stick_x) < deadzone:
        left_stick_x = 0.0
    if abs(left_stick_y) < deadzone:
        left_stick_y = 0.0
    
    # 计算扰动力大小（摇杆幅度）
    force_magnitude = np.sqrt(left_stick_x**2 + left_stick_y**2)
    force_magnitude = np.clip(force_magnitude, 0.0, 1.0)  # 限制在[0, 1]
    
    # 计算扰动力方向（摇杆方向）
    if force_magnitude > 0:
        # 归一化方向向量
        direction_x = left_stick_x / force_magnitude if force_magnitude > 0 else 0.0
        direction_y = left_stick_y / force_magnitude if force_magnitude > 0 else 0.0
    else:
        direction_x = 0.0
        direction_y = 0.0
    
    # 构建扰动力向量（在base坐标系下，水平面内，x和y方向）
    # 注意：base坐标系中，x通常是前进方向，y是左右方向
    force_base = np.array([
        direction_y * force_magnitude * disturbance_force_scale,  # 前后方向（左摇杆Y轴）
        direction_x * force_magnitude * disturbance_force_scale,  # 左右方向（左摇杆X轴）
        0.0  # z方向不施加力（只施加水平力）
    ])
    
    return force_base


def get_current_mode():
    """获取当前模式"""
    return current_mode

