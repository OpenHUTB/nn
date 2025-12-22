#!/usr/bin/env python3
"""
游戏手柄校准脚本
用于校准手柄摇杆的中位值（零位），解决机器人一直偏的问题
生成的校准文件可以被 deploy_mujoco_plot_20dof_gamepad.py 使用

功能：
- 一次性校准所有摇杆的中位值（简化流程）
- 保存手柄类型（罗技/北通/自定义）
- 校准所有按钮映射
"""

import pygame
import json
import os
import time
import numpy as np

# 校准文件保存路径（根据手柄类型添加后缀）
def get_calibration_file_path(gamepad_type):
    """根据手柄类型生成校准文件路径"""
    base_name = f"gamepad_calibration_{gamepad_type}.json"
    return os.path.join(os.path.dirname(__file__), base_name)

def init_gamepad():
    """初始化游戏手柄"""
    pygame.init()
    pygame.joystick.init()
    
    if pygame.joystick.get_count() == 0:
        print("错误: 未检测到游戏手柄！")
        return None
    
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"检测到游戏手柄: {joystick.get_name()}")
    print(f"  轴数量: {joystick.get_numaxes()}")
    print(f"  按钮数量: {joystick.get_numbuttons()}")
    print(f"  方向键数量: {joystick.get_numhats()}")
    return joystick

def calibrate_all_axes_at_once(joystick, axes_to_calibrate):
    """一次性校准所有轴的中位值"""
    print(f"\n{'='*60}")
    print("校准所有摇杆中位值")
    print(f"{'='*60}")
    print("请将所有摇杆移动到中位（不施加任何力）")
    print("准备好后按任意键开始采样...")
    input()
    
    # 采集中位值（采样多次取平均）
    print("正在采集中位值（采样100次，约1秒）...")
    samples = {axis_id: [] for axis_id in axes_to_calibrate.keys()}
    
    for _ in range(100):
        pygame.event.pump()
        for axis_id in axes_to_calibrate.keys():
            value = joystick.get_axis(axis_id)
            samples[axis_id].append(value)
        time.sleep(0.01)
    
    # 计算每个轴的中位值和标准差
    calibration_data = {}
    print("\n校准结果：")
    print("-"*60)
    for axis_id, axis_name in sorted(axes_to_calibrate.items()):
        center_value = np.mean(samples[axis_id])
        std_value = np.std(samples[axis_id])
        print(f"  轴 {axis_id:2d} ({axis_name:30s}): 中位值 = {center_value:8.4f} (标准差: {std_value:.4f})")
        
        calibration_data[axis_id] = {
            'axis_id': axis_id,
            'name': axis_name,
            'center': float(center_value),
            'reverse': False,  # 不再询问反转，默认不反转
            'std': float(std_value)
        }
    
    print("-"*60)
    return calibration_data

def calibrate_button(joystick, button_name, button_function):
    """校准单个按钮"""
    print(f"\n{'='*60}")
    print(f"校准 {button_name} 按钮 ({button_function})")
    print(f"{'='*60}")
    print(f"请按下 {button_name} 按钮（按住不放）...")
    print("（如果不想校准此按钮，直接按回车跳过）")
    print("（检测到按钮后会自动继续）")
    
    # 实时检测按钮按下
    print("正在检测按钮...", end="", flush=True)
    start_time = time.time()
    timeout = 10.0  # 10秒超时
    
    while time.time() - start_time < timeout:
        pygame.event.pump()
        
        # 检测所有按钮
        pressed_buttons = []
        for button_id in range(joystick.get_numbuttons()):
            if joystick.get_button(button_id):
                pressed_buttons.append(button_id)
        
        if len(pressed_buttons) == 1:
            button_id = pressed_buttons[0]
            print(f"\n  检测到按钮ID: {button_id}")
            
            # 等待按钮释放
            print("  请释放按钮...", end="", flush=True)
            while joystick.get_button(button_id):
                pygame.event.pump()
                time.sleep(0.01)
            
            print(" 已释放")
            
            # 确认：再次按下
            print(f"  请再次按下 {button_name} 按钮确认...", end="", flush=True)
            confirm_start = time.time()
            confirmed = False
            
            while time.time() - confirm_start < 5.0:  # 5秒内确认
                pygame.event.pump()
                if joystick.get_button(button_id):
                    confirmed = True
                    # 等待释放
                    while joystick.get_button(button_id):
                        pygame.event.pump()
                        time.sleep(0.01)
                    break
                time.sleep(0.01)
            
            if confirmed:
                print(" 确认成功！")
                print(f"  {button_name} 按钮映射到 button {button_id}")
                return {
                    'button_id': button_id,
                    'name': button_name,
                    'function': button_function
                }
            else:
                print(" 确认超时，跳过此按钮")
                return None
        
        elif len(pressed_buttons) > 1:
            print(f"\n  警告: 检测到多个按钮按下: {pressed_buttons}")
            print("  请只按一个按钮，重新校准...")
            time.sleep(1.0)
            return calibrate_button(joystick, button_name, button_function)
        
        time.sleep(0.05)
    
    # 超时，检查是否有输入（用户想跳过）
    print("\n  未检测到按钮按下，跳过此按钮")
    return None

def get_gamepad_type():
    """获取手柄类型"""
    print("\n" + "="*60)
    print("请选择手柄类型：")
    print("  1. 罗技 (Logitech) - 左摇杆(0,1), 右摇杆(2,3)")
    print("  2. 北通 (Betop) - 左摇杆(0,1), 右摇杆(3,4)")
    print("  3. 其他/自定义 - 手动指定轴映射")
    choice = input("请选择 (1/2/3，默认1): ").strip()
    
    if choice == '2':
        return 'betop', {
            0: '左摇杆X轴 (角速度)',
            1: '左摇杆Y轴 (抗扰动模式扰动力)',
            3: '右摇杆X轴 (左右速度)',
            4: '右摇杆Y轴 (前后速度)'
        }
    elif choice == '3':
        # 自定义映射
        print("\n请输入轴映射（格式：左摇杆X,左摇杆Y,右摇杆X,右摇杆Y）")
        print("例如：0,1,2,3 表示左摇杆X=轴0, 左摇杆Y=轴1, 右摇杆X=轴2, 右摇杆Y=轴3")
        mapping_input = input("轴映射 (默认0,1,2,3): ").strip()
        if not mapping_input:
            mapping_input = "0,1,2,3"
        
        try:
            axes = [int(x.strip()) for x in mapping_input.split(',')]
            if len(axes) != 4:
                raise ValueError("需要4个轴ID")
            return 'custom', {
                axes[0]: '左摇杆X轴 (角速度)',
                axes[1]: '左摇杆Y轴 (抗扰动模式扰动力)',
                axes[2]: '右摇杆X轴 (左右速度)',
                axes[3]: '右摇杆Y轴 (前后速度)'
            }
        except Exception as e:
            print(f"错误: 无效的轴映射: {e}")
            print("使用默认映射（罗技）")
            return 'logitech', {
                0: '左摇杆X轴 (角速度)',
                1: '左摇杆Y轴 (抗扰动模式扰动力)',
                2: '右摇杆X轴 (左右速度)',
                3: '右摇杆Y轴 (前后速度)'
            }
    else:
        # 默认罗技
        return 'logitech', {
            0: '左摇杆X轴 (角速度)',
            1: '左摇杆Y轴 (抗扰动模式扰动力)',
            2: '右摇杆X轴 (左右速度)',
            3: '右摇杆Y轴 (前后速度)'
        }

def calibrate_all_axes(joystick):
    """校准所有轴"""
    calibration = {
        'joystick_name': joystick.get_name(),
        'calibration_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'axes': {}
    }
    
    # 获取手柄类型和轴映射
    gamepad_type, axes_to_calibrate = get_gamepad_type()
    calibration['gamepad_type'] = gamepad_type  # 保存手柄类型
    
    print("\n" + "="*60)
    print("开始校准游戏手柄摇杆")
    print("="*60)
    print(f"手柄类型: {gamepad_type}")
    print("轴映射:")
    for axis_id, axis_name in sorted(axes_to_calibrate.items()):
        print(f"  {axis_name} -> 轴 {axis_id}")
    print("\n注意：")
    print("  - 请确保手柄连接稳定")
    print("  - 校准过程中请保持所有摇杆在中位（不施加任何力）")
    print("  - 可以随时按 Ctrl+C 退出")
    
    # 检查所有轴是否存在
    missing_axes = []
    for axis_id in axes_to_calibrate.keys():
        if axis_id >= joystick.get_numaxes():
            missing_axes.append(axis_id)
    
    if missing_axes:
        print(f"\n警告: 以下轴不存在，将被跳过: {missing_axes}")
        axes_to_calibrate = {k: v for k, v in axes_to_calibrate.items() if k not in missing_axes}
    
    if not axes_to_calibrate:
        print("\n错误: 没有可校准的轴")
        return None
    
    try:
        # 一次性校准所有轴
        axes_cal = calibrate_all_axes_at_once(joystick, axes_to_calibrate)
        
        # 保存校准数据
        for axis_id, axis_cal in axes_cal.items():
            calibration['axes'][str(axis_id)] = axis_cal
        
        return calibration
    except KeyboardInterrupt:
        print("\n\n校准已取消")
        return None
    except Exception as e:
        print(f"\n错误: 校准过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def calibrate_all_buttons(joystick):
    """校准所有按钮"""
    calibration = {
        'buttons': {}
    }
    
    # 定义需要校准的按钮（根据 deploy_mujoco_plot_20dof_gamepad.py 中的映射）
    buttons_to_calibrate = {
        'LB': '切换模式 (行走/奔跑/抗扰动测试)',
        'RB': '备用按钮（当前未使用）',
        'X': '切换显示足底力',
        'Y': '切换是否跟踪pelvis',
        'B': '重置机器人状态',
        'A': '切换显示足端接触状态'
    }
    
    print("\n" + "="*60)
    print("开始校准游戏手柄按钮")
    print("="*60)
    print("\n注意：")
    print("  - 请确保手柄连接稳定")
    print("  - 按照提示按下对应的按钮")
    print("  - 可以随时按 Ctrl+C 退出")
    print("  - 如果某个按钮不想校准，直接按回车跳过")
    
    for button_name, button_function in buttons_to_calibrate.items():
        try:
            button_cal = calibrate_button(joystick, button_name, button_function)
            if button_cal is not None:
                calibration['buttons'][button_name] = button_cal
        except KeyboardInterrupt:
            print("\n\n校准已取消")
            return None
        except Exception as e:
            print(f"\n错误: 校准按钮 {button_name} 时出错: {e}")
            continue
    
    return calibration

def test_calibration(joystick, calibration):
    """测试校准结果"""
    print("\n" + "="*60)
    print("测试校准结果")
    print("="*60)
    print("移动摇杆或按下按钮，查看实时值（已应用校准）")
    print("按任意键开始测试，按 Ctrl+C 退出测试")
    input()
    
    try:
        while True:
            pygame.event.pump()
            
            # 显示摇杆值
            print("\r", end="")
            if 'axes' in calibration:
                for axis_id_str, axis_cal in calibration['axes'].items():
                    axis_id = int(axis_id_str)  # 转换为整数
                    raw_value = joystick.get_axis(axis_id)
                    # 应用校准：减去中位值（不再反转方向）
                    calibrated_value = raw_value - axis_cal['center']
                    
                    print(f"{axis_cal['name']}: {calibrated_value:7.4f}  ", end="")
            
            # 显示按钮状态
            if 'buttons' in calibration:
                print(" | 按钮: ", end="")
                for button_name, button_cal in calibration['buttons'].items():
                    button_id = button_cal['button_id']
                    is_pressed = joystick.get_button(button_id)
                    status = "●" if is_pressed else "○"
                    print(f"{button_name}({button_id}){status} ", end="")
            
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\n测试结束")

def save_calibration(calibration):
    """保存校准结果到文件（根据手柄类型）"""
    try:
        gamepad_type = calibration.get('gamepad_type', 'logitech')
        calibration_file = get_calibration_file_path(gamepad_type)
        with open(calibration_file, 'w', encoding='utf-8') as f:
            json.dump(calibration, f, indent=2, ensure_ascii=False)
        print(f"\n校准结果已保存到: {calibration_file}")
        return True
    except Exception as e:
        print(f"\n错误: 保存校准文件失败: {e}")
        return False

def load_calibration(gamepad_type=None):
    """加载校准文件（如果指定了gamepad_type则加载对应文件，否则尝试加载所有可能的文件）"""
    if gamepad_type:
        calibration_file = get_calibration_file_path(gamepad_type)
        if os.path.exists(calibration_file):
            try:
                with open(calibration_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"警告: 加载校准文件失败: {e}")
                return None
        return None
    
    # 如果没有指定类型，尝试加载所有可能的校准文件（按优先级）
    for gp_type in ['betop', 'logitech', 'custom']:
        calibration_file = get_calibration_file_path(gp_type)
        if os.path.exists(calibration_file):
            try:
                with open(calibration_file, 'r', encoding='utf-8') as f:
                    cal = json.load(f)
                    # 验证文件中的gamepad_type是否匹配
                    if cal.get('gamepad_type') == gp_type:
                        return cal
            except Exception as e:
                continue
    return None

def main():
    """主函数"""
    print("="*60)
    print("游戏手柄校准工具")
    print("="*60)
    
    # 检查是否已有校准文件
    existing_cal = load_calibration()
    if existing_cal:
        print(f"\n检测到已有校准文件:")
        print(f"  手柄名称: {existing_cal.get('joystick_name', 'Unknown')}")
        print(f"  校准日期: {existing_cal.get('calibration_date', 'Unknown')}")
        choice = input("\n是否重新校准？(y/n，默认n): ").strip().lower()
        if choice != 'y':
            print("使用现有校准文件")
            return
    
    # 初始化手柄
    joystick = init_gamepad()
    if joystick is None:
        return
    
    # 询问要校准的内容
    print("\n请选择要校准的内容：")
    print("  1. 只校准摇杆")
    print("  2. 只校准按钮")
    print("  3. 校准摇杆和按钮（推荐）")
    choice = input("请选择 (1/2/3，默认3): ").strip()
    if choice not in ['1', '2', '3']:
        choice = '3'
    
    calibration = {
        'joystick_name': joystick.get_name(),
        'calibration_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'axes': {},
        'buttons': {}
    }
    
    # 校准摇杆
    if choice in ['1', '3']:
        axes_cal = calibrate_all_axes(joystick)
        if axes_cal is None:
            return
        calibration['axes'] = axes_cal.get('axes', {})
        # 确保手柄类型被保存（从axes_cal中获取）
        if 'gamepad_type' in axes_cal:
            calibration['gamepad_type'] = axes_cal['gamepad_type']
    
    # 校准按钮
    if choice in ['2', '3']:
        buttons_cal = calibrate_all_buttons(joystick)
        if buttons_cal is None:
            return
        calibration['buttons'] = buttons_cal.get('buttons', {})
    
    # 测试校准结果
    test_choice = input("\n是否测试校准结果？(y/n，默认y): ").strip().lower()
    if test_choice != 'n':
        test_calibration(joystick, calibration)
    
    # 保存校准结果
    if save_calibration(calibration):
        gamepad_type = calibration.get('gamepad_type', 'logitech')
        calibration_file = get_calibration_file_path(gamepad_type)
        print("\n校准完成！")
        print(f"校准文件位置: {calibration_file}")
        print("现在可以在 deploy_mujoco_plot_20dof_gamepad.py 中使用校准后的手柄了")
    else:
        print("\n警告: 校准完成但保存失败，请检查文件权限")
    
    # 清理
    joystick.quit()
    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已退出")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

