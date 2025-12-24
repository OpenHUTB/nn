# -*- coding: utf-8 -*-
"""
手势控制无人机系统 - 主入口模块
协调所有子模块，实现完整的手势控制无人机系统
作者: xiaoshiyuan888
优化版本：集成性能可视化模块
"""

import sys
import os
import time
import traceback
import cv2
import numpy as np

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入自定义模块
from config_manager import ConfigManager
from performance_analyzer import PerformanceAnalyzer
from speech_manager import EnhancedSpeechFeedbackManager
from gesture_recognizer import EnhancedGestureRecognizer
from drone_controller import SimpleDroneController
from ui_renderer import ChineseUIRenderer
from trajectory_recorder import GestureTrajectoryRecorder
from advanced_gesture_analyzer import AdvancedGestureAnalyzer
from gesture_stabilizer import GestureStabilizer  # 新增
from performance_visualizer import PerformanceVisualizer  # 新增


# 安全导入所需库
def safe_import_libs():
    """安全导入所有需要的库"""
    libs = {}
    status = {}

    try:
        import cv2
        import numpy as np
        libs['cv2'] = cv2
        libs['np'] = np
        status['OpenCV'] = True
        print("[OpenCV] ✓ 计算机视觉库就绪")
    except Exception as e:
        status['OpenCV'] = False
        print(f"[OpenCV] ✗ 导入失败: {e}")
        return None, status

    try:
        from PIL import Image, ImageDraw, ImageFont
        libs['PIL'] = {'Image': Image, 'ImageDraw': ImageDraw, 'ImageFont': ImageFont}
        status['PIL'] = True
        print("[PIL] ✓ 图像处理库就绪")
    except Exception as e:
        status['PIL'] = False
        print(f"[PIL] ✗ 导入失败: {e}")
        return None, status

    # 尝试导入AirSim
    airsim_module = None
    try:
        import airsim
        airsim_module = airsim
        libs['airsim'] = airsim_module
        status['AirSim'] = True
        print("[AirSim] ✓ 成功导入")
    except ImportError:
        print("\n" + "!" * 60)
        print("⚠ AirSim库未找到!")
        print("!" * 60)
        print("安装AirSim:")
        print("1. 首先安装: pip install msgpack-rpc-python")
        print("2. 然后安装: pip install airsim")
        print("\n或从源码安装:")
        print("  pip install git+https://github.com/microsoft/AirSim.git")
        print("!" * 60)

        print("\n无AirSim继续运行? (y/n)")
        choice = input().strip().lower()
        if choice != 'y':
            sys.exit(1)

    # 尝试导入语音库
    speech_module = None
    try:
        import pyttsx3
        speech_module = pyttsx3
        status['Speech'] = True
        print("[Speech] ✓ pyttsx3语音库就绪 (离线)")
    except ImportError:
        print("\n" + "!" * 60)
        print("⚠ pyttsx3语音库未找到!")
        print("!" * 60)
        print("安装语音库:")
        print("1. 安装离线语音库: pip install pyttsx3")
        print("2. 或者安装在线语音库: pip install gtts pygame")
        print("!" * 60)

        try:
            from gtts import gTTS
            speech_module = {'gTTS': gTTS, 'type': 'gtts'}
            status['Speech'] = True
            print("[Speech] ✓ gTTS语音库就绪 (需要网络连接)")

            try:
                import pygame
                pygame.mixer.init()
                speech_module['pygame'] = pygame
                print("[Speech] ✓ pygame音频播放库就绪")
            except ImportError:
                speech_module['play_method'] = 'system'
        except ImportError:
            print("[Speech] ✗ 所有语音库导入失败，语音功能将不可用")
            speech_module = None
            status['Speech'] = False

    # 尝试导入psutil
    try:
        import psutil
        libs['psutil'] = psutil
        status['psutil'] = True
        print("[psutil] ✓ 系统资源监控库就绪")
    except ImportError:
        print("[psutil] ⚠ 未找到，性能监控功能受限")
        libs['psutil'] = None

    libs['speech'] = speech_module
    return libs, status


def init_camera(config):
    """初始化摄像头"""
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get('camera', 'width'))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get('camera', 'height'))
            cap.set(cv2.CAP_PROP_FPS, config.get('camera', 'fps'))

            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

            print(f"✓ 摄像头已初始化")
            print(f"  分辨率: {actual_width}x{actual_height}")
            print(f"  帧率: {actual_fps}")
        else:
            print("❌ 摄像头不可用，使用模拟模式")
            cap = None
    except Exception as e:
        print(f"⚠ 摄像头初始化失败: {e}")
        cap = None

    return cap


def print_welcome_message(cap, speech_manager, libs):
    """打印欢迎信息"""
    print("\n" + "=" * 80)
    print("手势控制无人机系统 - 增强手势识别版 v2.0")
    print("=" * 80)
    print("系统状态:")
    print(f"  摄像头: {'已连接' if cap else '模拟模式'}")
    print(f"  手势识别: 改进的平滑算法 + 手势稳定性分析")
    print(f"  语音反馈: {'已启用' if speech_manager.enabled else '已禁用'}")
    print(f"  性能监控: 增强版 - 支持GPU监控和趋势预测")
    print(f"  轨迹记录: 支持录制/回放功能")
    print(f"  高级飞行模式: 方形轨迹、圆形盘旋、8字形飞行")
    print(f"  AirSim: {'可用' if libs['airsim'] else '模拟模式'}")
    print("=" * 80)


def print_instructions():
    """打印操作说明"""
    print("\n操作说明:")
    print("1. 按 [C] 连接无人机 (AirSim模拟器)")
    print("2. 按 [空格键] 起飞/降落")
    print("3. 性能统计功能:")
    print("   - 按 [P] 键显示详细性能报告")
    print("   - 按 [L] 键记录性能快照")
    print("   - 按 [K] 键导出性能日志")
    print("   - 按 [R] 键重置性能统计")
    print("   - 系统自动监控: FPS, CPU, 内存, GPU, 识别时间等")
    print("4. 性能模式选择:")
    print("   - 按 [O] 键循环切换性能模式: 最快(fast) → 平衡(balanced) → 最准(accurate)")
    print("5. 新手势控制:")
    print("   - 握拳手势: 抓取/释放物体 (模拟)")
    print("   - 旋转手势: 顺时针旋转 (模拟)")
    print("   - OK手势: 拍照/截图")
    print("   - 复杂手势: 返航、自动飞行等")
    print("6. 轨迹记录功能:")
    print("   [1]开始录制 [2]停止录制 [3]保存轨迹 [4]回放轨迹 [5]清除轨迹 [6]暂停/继续")
    print("7. 键盘控制:")
    print("   [W]向上 [S]向下 [A]向左 [D]向右 [F]向前 [B]向后 [X]停止 [H]悬停")
    print("   [G]返航 [Q]自动飞行模式 [E]圆形盘旋 [8]8字形飞行 [9]方形轨迹")
    print("   [T]增加高度 [Y]降低高度 [U]设定高度")
    print("8. 调试功能:")
    print("   [H]切换帮助显示 [R]重置手势识别 [T]切换显示模式 [D]调试信息")
    print("9. 语音控制:")
    print("   [V]切换语音反馈 [M]测试语音")
    print("10. 性能可视化:")
    print("   [F]切换性能图表显示 [G]切换性能仪表盘显示")
    print("11. 按 [ESC] 安全退出")
    print("=" * 80)
    print("程序启动成功!")
    print("-" * 80)


def main():
    """主函数"""
    print("=" * 60)
    print("手势控制无人机系统 - 增强手势识别版 v2.0")
    print("=" * 60)

    # 导入库
    libs, status = safe_import_libs()
    if not status.get('OpenCV', False) or not status.get('PIL', False):
        print("\n❌ 核心库缺失，无法启动。")
        input("按回车键退出...")
        sys.exit(1)

    print("-" * 60)
    print("✅ 环境检查通过，正在初始化...")
    print("-" * 60)

    # 初始化配置管理器
    config = ConfigManager()

    # 初始化语音管理器
    print("初始化语音反馈系统...")
    speech_manager = EnhancedSpeechFeedbackManager(libs['speech'], config)

    # 程序启动语音提示
    if speech_manager.enabled:
        speech_manager.speak('program_start', force=True, immediate=True)
        speech_manager.speak('system_ready', immediate=True)

    # 初始化其他组件
    print("初始化组件...")
    gesture_recognizer = EnhancedGestureRecognizer(speech_manager, config)
    drone_controller = SimpleDroneController(libs['airsim'], speech_manager, config)
    ui_renderer = ChineseUIRenderer(speech_manager, config)

    # 设置手势识别器的UI渲染器引用
    gesture_recognizer.set_ui_renderer(ui_renderer)

    # 初始化性能分析器
    print("初始化性能分析器...")
    performance_analyzer = PerformanceAnalyzer(speech_manager, libs.get('psutil'), config)

    # 手势轨迹记录器
    print("初始化手势轨迹记录器...")
    trajectory_recorder = GestureTrajectoryRecorder(speech_manager, config)

    # 高级手势分析器
    print("初始化高级手势分析器...")
    advanced_gesture_analyzer = AdvancedGestureAnalyzer(speech_manager, config)

    # 新增：手势稳定性分析器
    print("初始化手势稳定性分析器...")
    gesture_stabilizer = GestureStabilizer(window_size=15)

    # 新增：性能可视化器
    print("初始化性能可视化器...")
    performance_visualizer = PerformanceVisualizer()

    # 初始化摄像头
    cap = init_camera(config)
    if cap and speech_manager.enabled:
        speech_manager.speak('camera_ready', immediate=True)
    elif not cap and speech_manager.enabled:
        speech_manager.speak('camera_error', immediate=True)

    # 显示欢迎信息
    print_welcome_message(cap, speech_manager, libs)

    # 显示操作说明
    print_instructions()

    # 键盘手势映射
    key_to_gesture = {
        ord('w'): "Up", ord('W'): "Up",
        ord('s'): "Down", ord('S'): "Down",
        ord('a'): "Left", ord('A'): "Left",
        ord('d'): "Right", ord('D'): "Right",
        ord('f'): "Forward", ord('F'): "Forward",
        ord('b'): "Backward", ord('B'): "Backward",
        ord('x'): "Stop", ord('X'): "Stop",
        ord('h'): "Hover", ord('H'): "Hover",
        ord('g'): "ReturnHome", ord('G'): "ReturnHome",
        ord('q'): "AutoFlight", ord('Q'): "AutoFlight",
        ord('p'): "TakePhoto", ord('P'): "TakePhoto",
        ord('r'): "RotateCW", ord('R'): "RotateCW",
        ord('l'): "RotateCCW", ord('L'): "RotateCCW",
        ord('e'): "CircleFlight", ord('E'): "CircleFlight",
        ord('8'): "EightFlight", ord('*'): "EightFlight",
        ord('9'): "SquareFlight", ord('('): "SquareFlight",
        ord('t'): "IncreaseAltitude", ord('T'): "IncreaseAltitude",
        ord('y'): "DecreaseAltitude", ord('Y'): "DecreaseAltitude",
        ord('u'): "SetAltitude", ord('U'): "SetAltitude",
    }

    # 显示模式
    display_modes = ['normal', 'detailed', 'minimal']
    current_display_mode = 0

    # 新增：性能可视化模式
    visualization_modes = ['none', 'charts', 'gauges']
    current_visualization_mode = 0

    # 主循环
    print("\n进入主循环，按ESC退出...")

    try:
        while True:
            # 更新性能监控
            performance_analyzer.update_frame()
            performance_analyzer.update_system_resources()
            performance_analyzer.auto_report()

            # 读取摄像头帧
            if cap:
                ret, frame = cap.read()
                if not ret:
                    # 创建空白帧
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    gesture, confidence = "摄像头错误", 0.0
                else:
                    # 手势识别
                    gesture, confidence, frame = gesture_recognizer.recognize(frame)

                    # 使用高级手势分析器进行补充分析
                    advanced_result = advanced_gesture_analyzer.analyze(frame, gesture_recognizer.last_hand_data)
                    if advanced_result and advanced_result.get('confidence', 0) > confidence:
                        gesture = advanced_result.get('gesture', gesture)
                        confidence = advanced_result.get('confidence', confidence)

                    # 记录手势统计
                    if gesture and gesture != "Waiting" and gesture != "摄像头错误":
                        performance_analyzer.record_gesture(gesture, confidence)

                        # 新增：更新手势稳定性分析
                        if gesture_recognizer.last_hand_data:
                            position = gesture_recognizer.last_hand_data.get('position')
                            stability_info = gesture_stabilizer.add_gesture(gesture, confidence, position)
            else:
                # 模拟模式
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                gesture, confidence = gesture_recognizer.current_gesture, gesture_recognizer.current_confidence

            # 获取性能统计
            process_time, frame_rate = gesture_recognizer.get_performance_stats()

            # 更新手势识别时间
            performance_analyzer.update_gesture_recognition_time(process_time)

            # 根据显示模式调整显示选项
            if display_modes[current_display_mode] == 'normal':
                config.set('display', 'show_contours', value=True)
                config.set('display', 'show_bbox', value=True)
                config.set('display', 'show_fingertips', value=True)
                config.set('display', 'show_gesture_history', value=True)
                config.set('display', 'show_stability_indicator', value=True)
                config.set('display', 'show_trajectory', value=True)
                config.set('display', 'show_recording_status', value=True)
                config.set('display', 'show_performance_mode', value=True)
                config.set('display', 'show_performance_stats', value=True)
                config.set('display', 'show_system_resources', value=True)
                config.set('display', 'show_advanced_gestures', value=True)
                config.set('display', 'show_flight_mode', value=True)
                config.set('display', 'show_debug_info', value=False)
            elif display_modes[current_display_mode] == 'detailed':
                config.set('display', 'show_contours', value=True)
                config.set('display', 'show_bbox', value=True)
                config.set('display', 'show_fingertips', value=True)
                config.set('display', 'show_palm_center', value=True)
                config.set('display', 'show_hand_direction', value=True)
                config.set('display', 'show_gesture_history', value=True)
                config.set('display', 'show_stability_indicator', value=True)
                config.set('display', 'show_trajectory', value=True)
                config.set('display', 'show_recording_status', value=True)
                config.set('display', 'show_performance_mode', value=True)
                config.set('display', 'show_performance_stats', value=True)
                config.set('display', 'show_system_resources', value=True)
                config.set('display', 'show_advanced_gestures', value=True)
                config.set('display', 'show_flight_mode', value=True)
                config.set('display', 'show_debug_info', value=True)
            elif display_modes[current_display_mode] == 'minimal':
                config.set('display', 'show_contours', value=False)
                config.set('display', 'show_bbox', value=True)
                config.set('display', 'show_fingertips', value=False)
                config.set('display', 'show_gesture_history', value=False)
                config.set('display', 'show_stability_indicator', value=False)
                config.set('display', 'show_trajectory', value=True)
                config.set('display', 'show_recording_status', value=True)
                config.set('display', 'show_performance_mode', value=True)
                config.set('display', 'show_performance_stats', value=True)
                config.set('display', 'show_system_resources', value=True)
                config.set('display', 'show_advanced_gestures', value=False)
                config.set('display', 'show_flight_mode', value=True)
                config.set('display', 'show_debug_info', value=False)

            # 绘制轨迹（如果启用）
            if config.get('display', 'show_trajectory'):
                frame = trajectory_recorder.draw_trajectory(frame)

            # 绘制高级手势信息
            if config.get('display', 'show_advanced_gestures'):
                frame = advanced_gesture_analyzer.draw_gesture_info(frame)

            # 绘制UI
            frame = ui_renderer.draw_status_bar(
                frame, drone_controller, gesture, confidence,
                performance_analyzer.get_current_fps(), process_time,
                trajectory_recorder, gesture_recognizer, performance_analyzer
            )

            # 绘制飞行模式信息
            if config.get('display', 'show_flight_mode'):
                frame = ui_renderer.draw_flight_mode(frame, drone_controller)

            # 绘制性能可视化
            if visualization_modes[current_visualization_mode] == 'charts':
                frame = performance_visualizer.draw_comprehensive_charts(frame, performance_analyzer)
            elif visualization_modes[current_visualization_mode] == 'gauges':
                frame = performance_visualizer.draw_performance_gauges(frame, performance_analyzer)

            frame = ui_renderer.draw_help_bar(frame)

            # 显示连接提示
            if not drone_controller.connected:
                warning_msg = "⚠ 按C键连接无人机，或使用模拟模式"
                frame = ui_renderer.draw_warning(frame, warning_msg)

            # 显示性能警告
            if performance_analyzer.performance_status != "良好":
                warning_msg = f"⚠ 性能状态: {performance_analyzer.performance_status}"
                frame = ui_renderer.draw_warning(frame, warning_msg)

            # 显示图像
            cv2.imshow('Gesture Controlled Drone - Enhanced Gestures v2.0', frame)

            # ========== 键盘控制 ==========
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC键
                print("\n退出程序...")
                break

            elif key == ord('c') or key == ord('C'):
                if not drone_controller.connected:
                    drone_controller.connect()

            elif key == 32:  # 空格键
                if drone_controller.connected:
                    if drone_controller.flying:
                        drone_controller.land()
                    else:
                        drone_controller.takeoff()
                    time.sleep(0.5)

            elif key == ord('h') or key == ord('H'):
                # 切换帮助显示
                current = config.get('display', 'show_help')
                config.set('display', 'show_help', value=not current)
                print(f"帮助显示: {'开启' if not current else '关闭'}")

                # 语音提示
                if speech_manager.enabled:
                    speech_manager.speak('help_toggled', immediate=True)

            elif key == ord('r') or key == ord('R'):
                # 重置手势识别
                print("重置手势识别...")
                gesture_recognizer = EnhancedGestureRecognizer(speech_manager, config)
                # 重新设置UI渲染器引用
                gesture_recognizer.set_ui_renderer(ui_renderer)
                print("✓ 手势识别已重置")

                # 重置手势稳定性分析
                gesture_stabilizer.reset()
                print("✓ 手势稳定性分析已重置")

                # 语音提示
                if speech_manager.enabled:
                    speech_manager.speak_direct("手势识别已重置")

            elif key == ord('t') or key == ord('T'):
                # 切换显示模式
                current_display_mode = (current_display_mode + 1) % len(display_modes)
                mode_name = display_modes[current_display_mode]
                print(f"显示模式: {mode_name}")

                # 语音提示
                if speech_manager.enabled:
                    speech_manager.speak('display_mode_changed', immediate=True)

            elif key == ord('d') or key == ord('D'):
                # 切换调试信息
                current = config.get('display', 'show_debug_info')
                config.set('display', 'show_debug_info', value=not current)
                status = '开启' if not current else '关闭'
                print(f"调试信息: {status}")

                # 语音提示
                if speech_manager.enabled:
                    if not current:
                        speech_manager.speak('debug_mode_on', immediate=True)
                    else:
                        speech_manager.speak('debug_mode_off', immediate=True)

            elif key == ord('v') or key == ord('V'):
                # 切换语音反馈
                new_status = speech_manager.toggle_enabled()
                status = '启用' if new_status else '禁用'
                print(f"语音反馈: {status}")
                config.set('speech', 'enabled', value=new_status)

            elif key == ord('m') or key == ord('M'):
                # 测试语音
                if speech_manager.enabled:
                    print("测试语音...")
                    speech_manager.speak_direct("语音反馈测试，系统运行正常")
                else:
                    print("语音反馈已禁用，按V键启用")

            elif key == ord('p') or key == ord('P'):
                # 性能报告
                if key == ord('p'):  # 小写p - 简要报告
                    print("生成简要性能报告...")
                    performance_analyzer.print_report(detailed=False)
                else:  # 大写P - 详细报告
                    print("生成详细性能报告...")
                    performance_analyzer.print_report(detailed=True)

                # 语音提示
                if speech_manager.enabled:
                    speech_manager.speak('performance_report', immediate=True)

            elif key == ord('l') or key == ord('L'):
                # 性能快照
                snapshot_label = f"快照_{time.strftime('%H:%M:%S')}"
                snapshot = performance_analyzer.take_snapshot(snapshot_label)

                print(f"📸 性能快照已保存: {snapshot_label}")
                print(f"  当前FPS: {snapshot['fps']:.1f}")
                print(f"  平均FPS: {snapshot['avg_fps']:.1f}")
                print(f"  CPU使用率: {snapshot['cpu_usage']:.1f}%")
                print(f"  内存使用率: {snapshot['memory_usage']:.1f}%")

                # 显示手势稳定性信息
                stability_stats = gesture_stabilizer.get_stats()
                print(f"  手势稳定性评分: {stability_stats['stability_score']:.2f}")
                print(f"  手势稳定性等级: {stability_stats['stability_level']}")

                # 语音提示
                if speech_manager.enabled:
                    speech_manager.speak('performance_snapshot', immediate=True)

            elif key == ord('k') or key == ord('K'):
                # 导出性能日志
                if performance_analyzer.export_log():
                    print("✅ 性能日志导出成功")

                    # 语音提示
                    if speech_manager.enabled:
                        speech_manager.speak('performance_log_exported', immediate=True)
                else:
                    print("❌ 性能日志导出失败")

            elif key == ord('o') or key == ord('O'):
                # 切换性能模式
                if config.cycle_performance_mode():
                    new_mode = config.get_current_performance_mode()
                    gesture_recognizer.set_performance_mode(new_mode)

                    # 语音提示
                    if speech_manager.enabled:
                        if new_mode == 'fast':
                            speech_manager.speak('performance_mode_fast', immediate=True)
                        elif new_mode == 'balanced':
                            speech_manager.speak('performance_mode_balanced', immediate=True)
                        else:  # accurate
                            speech_manager.speak('performance_mode_accurate', immediate=True)

                    print(f"✓ 已切换到性能模式: {gesture_recognizer.performance_mode_name}")

            # 重置性能统计
            elif key == ord('R') and chr(key).isupper():  # 大写R
                print("重置性能统计...")
                performance_analyzer.reset_session()
                print("✓ 性能统计已重置")

                # 语音提示
                if speech_manager.enabled:
                    speech_manager.speak_direct("性能统计已重置")

            # 切换性能可视化模式
            elif key == ord('f') or key == ord('F'):
                current_visualization_mode = (current_visualization_mode + 1) % len(visualization_modes)
                mode_name = visualization_modes[current_visualization_mode]
                print(f"性能可视化模式: {mode_name}")

                # 语音提示
                if speech_manager.enabled:
                    speech_manager.speak_direct(f"性能可视化模式: {mode_name}")

            # 轨迹记录控制
            elif key == ord('1'):
                # 开始录制轨迹
                if trajectory_recorder.start_recording():
                    print("✅ 开始录制手势轨迹")
                    performance_analyzer.record_recording_session()
                else:
                    print("⚠ 已经在录制中")

            elif key == ord('2'):
                # 停止录制轨迹
                if trajectory_recorder.stop_recording():
                    print("✅ 停止录制手势轨迹")
                    performance_analyzer.record_recording_session(len(trajectory_recorder.trajectory_data))
                else:
                    print("⚠ 当前没有在录制")

            elif key == ord('3'):
                # 保存轨迹
                if trajectory_recorder.save_trajectory():
                    print("✅ 轨迹保存成功")
                else:
                    print("⚠ 没有轨迹数据可保存")

            elif key == ord('4'):
                # 回放轨迹
                if trajectory_recorder.start_playback():
                    print("✅ 开始回放手势轨迹")
                else:
                    print("⚠ 没有轨迹数据可回放")

            elif key == ord('5'):
                # 清除轨迹
                if trajectory_recorder.clear_trajectory():
                    print("✅ 轨迹数据已清除")
                else:
                    print("⚠ 清除轨迹失败")

            elif key == ord('6'):
                # 暂停/继续回放
                if trajectory_recorder.pause_playback():
                    print("✅ 切换回放暂停状态")
                else:
                    print("⚠ 当前没有在回放")

            elif key in key_to_gesture:
                # 键盘控制
                simulated_gesture = key_to_gesture[key]
                gesture_recognizer.set_simulated_gesture(simulated_gesture)
                gesture = simulated_gesture
                confidence = 0.9

                # 记录手势统计
                performance_analyzer.record_gesture(gesture, confidence)

                if drone_controller.connected and drone_controller.flying:
                    success = drone_controller.move_by_gesture(gesture, confidence)
                    performance_analyzer.record_drone_command(success)

            # 真实手势控制
            current_time = time.time()
            if (gesture and gesture != "Waiting" and
                    gesture != "摄像头错误" and gesture != "Error" and
                    drone_controller.connected and drone_controller.flying):
                success = drone_controller.move_by_gesture(gesture, confidence)
                performance_analyzer.record_drone_command(success)

            # 处理轨迹记录
            if cap and ret:
                # 如果正在录制，添加轨迹点
                if trajectory_recorder.is_recording:
                    # 获取手势识别的手部数据
                    if hasattr(gesture_recognizer, 'last_hand_data'):
                        hand_data = gesture_recognizer.last_hand_data
                        trajectory_recorder.add_trajectory_point(
                            hand_data, gesture, confidence, frame.shape
                        )

                # 如果正在回放，获取回放点
                if trajectory_recorder.is_playing and not trajectory_recorder.playback_paused:
                    playback_point = trajectory_recorder.get_next_playback_point()
                    if playback_point:
                        # 这里可以添加回放点的可视化或处理
                        pass

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序错误: {e}")
        traceback.print_exc()
    finally:
        # 清理资源
        print("\n清理资源...")
        if cap:
            cap.release()
        cv2.destroyAllWindows()

        # 生成最终性能报告
        print("\n" + "=" * 80)
        print("📊 最终性能总结")
        print("=" * 80)
        performance_analyzer.print_report(detailed=True)

        # 导出最终性能日志
        performance_analyzer.export_log()

        print("=" * 80)

        # 程序退出语音提示
        if speech_manager.enabled:
            speech_manager.speak('program_exit', force=True, immediate=True)
            time.sleep(1)

        drone_controller.emergency_stop()

        # 自动保存轨迹
        if trajectory_recorder and len(trajectory_recorder.trajectory_data) > 0:
            print("自动保存轨迹数据...")
            trajectory_recorder.save_trajectory()

        config.save_config()

        print("程序安全退出")
        print("=" * 80)
        print("\n感谢使用手势控制无人机系统!")
        input("按回车键退出...")


if __name__ == "__main__":
    main()