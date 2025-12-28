# 用户界面控制模块
import os
import sys
import pygame
import numpy as np
import math

print("🎮 PyCharm UI演示 - 增强版（带醒目按键提示）")
print("=" * 60)


def run_enhanced_ui():
    """增强版UI演示，按键操作更醒目"""
    # 初始化
    pygame.init()

    # 创建窗口
    screen_width, screen_height = 1200, 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("AI无人机面部识别与追踪系统 - 增强演示")

    # 颜色
    COLORS = {
        'bg': (20, 20, 35),
        'panel': (35, 35, 55),
        'text': (255, 255, 255),
        'button': (80, 140, 200),
        'button_hover': (110, 170, 230),
        'success': (0, 220, 120),
        'warning': (255, 220, 70),
        'error': (255, 90, 90),
        'key_hint': (255, 180, 50),
        'highlight': (255, 255, 0),
        'grid': (40, 40, 60)
    }

    # 字体
    font_title = pygame.font.Font(None, 52)
    font_large = pygame.font.Font(None, 42)
    font_medium = pygame.font.Font(None, 32)
    font_small = pygame.font.Font(None, 26)
    font_key = pygame.font.Font(None, 36)

    # 按钮
    buttons = [
        {"rect": pygame.Rect(800, 80, 350, 60), "text": "🛸 连接无人机 (C键)", "id": "connect", "key": pygame.K_c},
        {"rect": pygame.Rect(800, 160, 350, 60), "text": "🔍 开始/停止人物检测 (D键)", "id": "detect",
         "key": pygame.K_d},
        {"rect": pygame.Rect(800, 240, 350, 60), "text": "😊 开始/停止人脸识别 (R键)", "id": "recognize",
         "key": pygame.K_r},
        {"rect": pygame.Rect(800, 320, 350, 60), "text": "🎯 开始/停止目标跟踪 (T键)", "id": "track", "key": pygame.K_t},
        {"rect": pygame.Rect(800, 400, 350, 60), "text": "📸 添加新人脸到数据库 (A键)", "id": "add_face",
         "key": pygame.K_a},
        {"rect": pygame.Rect(800, 480, 350, 60), "text": "🗑️ 清除选择 (Del键)", "id": "clear", "key": pygame.K_DELETE},
        {"rect": pygame.Rect(800, 650, 350, 60), "text": "❌ 退出系统 (ESC键)", "id": "exit", "key": pygame.K_ESCAPE}
    ]

    # 状态
    drone_connected = False
    detection_active = True
    recognition_active = True
    tracking_active = False
    selected_person = None

    # 模拟数据
    detected_persons = 3
    recognized_faces = 2
    drone_position = {"x": 15.5, "y": 28.3, "z": 12.0}

    # 动画变量
    animation_time = 0
    last_key_pressed = None
    key_press_time = 0

    # 主循环
    clock = pygame.time.Clock()
    running = True

    print("\n🖥️ 增强版UI窗口已启动!")
    print("   • 所有按钮上都标明了对应的按键")
    print("   • 你可以点击按钮或直接按键盘按键")
    print("   • 按键操作会有视觉反馈")
    print("   • 按ESC或点击'退出系统'关闭窗口\n")

    while running:
        # 更新动画时间
        animation_time += 1
        current_time = pygame.time.get_ticks()

        # 获取鼠标位置
        mouse_pos = pygame.mouse.get_pos()

        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                last_key_pressed = event.key
                key_press_time = current_time

                # ESC键退出
                if event.key == pygame.K_ESCAPE:
                    print("   👋 退出系统")
                    running = False

                # 处理其他按键
                for button in buttons:
                    if event.key == button["key"]:
                        button_id = button["id"]
                        print(f"⌨️  按键操作: {button['text']}")

                        if button_id == "connect":
                            drone_connected = not drone_connected
                            status = "已连接" if drone_connected else "已断开"
                            print(f"   🛸 无人机{status}")

                        elif button_id == "detect":
                            detection_active = not detection_active
                            status = "开始" if detection_active else "停止"
                            print(f"   👤 人物检测{status}")

                        elif button_id == "recognize":
                            recognition_active = not recognition_active
                            status = "开始" if recognition_active else "停止"
                            print(f"   😊 人脸识别{status}")

                        elif button_id == "track":
                            tracking_active = not tracking_active
                            status = "开始" if tracking_active else "停止"
                            print(f"   🎯 目标跟踪{status}")

                        elif button_id == "add_face":
                            print("   📸 模拟: 添加新人脸到数据库")

                        elif button_id == "clear":
                            selected_person = None
                            print("   🗑️ 已清除选择")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键
                    for button in buttons:
                        if button["rect"].collidepoint(mouse_pos):
                            last_key_pressed = button["key"]
                            key_press_time = current_time

                            button_id = button["id"]
                            print(f"🖱️ 点击按钮: {button['text']}")

                            if button_id == "connect":
                                drone_connected = not drone_connected
                                status = "已连接" if drone_connected else "已断开"
                                print(f"   🛸 无人机{status}")

                            elif button_id == "detect":
                                detection_active = not detection_active
                                status = "开始" if detection_active else "停止"
                                print(f"   👤 人物检测{status}")

                            elif button_id == "recognize":
                                recognition_active = not recognition_active
                                status = "开始" if recognition_active else "停止"
                                print(f"   😊 人脸识别{status}")

                            elif button_id == "track":
                                tracking_active = not tracking_active
                                status = "开始" if tracking_active else "停止"
                                print(f"   🎯 目标跟踪{status}")

                            elif button_id == "add_face":
                                print("   📸 模拟: 添加新人脸到数据库")

                            elif button_id == "clear":
                                selected_person = None
                                print("   🗑️ 已清除选择")

                            elif button_id == "exit":
                                print("   👋 退出系统")
                                running = False

        # 绘制背景
        screen.fill(COLORS['bg'])

        # 绘制标题
        title = font_title.render("AI无人机面部识别与追踪系统", True, COLORS['text'])
        screen.blit(title, (screen_width // 2 - title.get_width() // 2, 20))

        # 绘制视频模拟区域
        video_rect = pygame.Rect(50, 80, 700, 500)
        pygame.draw.rect(screen, (10, 10, 20), video_rect)
        pygame.draw.rect(screen, COLORS['button'], video_rect, 4)

        # 绘制视频标题
        video_title = font_medium.render("无人机摄像头视图", True, COLORS['text'])
        screen.blit(video_title, (video_rect.x + 10, video_rect.y - 35))

        # 在视频区域绘制模拟内容
        # 动态网格背景
        grid_color = COLORS['grid']
        for i in range(0, 700, 25):
            offset = int(10 * math.sin(animation_time * 0.01 + i * 0.01))
            pygame.draw.line(screen, grid_color,
                             (video_rect.x + i, video_rect.y + offset),
                             (video_rect.x + i, video_rect.y + video_rect.height + offset), 1)

        for i in range(0, 500, 25):
            offset = int(10 * math.cos(animation_time * 0.01 + i * 0.01))
            pygame.draw.line(screen, grid_color,
                             (video_rect.x + offset, video_rect.y + i),
                             (video_rect.x + video_rect.width + offset, video_rect.y + i), 1)

        # 模拟检测到的人物框
        if detection_active:
            # 移动的人物框1
            time_ms = animation_time * 0.05
            box1_x = 150 + int(120 * math.sin(time_ms))
            box1_y = 150 + int(100 * math.cos(time_ms * 0.8))

            # 绘制人物框（带动画效果）
            pulse = abs(math.sin(time_ms * 2)) * 2 + 1
            pygame.draw.rect(screen, COLORS['success'],
                             (video_rect.x + box1_x, video_rect.y + box1_y, 140, 220), int(pulse))

            # 人物标签
            person_label = font_small.render("Person 1 (85%)", True, COLORS['success'])
            screen.blit(person_label, (video_rect.x + box1_x, video_rect.y + box1_y - 25))

            # 人物框2
            box2_x = 400
            box2_y = 180
            pygame.draw.rect(screen, COLORS['warning'],
                             (video_rect.x + box2_x, video_rect.y + box2_y, 120, 200), 3)

            person_label2 = font_small.render("Person 2 (72%)", True, COLORS['warning'])
            screen.blit(person_label2, (video_rect.x + box2_x, video_rect.y + box2_y - 25))

            # 人物框3
            box3_x = 500 + int(60 * math.sin(time_ms * 1.5))
            box3_y = 300 + int(40 * math.cos(time_ms * 1.2))
            pygame.draw.rect(screen, (180, 100, 255),
                             (video_rect.x + box3_x, video_rect.y + box3_y, 100, 180), 3)

            person_label3 = font_small.render("Person 3 (68%)", True, (180, 100, 255))
            screen.blit(person_label3, (video_rect.x + box3_x, video_rect.y + box3_y - 25))

            # 添加人脸标记
            if recognition_active:
                # 人脸1
                face_pulse = abs(math.sin(time_ms * 3)) * 10 + 5
                pygame.draw.circle(screen, COLORS['error'],
                                   (video_rect.x + box1_x + 70, video_rect.y + box1_y + 50),
                                   35 + int(face_pulse), 2)

                face_label = font_small.render("张三", True, COLORS['error'])
                screen.blit(face_label, (video_rect.x + box1_x + 60, video_rect.y + box1_y + 90))

                # 人脸2
                pygame.draw.circle(screen, (255, 150, 50),
                                   (video_rect.x + box2_x + 60, video_rect.y + box2_y + 40),
                                   30, 2)

                face_label2 = font_small.render("李四", True, (255, 150, 50))
                screen.blit(face_label2, (video_rect.x + box2_x + 50, video_rect.y + box2_y + 80))

        # 绘制状态面板
        status_panel = pygame.Rect(50, 600, 700, 180)
        pygame.draw.rect(screen, COLORS['panel'], status_panel, border_radius=10)
        pygame.draw.rect(screen, COLORS['button'], status_panel, 3, border_radius=10)

        # 绘制状态信息
        status_y = 620
        status_items = [
            f"🛸 无人机状态: {'🟢 已连接' if drone_connected else '🔴 未连接'}",
            f"🔍 人物检测: {'🟢 进行中' if detection_active else '🔴 已停止'}  ({detected_persons}人)",
            f"😊 人脸识别: {'🟢 进行中' if recognition_active else '🔴 已停止'}  ({recognized_faces}人)",
            f"🎯 目标跟踪: {'🟢 进行中' if tracking_active else '🔴 已停止'}",
            f"📍 无人机位置: X={drone_position['x']:.1f}m, Y={drone_position['y']:.1f}m, 高度={drone_position['z']:.1f}m",
            f"🎯 选定目标: {'✅ 已选择' if selected_person else '❌ 未选择'}"
        ]

        for i, item in enumerate(status_items):
            text = font_small.render(item, True, COLORS['text'])
            screen.blit(text, (70, status_y + i * 30))

        # 绘制控制面板
        control_panel = pygame.Rect(800, 20, 370, 760)
        pygame.draw.rect(screen, COLORS['panel'], control_panel, border_radius=15)
        pygame.draw.rect(screen, COLORS['button'], control_panel, 4, border_radius=15)

        control_title = font_large.render("控制面板", True, COLORS['text'])
        screen.blit(control_title, (control_panel.x + control_panel.width // 2 - control_title.get_width() // 2, 40))

        # 绘制按钮
        for button in buttons:
            # 检查鼠标悬停
            is_hover = button["rect"].collidepoint(mouse_pos)

            # 按键按下效果
            is_key_pressed = (last_key_pressed == button["key"] and
                              current_time - key_press_time < 300)  # 300ms高亮

            # 按钮颜色
            if button["id"] == "exit":
                base_color = (220, 80, 80)
                hover_color = (250, 100, 100)
            else:
                base_color = COLORS['button']
                hover_color = COLORS['button_hover']

            # 高亮最近按下的按钮
            if is_key_pressed:
                color = COLORS['highlight']
                border_color = COLORS['highlight']
            elif is_hover:
                color = hover_color
                border_color = COLORS['text']
            else:
                color = base_color
                border_color = COLORS['text']

            # 绘制按钮
            pygame.draw.rect(screen, color, button["rect"], border_radius=10)
            pygame.draw.rect(screen, border_color, button["rect"], 3, border_radius=10)

            # 绘制按钮文字
            text_lines = button["text"].split(" (")
            if len(text_lines) > 1:
                main_text = text_lines[0]
                key_text = "(" + text_lines[1]

                # 主文本
                main_render = font_medium.render(main_text, True, COLORS['text'])
                main_rect = main_render.get_rect(center=(button["rect"].centerx, button["rect"].centery - 10))
                screen.blit(main_render, main_rect)

                # 按键文本
                key_render = font_key.render(key_text, True, COLORS['key_hint'])
                key_rect = key_render.get_rect(center=(button["rect"].centerx, button["rect"].centery + 15))
                screen.blit(key_render, key_rect)
            else:
                text = font_medium.render(button["text"], True, COLORS['text'])
                text_rect = text.get_rect(center=button["rect"].center)
                screen.blit(text, text_rect)

        # 绘制按键提示面板
        hint_panel = pygame.Rect(800, 560, 370, 120)
        pygame.draw.rect(screen, (45, 45, 70), hint_panel, border_radius=10)
        pygame.draw.rect(screen, COLORS['key_hint'], hint_panel, 2, border_radius=10)

        hint_title = font_medium.render("💡 快速操作提示", True, COLORS['key_hint'])
        screen.blit(hint_title, (hint_panel.x + 20, hint_panel.y + 15))

        hints = [
            "• 点击按钮 或 直接按对应按键",
            "• ESC: 退出系统",
            "• 空格键: 暂停/继续动画"
        ]

        for i, hint in enumerate(hints):
            hint_render = font_small.render(hint, True, COLORS['text'])
            screen.blit(hint_render, (hint_panel.x + 20, hint_panel.y + 45 + i * 25))

        # 绘制底部状态栏
        if last_key_pressed and current_time - key_press_time < 1000:
            key_name = pygame.key.name(last_key_pressed).upper()
            status_text = f"最近操作: 按下了 [{key_name}] 键"
            status_render = font_small.render(status_text, True, COLORS['key_hint'])
            screen.blit(status_render, (screen_width // 2 - status_render.get_width() // 2, screen_height - 30))

        # 绘制版本信息
        version_text = font_small.render("AI无人机系统 v1.0 - PyCharm演示版", True, (150, 150, 180))
        screen.blit(version_text, (screen_width - version_text.get_width() - 20, screen_height - 30))

        # 更新显示
        pygame.display.flip()
        clock.tick(60)  # 60 FPS

    # 清理
    pygame.quit()
    print("\n✅ 增强版UI演示结束")


def main():
    """主函数"""
    print("=" * 60)
    print("🎮 AI无人机系统 - 增强版UI演示")
    print("=" * 60)

    try:
        import pygame
        print(f"✅ Pygame版本: {pygame.version.ver}")

        run_enhanced_ui()

    except ImportError:
        print("❌ Pygame未安装")
        print("💡 请运行: pip install pygame")
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()