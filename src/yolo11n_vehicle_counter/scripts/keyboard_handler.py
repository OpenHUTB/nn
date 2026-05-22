import cv2 as cv
import os

# 编辑模式状态（全局变量）
edit_mode = False

def toggle_edit_mode():
    """切换编辑模式"""
    global edit_mode
    edit_mode = not edit_mode
    status = "开启" if edit_mode else "关闭"
    print(f"📝 编辑模式已{status}")
    return edit_mode

def is_edit_mode():
    """检查是否处于编辑模式"""
    return edit_mode

def handle_keyboard_events(key, frame, frame_count, cap, out, window_name):
    """处理键盘事件

    Args:
        key: 按下的键
        frame: 当前帧
        frame_count: 当前帧计数
        cap: 视频捕获对象
        out: 视频输出对象
        window_name: 窗口名称

    Returns:
        tuple: (是否继续运行, 是否需要更新显示)
    """
    need_update = False

    if key == ord('d'):  # 按'd'键切换编辑模式
        toggle_edit_mode()
        need_update = True
    elif key == ord('p'):  # 按'p'键暂停
        # 进入暂停状态
        paused = True
        while paused:
            pause_key = cv.waitKey(0) & 0xff
            if pause_key == ord('p'):  # 再次按'p'键继续
                paused = False
            elif pause_key == ord(' '):  # 按空格键逐帧播放
                # 读取下一帧
                ret, frame = cap.read()
                if not ret:
                    paused = False
                    break
                frame_count += 1
                # 显示当前帧
                cv.imshow(window_name, frame)
                # 写入帧到输出视频
                out.write(frame)
            elif pause_key == ord('q'):  # 按'q'键退出程序
                # 释放资源
                cap.release()
                out.release()
                cv.destroyAllWindows()
                return False, False
    elif key == ord('s'):  # 按's'键保存当前帧截图
        # 创建截图目录（如果不存在）
        screenshot_dir = "screenshots"
        if not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)
        # 保存截图
        screenshot_path = os.path.join(screenshot_dir, f"frame_{frame_count}.jpg")
        cv.imwrite(screenshot_path, frame)
        print(f"✅ 截图已保存: {screenshot_path}")
    elif key == ord('q'):  # 按'q'键退出程序
        return False, False

    return True, need_update
