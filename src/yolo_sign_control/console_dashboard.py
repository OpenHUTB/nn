"""
控制台实时状态面板 - 不依赖Pygame显示，直接在终端输出实时状态
"""

import os
import shutil
import time

# ==============================================================================
# -- ANSI 控制码 ----------------------------------------------------------------
# ==============================================================================

class Style:
    """ANSI 样式控制"""
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'
    CLEAR = '\033[2J'       # 清屏
    HOME = '\033[H'         # 光标回原点
    ERASE_LINE = '\033[2K'  # 清除整行
    CURSOR_UP = '\033[A'    # 光标上移

class Color:
    """ANSI 颜色"""
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

class BgColor:
    """ANSI 背景色"""
    RED = '\033[41m'
    GREEN = '\033[42m'
    YELLOW = '\033[43m'


def enable_ansi_support():
    """启用 Windows 终端 ANSI 支持（Windows 10+）"""
    if os.name == 'nt':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return True


def _truncate(text, max_len):
    """截断文本并添加省略号"""
    if len(text) > max_len:
        return text[:max_len - 3] + '...'
    return text


# ==============================================================================
# -- 颜色辅助函数 ----------------------------------------------------------------
# ==============================================================================

def color_speed(speed):
    """根据速度返回颜色"""
    if speed > 80:
        return Color.RED
    elif speed > 50:
        return Color.YELLOW
    return Color.GREEN


def color_distance(dist):
    """根据距离返回颜色"""
    if dist is None:
        return Style.DIM
    if dist < 8:
        return Color.RED + Style.BOLD
    elif dist < 15:
        return Color.RED
    elif dist < 25:
        return Color.YELLOW
    return Color.GREEN


def color_confidence(conf):
    """根据置信度返回颜色"""
    if conf > 0.8:
        return Color.GREEN
    elif conf > 0.6:
        return Color.YELLOW
    return Color.RED


# ==============================================================================
# -- 主渲染函数 ------------------------------------------------------------------
# ==============================================================================

def render_dashboard(elapsed, speed, max_speed, signs,
                     warning_text, is_emergency,
                     throttle, brake, steer,
                     traffic_light_state, nearest_obs,
                     fps=None, vehicle_count=None):
    """
    在控制台绘制实时状态面板

    参数:
        elapsed: 已运行时间（秒）
        speed: 当前速度 (km/h)
        max_speed: 最高速度 (km/h)
        signs: detect_traffic_signs() 返回的检测结果列表
        warning_text: 碰撞预警文字
        is_emergency: 是否紧急状态
        throttle: 油门值 (0-1)
        brake: 刹车值 (0-1)
        steer: 转向值 (-1 到 1)
        traffic_light_state: CARLA 交通灯状态
        nearest_obs: get_nearest_obstacle() 返回的最近障碍物
        vehicle_count: NPC 车辆数量（可选）
    """
    m = int(elapsed // 60)
    s = int(elapsed % 60)

    # 获取终端宽度
    try:
        term_width = shutil.get_terminal_size().columns
    except Exception:
        term_width = 90
    W = min(term_width, 110)

    sep = '─' * W

    lines = []
    lines.append(Style.CLEAR + Style.HOME)

    # ── 标题 ──
    title = " CARLA Traffic Sign Detection - Dashboard "
    padding = (W - len(title)) // 2
    lines.append(' ' * padding + Style.BOLD + Color.CYAN + title + Style.RESET)
    lines.append(Style.DIM + sep + Style.RESET)

    # ── 状态行 ──
    time_str = f"Time:  {m:02d}:{s:02d}"
    speed_colored = color_speed(speed)
    speed_str = f"{speed_colored}Speed: {speed:.1f} km/h{Style.RESET}"
    max_str = f"Max: {max_speed:.1f} km/h"
    fps_str = f"FPS: {fps:.0f}" if fps else ""

    lines.append(f"  {Style.BOLD}{time_str}{Style.RESET}    {speed_str}    {Style.DIM}{max_str}  {fps_str}{Style.RESET}")

    # ── 状态 / 警告 ──
    if warning_text:
        if is_emergency or "EMERGENCY" in warning_text or "WARNING" in warning_text:
            lines.append(f"  Status: {BgColor.RED}{Color.WHITE}{Style.BOLD} {warning_text} {Style.RESET}")
        elif "Caution" in warning_text:
            lines.append(f"  Status: {Color.YELLOW}{Style.BOLD}⚠ {warning_text}{Style.RESET}")
        else:
            lines.append(f"  Status: {Color.YELLOW}ℹ {warning_text}{Style.RESET}")
    else:
        lines.append(f"  Status: {Color.GREEN}● Running normally{Style.RESET}")

    # ── 碰撞预警（特大显示） ──
    if is_emergency or (warning_text and "EMERGENCY" in warning_text):
        emoji = "🔴"
        warn_line = f" {emoji} {warning_text} {emoji} "
        padding = (W - len(warn_line)) // 2
        lines.append('')
        lines.append(' ' * padding + BgColor.RED + Color.WHITE + Style.BOLD +
                     f" {warning_text} " + Style.RESET)
        lines.append('')

    lines.append(Style.DIM + sep + Style.RESET)

    # ── 检测结果表格 ──
    lines.append(f"  {Style.BOLD}Detected Objects{Style.RESET}" +
                 (f"  ({len(signs)} total)" if signs else ""))
    lines.append('')

    if signs:
        # 表头
        header = (f"  {Style.DIM}{'Type':<20} {'Conf':<8} {'Dist':<8} "
                  f"{'BBox':<22} Action{Style.RESET}")
        lines.append(header)
        lines.append(f"  {Style.DIM}{'─' * (W - 4)}{Style.RESET}")

        # 按距离排序（有距离的在前）
        sorted_signs = []
        for label, conf, bbox in signs:
            x1, y1, x2, y2 = bbox
            dist = _estimate_distance(label, (x1, y1, x2, y2), 600)
            sorted_signs.append((dist if dist else 999, label, conf, bbox))
        sorted_signs.sort(key=lambda x: x[0])

        for rank, (dist_val, label, conf, bbox) in enumerate(sorted_signs[:12]):
            x1, y1, x2, y2 = bbox
            dist_str = f"{dist_val:.1f}m" if dist_val < 999 else "N/A"
            dist_color = color_distance(dist_val if dist_val < 999 else None)
            conf_color = color_confidence(conf)
            conf_str = f"{conf:.0%}"

            # 动作为自动判定
            action = "─"
            if "stop" in label.lower():
                action = "🛑 STOP"
            elif "speed limit" in label.lower():
                action = "⚠ SPEED"
            elif "traffic light" in label.lower():
                action = "🚦 LIGHT"
            elif any(cls in label.lower() for cls in ('person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle')):
                if dist_val < 8:
                    action = "🔴 EMERGENCY"
                elif dist_val < 15:
                    action = "🟡 BRAKING"
                elif dist_val < 25:
                    action = "🟢 SLOWING"
                else:
                    action = "○ MONITOR"

            bbox_str = f"({x1},{y1})→({x2},{y2})"
            line = (f"  {_truncate(label, 20):<20} "
                    f"{conf_color}{conf_str:<8}{Style.RESET} "
                    f"{dist_color}{dist_str:<8}{Style.RESET} "
                    f"{Style.DIM}{bbox_str:<22}{Style.RESET} "
                    f"{action}")
            lines.append(line)
    else:
        lines.append(f"  {Style.DIM}No objects detected in current frame{Style.RESET}")

    lines.append(Style.DIM + sep + Style.RESET)

    # ── 车辆控制 ──
    # 进度条
    bar_width = W - 30
    throttle_bar = _make_bar(throttle, bar_width, Color.GREEN)
    brake_bar = _make_bar(brake, bar_width, Color.RED)

    lines.append(f"  {Style.BOLD}Controls:{Style.RESET}")
    lines.append(f"    Throttle: [{throttle_bar}] {throttle:.2f}")
    lines.append(f"    Brake:    [{brake_bar}] {brake:.2f}")
    steer_arrow = "←" if steer < -0.05 else ("→" if steer > 0.05 else "─")
    steer_color = Color.RED if abs(steer) > 0.3 else Color.YELLOW if abs(steer) > 0.1 else Color.GREEN
    lines.append(f"    Steer:    {steer_color}{steer_arrow} {steer:+.3f}{Style.RESET}")

    # 交通灯状态
    tl_str = "Unknown"
    tl_color = Style.DIM
    if traffic_light_state is not None:
        tl_name = str(traffic_light_state).split('.')[-1]
        if tl_name == 'Red':
            tl_str = '🔴 Red'
            tl_color = Color.RED + Style.BOLD
        elif tl_name == 'Green':
            tl_str = '🟢 Green'
            tl_color = Color.GREEN
        elif tl_name == 'Yellow':
            tl_str = '🟡 Yellow'
            tl_color = Color.YELLOW
        else:
            tl_str = f'○ {tl_name}'
    lines.append(f"    Traffic:  {tl_color}{tl_str}{Style.RESET}")

    # NPC 车辆计数
    if vehicle_count is not None:
        lines.append(f"    NPC:     {vehicle_count} vehicles in scene")

    lines.append(Style.DIM + sep + Style.RESET)

    # ── 最近障碍物详情（右下信息区） ──
    if nearest_obs:
        label, conf, bbox, dist = nearest_obs
        dist_color = color_distance(dist)
        lines.append(f"  {Style.BOLD}Nearest:{Style.RESET} "
                     f"{_truncate(label, 15):<15} "
                     f"at {dist_color}{dist:.1f}m{Style.RESET} "
                     f"(conf: {conf:.0%})")

    # ── 操作提示 ──
    lines.append(Style.DIM + sep + Style.RESET)
    lines.append(Style.DIM + "  ESC = Exit  |  Console updates every frame" + Style.RESET)

    # 一次性输出
    print('\n'.join(lines), end='', flush=True)


def _make_bar(value, width, color):
    """生成文本进度条"""
    filled = int(value * width)
    filled = max(0, min(filled, width))
    bar = color + '█' * filled + Style.DIM + '░' * (width - filled) + Style.RESET
    return bar


def _estimate_distance(label, bbox, image_height):
    """
    简化版距离估计（避免循环导入）
    与 main.py 中的 estimate_distance 逻辑相同
    """
    x1, y1, x2, y2 = bbox
    bbox_height = y2 - y1

    if bbox_height < 5:
        return None

    typical_heights = {
        'person': 1.7, 'car': 1.5, 'truck': 3.0, 'bus': 3.2,
        'bicycle': 1.0, 'motorcycle': 1.2, 'stop sign': 2.0, 'traffic light': 1.0,
    }

    height = 1.5
    for key, h in typical_heights.items():
        if key in label.lower():
            height = h
            break

    return (height * image_height) / bbox_height
