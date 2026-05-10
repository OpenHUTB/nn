"""
可拖拽区域处理器
================

提供可拖拽的矩形区域功能，支持鼠标拖拽调整区域位置和大小，
并可将配置保存到JSON文件。

使用方法:
    from draggable_handler import DraggableRect

    # 初始化
    draggable = DraggableRect([100, 100, 500, 400])

    # 加载保存的配置
    draggable.load_config("config/counting_region.json")

    # 设置鼠标回调
    cv2.setMouseCallback("Window", draggable.mouse_callback)

    # 绘制区域
    draggable.draw(frame, edit_mode=True)

    # 保存配置
    draggable.save_config("config/counting_region.json")
"""

import cv2 as cv
import json
import os


class DraggableRect:
    """可拖拽的矩形区域

    支持8个控制点：4个角点 + 4个边中点
    支持配置保存和加载

    Attributes:
        region: 矩形区域 [left, top, right, bottom]
        handle_radius: 控制点半径
        dragging_idx: 当前拖拽的控制点索引
        hover_idx: 鼠标悬停的控制点索引
        min_size: 区域最小尺寸
    """

    # 控制点类型
    HANDLE_TOP_LEFT = 0
    HANDLE_TOP_RIGHT = 1
    HANDLE_BOTTOM_RIGHT = 2
    HANDLE_BOTTOM_LEFT = 3
    HANDLE_TOP = 4
    HANDLE_RIGHT = 5
    HANDLE_BOTTOM = 6
    HANDLE_LEFT = 7

    def __init__(self, region, handle_radius=20, min_size=50):
        """初始化可拖拽矩形

        Args:
            region: 初始区域 [left, top, right, bottom]
            handle_radius: 控制点半径
            min_size: 区域最小宽高
        """
        self.region = list(region)
        self.handle_radius = handle_radius
        self.min_size = min_size
        self.dragging_idx = None
        self.hover_idx = None
        self._frame_shape = None

    def get_handles(self):
        """返回8个控制点坐标

        Returns:
            list: [(x, y), ...] 8个控制点坐标
        """
        l, t, r, b = self.region
        return [
            (l, t), (r, t), (r, b), (l, b),  # 四角: 0-3
            ((l + r) // 2, t),  # 上边中点: 4
            (r, (t + b) // 2),  # 右边中点: 5
            ((l + r) // 2, b),  # 下边中点: 6
            (l, (t + b) // 2)   # 左边中点: 7
        ]

    def find_handle(self, x, y):
        """查找鼠标位置对应的控制点索引

        Args:
            x: 鼠标X坐标
            y: 鼠标Y坐标

        Returns:
            int: 控制点索引，未找到返回None
        """
        handles = self.get_handles()
        for i, (hx, hy) in enumerate(handles):
            distance = ((x - hx) ** 2 + (y - hy) ** 2) ** 0.5
            if distance < self.handle_radius + 5:  # 增加一点容差
                return i
        return None

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数

        Args:
            event: 鼠标事件类型
            x: 鼠标X坐标
            y: 鼠标Y坐标
            flags: 附加标志
            param: 用户参数
        """
        if event == cv.EVENT_LBUTTONDOWN:
            self.dragging_idx = self.find_handle(x, y)
        elif event == cv.EVENT_MOUSEMOVE:
            if self.dragging_idx is not None:
                self._update_region(x, y)
            else:
                self.hover_idx = self.find_handle(x, y)
        elif event == cv.EVENT_LBUTTONUP:
            self.dragging_idx = None

    def _update_region(self, x, y):
        """根据拖拽的控制点更新区域

        Args:
            x: 鼠标X坐标
            y: 鼠标Y坐标
        """
        l, t, r, b = self.region

        # 限制在帧范围内
        if self._frame_shape is not None:
            h, w = self._frame_shape[:2]
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))

        if self.dragging_idx == self.HANDLE_TOP_LEFT:
            l, t = x, y
        elif self.dragging_idx == self.HANDLE_TOP_RIGHT:
            r, t = x, y
        elif self.dragging_idx == self.HANDLE_BOTTOM_RIGHT:
            r, b = x, y
        elif self.dragging_idx == self.HANDLE_BOTTOM_LEFT:
            l, b = x, y
        elif self.dragging_idx == self.HANDLE_TOP:
            t = y
        elif self.dragging_idx == self.HANDLE_RIGHT:
            r = x
        elif self.dragging_idx == self.HANDLE_BOTTOM:
            b = y
        elif self.dragging_idx == self.HANDLE_LEFT:
            l = x

        # 确保区域有效（左上角在左上方）
        l, r = min(l, r - self.min_size), max(r, l + self.min_size)
        t, b = min(t, b - self.min_size), max(b, t + self.min_size)

        self.region = [l, t, r, b]

    def draw(self, frame, edit_mode=False):
        """绘制区域和控制点

        Args:
            frame: 输入帧
            edit_mode: 是否处于编辑模式
        """
        self._frame_shape = frame.shape
        l, t, r, b = [int(v) for v in self.region]

        # 绘制区域边框
        color = (0, 255, 0) if not edit_mode else (0, 200, 255)
        thickness = 2 if not edit_mode else 3
        cv.rectangle(frame, (l, t), (r, b), color, thickness)

        # 绘制区域标签
        label = "Counting Region"
        if edit_mode:
            label = "EDIT MODE - Drag to adjust"
        cv.putText(frame, label, (l, t - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if edit_mode:
            # 绘制控制点
            handles = self.get_handles()
            for i, (hx, hy) in enumerate(handles):
                # 拖拽中的控制点显示红色，悬停的显示黄色，其他显示绿色
                if i == self.dragging_idx:
                    color = (0, 0, 255)  # 红色 - 拖拽中
                    radius = self.handle_radius + 3
                elif i == self.hover_idx:
                    color = (0, 255, 255)  # 黄色 - 悬停
                    radius = self.handle_radius + 2
                else:
                    color = (0, 255, 0)  # 绿色 - 普通
                    radius = self.handle_radius

                cv.circle(frame, (int(hx), int(hy)), radius, color, -1)
                cv.circle(frame, (int(hx), int(hy)), radius, (255, 255, 255), 2)  # 白色边框

            # 绘制对角线（辅助定位）
            cv.line(frame, (l, t), (r, b), (100, 100, 100), 1, cv.LINE_AA)
            cv.line(frame, (r, t), (l, b), (100, 100, 100), 1, cv.LINE_AA)

            # 显示区域尺寸
            width = r - l
            height = b - t
            size_text = f"{width} x {height}"
            cv.putText(frame, size_text, ((l + r) // 2 - 40, (t + b) // 2),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def save_config(self, path):
        """保存配置到JSON文件

        Args:
            path: 配置文件路径
        """
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        config = {
            'region': self.region,
            'handle_radius': self.handle_radius,
            'min_size': self.min_size
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        print(f"✅ 区域配置已保存: {path}")
        return True

    def load_config(self, path):
        """从JSON文件加载配置

        Args:
            path: 配置文件路径

        Returns:
            bool: 是否成功加载
        """
        if not os.path.exists(path):
            return False

        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            if 'region' in config:
                self.region = config['region']
                print(f"✅ 已加载区域配置: {path}")
                print(f"   区域: {self.region}")
                return True
        except Exception as e:
            print(f"❌ 加载配置失败: {e}")

        return False

    def reset(self, region=None):
        """重置区域

        Args:
            region: 新区域，为None则使用初始值
        """
        if region is not None:
            self.region = list(region)
        self.dragging_idx = None
        self.hover_idx = None

    @property
    def left(self):
        return self.region[0]

    @property
    def top(self):
        return self.region[1]

    @property
    def right(self):
        return self.region[2]

    @property
    def bottom(self):
        return self.region[3]

    @property
    def width(self):
        return self.region[2] - self.region[0]

    @property
    def height(self):
        return self.region[3] - self.region[1]

    @property
    def center(self):
        return ((self.region[0] + self.region[2]) // 2,
                (self.region[1] + self.region[3]) // 2)
