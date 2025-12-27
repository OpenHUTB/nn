# camera_detector.py
# 功能：封装基于摄像头的实时目标检测流程
# 特性：
#   - 支持 FPS（帧率）动态统计与输出
#   - 支持按 's' 键保存当前检测帧
#   - 支持按 'q' 键优雅退出
#   - 自动创建可缩放 OpenCV 窗口
#   - 异常安全：确保摄像头和窗口资源被正确释放
#   - 兼容模型推理失败场景（不会因单帧错误崩溃）

import cv2          # OpenCV：用于摄像头读取、图像显示与保存
import time         # 用于时间戳生成和 FPS 计算
import os           # 用于文件路径操作（保存帧时）
import traceback    # 用于打印完整的错误调用栈，便于调试


# ========================
# 🔧 自定义异常类（提升错误语义清晰度）
# ========================

class CameraOpenError(Exception):
    """
    摄像头设备无法打开时抛出的专用异常。
    用于区分“设备不存在”、“权限不足”、“已被占用”等场景。
    """
    pass


# ========================
# 🎥 主检测器类
# ========================

class CameraDetector:
    """
    摄像头实时目标检测器。

    职责：
      - 管理 VideoCapture 生命周期
      - 调用外部 detection_engine 执行每帧推理
      - 显示带标注的视频流
      - 响应用户键盘输入（退出/保存）
      - 实时计算并输出 FPS（帧率）
      - 安全保存检测结果帧

    设计原则：
      - **松耦合**：不依赖具体模型，只依赖实现了 detect(frame) 接口的对象
      - **健壮性**：即使某帧推理失败，也不中断整个检测循环
      - **资源安全**：无论是否发生异常，都确保释放摄像头和关闭窗口
    """

    def __init__(self, detection_engine, output_interval=1.0):
        """
        初始化摄像头检测器。

        参数:
            detection_engine (object):
                必须实现 detect(frame) 方法，返回 (annotated_frame, results)。
                通常为 DetectionEngine 实例。

            output_interval (float, optional):
                FPS 信息的输出间隔（单位：秒）。默认每 1 秒打印一次。
                设置过小会导致控制台刷屏；过大则反馈延迟。
        """
        self.engine = detection_engine           # 外部传入的检测引擎（如 YOLO 封装）
        self.output_interval = output_interval   # FPS 打印的时间间隔（秒）
        self.last_output_time = 0                # 上次打印 FPS 的 Unix 时间戳
        self.frame_count = 0                     # 自上次打印以来成功处理的帧数
        self.window_name = "YOLO_Live_Detection" # OpenCV 窗口标题（使用英文避免编码问题）

    def start_detection(self, camera_index=0):
        """
        启动摄像头并进入实时检测主循环。

        参数:
            camera_index (int, optional):
                摄像头设备索引。通常：
                  - 0：内置摄像头（笔记本）
                  - 1、2...：外接 USB 摄像头
                若指定设备不可用，将抛出 CameraOpenError。

        流程概览：
          1. 尝试打开摄像头设备
          2. 创建可调整大小的 OpenCV 显示窗口
          3. 进入无限循环：
               a. 读取一帧
               b. 调用检测引擎推理
               c. 显示结果
               d. 监听键盘事件（退出/保存）
               e. 更新 FPS 统计
          4. 捕获所有异常并记录
          5. finally 块确保资源释放

        注意：
          - 使用 cv2.waitKey(1) 实现非阻塞键盘监听
          - 即使 detect() 返回无效图像，也尝试显示（避免黑屏）
        """
        cap = None  # 初始化为 None，便于 finally 块安全检查
        try:
            # ----------------------------
            # 🔌 步骤 1：打开摄像头设备
            # ----------------------------
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                # 如果无法打开，抛出自定义异常，携带上下文信息
                raise CameraOpenError(
                    f"Cannot open camera device {camera_index}. "
                    "Possible reasons: "
                    "- Device does not exist; "
                    "- Already in use by another application; "
                    "- Insufficient permissions (e.g., on Linux)."
                )

            # ----------------------------
            # 🖼️ 步骤 2：创建显示窗口
            # ----------------------------
            # 使用 WINDOW_NORMAL 允许用户手动缩放窗口（对高分辨率摄像头友好）
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            print("Starting live detection. Press 'q' to quit, 's' to save frame.")

            # ----------------------------
            # 🔁 步骤 3：主检测循环
            # ----------------------------
            while True:
                # 从摄像头读取一帧（BGR 格式，HWC 布局）
                ret, frame = cap.read()
                if not ret:
                    # 读取失败常见于：摄像头被拔出、驱动崩溃、USB 断开
                    print("⚠️ Warning: Failed to read frame from camera. "
                          "Camera may have been disconnected.")
                    break  # 退出循环，进入清理阶段

                # 获取当前时间戳，用于 FPS 计算
                current_time = time.time()

                # ----------------------------
                # 🧠 步骤 4：执行目标检测
                # ----------------------------
                # 调用外部引擎进行推理。即使内部出错，detect() 也会返回原图（见 detection_engine.py）
                annotated_frame, _ = self.engine.detect(frame)

                # 安全检查：防止 OpenCV 显示空图像导致崩溃
                if annotated_frame.size == 0:
                    print("⚠️ Warning: Received empty annotated frame. Skipping display.")
                    continue  # 跳过当前帧，继续下一帧

                # ----------------------------
                # 👁️ 步骤 5：显示结果
                # ----------------------------
                cv2.imshow(self.window_name, annotated_frame)

                # ----------------------------
                # ⌨️ 步骤 6：监听键盘输入（非阻塞）
                # ----------------------------
                # waitKey(1) 表示等待 1 毫秒，若无按键则返回 -1
                # & 0xFF 是为了兼容某些系统返回高位字节的情况
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):      # 'q' 键：退出程序
                    print("User pressed 'q'. Exiting live detection...")
                    break
                elif key == ord('s'):    # 's' 键：保存当前帧
                    self.save_frame(annotated_frame)

                # ----------------------------
                # 📊 步骤 7：更新并输出 FPS
                # ----------------------------
                self._print_fps_if_needed(current_time)
                self.frame_count += 1

        # ----------------------------
        # 🚨 异常处理区
        # ----------------------------
        except KeyboardInterrupt:
            # 用户按下 Ctrl+C 中断程序
            print("\nUser interrupted live detection via Ctrl+C.")
        except CameraOpenError as e:
            # 摄像头打开失败（由我们主动抛出）
            print(f"❌ Camera error: {e}")
        except Exception as e:
            # 捕获其他未预期的异常（如 OpenCV 内部错误）
            print(f"💥 Unexpected error during camera detection loop: {e}")
            traceback.print_exc()  # 打印完整错误栈，便于开发者定位问题

        # ----------------------------
        # ♻️ 资源清理区（无论是否出错都会执行）
        # ----------------------------
        finally:
            # 安全释放摄像头资源（避免“设备忙”错误）
            if cap is not None and cap.isOpened():
                cap.release()
                print("✅ Camera resource released.")

            # 关闭所有 OpenCV 窗口（防止残留窗口）
            cv2.destroyAllWindows()
            print("AllWindows closed. Live detection terminated.")

    def _print_fps_if_needed(self, current_time):
        """
        根据预设的时间间隔，计算并打印当前 FPS（Frames Per Second）。

        参数:
            current_time (float): 当前 Unix 时间戳（由 time.time() 获取）

        逻辑说明：
          - 首次运行时 last_output_time 为 0，跳过 FPS 计算（避免除零）
          - 达到 output_interval 后，计算平均 FPS 并重置计数器
          - FPS = 处理帧数 / 时间间隔

        示例输出：
          FPS: 28.45
        """
        elapsed = current_time - self.last_output_time
        if elapsed >= self.output_interval:
            if self.last_output_time > 0:  # 避免除零（首次不计算）
                fps = self.frame_count / elapsed
                print(f"FPS: {fps:.2f}")  # 保留两位小数
            # 重置计时器和帧计数器，开始下一轮统计
            self.last_output_time = current_time
            self.frame_count = 0

    def save_frame(self, frame):
        """
        将当前检测帧保存为 JPEG 图像文件。

        参数:
            frame (np.ndarray): BGR 格式的图像数组（来自 OpenCV）

        文件命名规则：
          saved_frame_<Unix时间戳>.jpg
          例如：saved_frame_1734567890.jpg

        优势：
          - 时间戳确保文件名全局唯一
          - 避免覆盖用户已有文件

        异常处理：
          - 捕获写入失败（如磁盘满、路径无权限）
          - 不中断主流程，仅打印警告
        """
        timestamp = int(time.time())  # 获取当前秒级时间戳
        filename = f"saved_frame_{timestamp}.jpg"

        try:
            # 使用 OpenCV 将图像写入磁盘（JPEG 格式，质量默认）
            success = cv2.imwrite(filename, frame)
            if success:
                # 获取绝对路径便于用户定位文件
                abs_path = os.path.abspath(filename)
                print(f"✅ Frame saved successfully: {abs_path}")
            else:
                print(f"❌ OpenCV failed to write image to: {filename}")
        except Exception as e:
            # 捕获文件系统相关异常（如权限、磁盘空间）
            print(f"❌ Error saving frame: {e}")
