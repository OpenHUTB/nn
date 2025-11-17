import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading

class LaneDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("道路方向识别系统")
        self.root.geometry("1000x700")
        
        # 创建主框架
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # 创建控件
        self.create_widgets(main_frame)
        
        # 当前显示的图像
        self.original_photo = None
        self.result_photo = None
        
    def create_widgets(self, parent):
        # 标题
        title_label = ttk.Label(parent, text="道路方向识别系统", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 选择文件按钮
        select_btn = ttk.Button(parent, text="选择道路图片", command=self.select_image)
        select_btn.grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        
        # 文件路径显示
        self.file_path_var = tk.StringVar()
        file_path_label = ttk.Label(parent, textvariable=self.file_path_var, wraplength=400)
        file_path_label.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 检测按钮
        detect_btn = ttk.Button(parent, text="检测道路方向", command=self.detect_direction)
        detect_btn.grid(row=2, column=0, sticky=tk.W, pady=(0, 10))
        
        # 结果显示
        self.result_var = tk.StringVar()
        self.result_var.set("请选择图片并点击检测")
        result_label = ttk.Label(parent, textvariable=self.result_var, font=("Arial", 12, "bold"))
        result_label.grid(row=2, column=1, sticky=tk.W, pady=(0, 10))
        
        # 创建左右两个框架用于显示图片
        images_frame = ttk.Frame(parent)
        images_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)
        images_frame.rowconfigure(0, weight=1)
        
        # 原图显示区域
        original_frame = ttk.LabelFrame(images_frame, text="原图", padding="5")
        original_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        original_frame.columnconfigure(0, weight=1)
        original_frame.rowconfigure(0, weight=1)
        
        self.original_label = ttk.Label(original_frame, text="原图将显示在这里", relief="solid", background="white")
        self.original_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 结果图显示区域
        result_frame = ttk.LabelFrame(images_frame, text="道路高亮图", padding="5")
        result_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        self.result_label = ttk.Label(result_frame, text="检测结果将显示在这里", relief="solid", background="white")
        self.result_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 进度条
        self.progress = ttk.Progressbar(parent, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief="sunken")
        status_bar.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def select_image(self):
        """选择图片文件"""
        file_types = [
            ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("所有文件", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="选择道路图片",
            filetypes=file_types
        )
        
        if file_path:
            self.file_path_var.set(file_path)
            self.display_original_image(file_path)
            self.result_var.set("图片已加载，点击'检测道路方向'进行分析")
            self.status_var.set(f"已加载: {os.path.basename(file_path)}")
            # 清空结果图
            self.result_label.configure(image='')
            self.result_label.text = "检测结果将显示在这里"
    
    def display_original_image(self, file_path):
        """在UI中显示原图"""
        try:
            # 使用PIL打开图片并调整大小以适应显示区域
            image = Image.open(file_path)
            
            # 调整图片大小，保持宽高比
            image.thumbnail((450, 400), Image.Resampling.LANCZOS)
            
            # 转换为Tkinter可用的格式
            self.original_photo = ImageTk.PhotoImage(image)
            
            # 更新标签
            self.original_label.configure(image=self.original_photo, text="")
            self.original_label.image = self.original_photo  # 保持引用
            
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片: {str(e)}")
    
    def display_result_image(self, image):
        """在UI中显示结果图"""
        try:
            # 将OpenCV图像转换为PIL图像
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # 调整图片大小，保持宽高比
            pil_image.thumbnail((450, 400), Image.Resampling.LANCZOS)
            
            # 转换为Tkinter可用的格式
            self.result_photo = ImageTk.PhotoImage(pil_image)
            
            # 更新标签
            self.result_label.configure(image=self.result_photo, text="")
            self.result_label.image = self.result_photo  # 保持引用
            
        except Exception as e:
            messagebox.showerror("错误", f"无法显示结果图片: {str(e)}")
    
    def detect_direction(self):
        """检测道路方向"""
        file_path = self.file_path_var.get()
        
        if not file_path or not os.path.exists(file_path):
            messagebox.showwarning("警告", "请先选择有效的图片文件")
            return
        
        # 在后台线程中执行检测，避免UI冻结
        self.progress.start()
        self.status_var.set("正在分析道路方向...")
        
        thread = threading.Thread(target=self._detect_direction_thread, args=(file_path,))
        thread.daemon = True
        thread.start()
    
    def _detect_direction_thread(self, file_path):
        """在后台线程中执行方向检测"""
        try:
            direction, result_image = self._detect_road_direction(file_path)
            
            # 在主线程中更新UI
            self.root.after(0, self._update_result, direction, result_image)
            
        except Exception as e:
            self.root.after(0, self._show_error, str(e))
    
    def _update_result(self, direction, result_image):
        """更新检测结果"""
        self.progress.stop()
        self.result_var.set(f"检测结果: {direction}")
        self.display_result_image(result_image)
        self.status_var.set("分析完成")
    
    def _show_error(self, error_msg):
        """显示错误信息"""
        self.progress.stop()
        messagebox.showerror("错误", f"检测过程中发生错误: {error_msg}")
        self.status_var.set("检测失败")
    
    def _detect_road_direction(self, image_path):
        """检测道路方向的核心算法"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return "错误: 无法读取图像", np.zeros((300, 400, 3), dtype=np.uint8)
        
        # 创建结果图像的副本
        result_image = image.copy()
        height, width = image.shape[:2]
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 边缘检测
        edges = cv2.Canny(blur, 50, 150)
        
        # 定义ROI（感兴趣区域）- 图像下半部分
        roi_vertices = np.array([[
            (width * 0.1, height * 0.95),
            (width * 0.4, height * 0.6),
            (width * 0.6, height * 0.6),
            (width * 0.9, height * 0.95)
        ]], dtype=np.int32)
        
        # 创建ROI掩码
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(
            masked_edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=30, 
            minLineLength=20, 
            maxLineGap=50
        )
        
        if lines is None:
            # 绘制ROI区域
            cv2.polylines(result_image, [roi_vertices], True, (0, 255, 255), 2)
            cv2.putText(result_image, "No lines detected", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return "未知 - 未检测到车道线", result_image
        
        # 分类左右车道线
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 计算斜率
            if x2 - x1 == 0:
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            # 过滤接近水平的线
            if abs(slope) < 0.3:
                continue
                
            # 根据斜率分类
            if slope < 0:  # 左车道线
                left_lines.append((x1, y1, x2, y2, slope))
            else:  # 右车道线
                right_lines.append((x1, y1, x2, y2, slope))
        
        # 判断方向
        if not left_lines and not right_lines:
            # 绘制ROI区域
            cv2.polylines(result_image, [roi_vertices], True, (0, 255, 255), 2)
            cv2.putText(result_image, "No valid lane lines", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return "未知 - 未检测到有效的车道线", result_image
        elif not left_lines:
            direction = "右转 - 只检测到右侧车道线"
            color = (0, 165, 255)  # 橙色
        elif not right_lines:
            direction = "左转 - 只检测到左侧车道线"
            color = (0, 165, 255)  # 橙色
        else:
            # 计算左右车道线在图像顶部的平均位置
            left_x_top = np.mean([line[1] for line in left_lines])  # 使用y1作为顶部参考
            right_x_top = np.mean([line[1] for line in right_lines])
            
            # 计算车道中心在顶部的偏移
            lane_center = (left_x_top + right_x_top) / 2
            image_center = width / 2
            
            # 判断方向
            deviation = lane_center - image_center
            deviation_ratio = abs(deviation) / (width / 2)
            
            if deviation_ratio < 0.1:  # 阈值可根据需要调整
                direction = "直行"
                color = (0, 255, 0)  # 绿色
            elif deviation > 0:
                direction = "右转"
                color = (0, 165, 255)  # 橙色
            else:
                direction = "左转"
                color = (0, 165, 255)  # 橙色
        
        # 在结果图像上绘制车道线和区域
        self._draw_lanes(result_image, left_lines, right_lines, roi_vertices, direction, color)
        
        return direction, result_image
    
    def _draw_lanes(self, image, left_lines, right_lines, roi_vertices, direction, color):
        """在图像上绘制车道线和区域"""
        height, width = image.shape[:2]
        
        # 绘制ROI区域
        cv2.polylines(image, [roi_vertices], True, (0, 255, 255), 2)
        
        # 绘制检测到的所有线段
        for line in left_lines + right_lines:
            x1, y1, x2, y2, slope = line
            line_color = (0, 0, 255) if slope < 0 else (255, 0, 0)  # 左红右蓝
            cv2.line(image, (x1, y1), (x2, y2), line_color, 2)
        
        # 如果检测到左右车道线，绘制车道区域
        if left_lines and right_lines:
            # 拟合左右车道线
            left_points = np.array([[line[0], line[1]] for line in left_lines] + 
                                  [[line[2], line[3]] for line in left_lines])
            right_points = np.array([[line[0], line[1]] for line in right_lines] + 
                                   [[line[2], line[3]] for line in right_lines])
            
            # 使用最小二乘法拟合直线
            if len(left_points) > 1:
                left_fit = np.polyfit(left_points[:, 1], left_points[:, 0], 1)
                left_y_top = int(height * 0.6)
                left_y_bottom = height
                left_x_top = int(left_fit[0] * left_y_top + left_fit[1])
                left_x_bottom = int(left_fit[0] * left_y_bottom + left_fit[1])
                
                cv2.line(image, (left_x_bottom, left_y_bottom), (left_x_top, left_y_top), (0, 0, 255), 3)
            
            if len(right_points) > 1:
                right_fit = np.polyfit(right_points[:, 1], right_points[:, 0], 1)
                right_y_top = int(height * 0.6)
                right_y_bottom = height
                right_x_top = int(right_fit[0] * right_y_top + right_fit[1])
                right_x_bottom = int(right_fit[0] * right_y_bottom + right_fit[1])
                
                cv2.line(image, (right_x_bottom, right_y_bottom), (right_x_top, right_y_top), (255, 0, 0), 3)
            
            # 绘制车道区域
            if len(left_points) > 1 and len(right_points) > 1:
                # 创建车道区域的多边形
                lane_polygon = np.array([
                    [left_x_bottom, left_y_bottom],
                    [left_x_top, left_y_top],
                    [right_x_top, right_y_top],
                    [right_x_bottom, right_y_bottom]
                ], np.int32)
                
                # 绘制半透明的车道区域
                overlay = image.copy()
                cv2.fillPoly(overlay, [lane_polygon], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # 添加方向文本
        cv2.putText(image, f"Direction: {direction}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # 添加检测到的线条数量
        cv2.putText(image, f"Left lines: {len(left_lines)}, Right lines: {len(right_lines)}", 
                   (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def main():
    """主函数"""
    root = tk.Tk()
    app = LaneDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()