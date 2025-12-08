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
        self.root.geometry("800x600")
        
        # 创建主框架
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 创建控件
        self.create_widgets(main_frame)
        
        # 当前显示的图像
        self.current_image = None
        self.photo = None
        
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
        
        # 图像显示区域
        self.image_label = ttk.Label(parent, text="图片将显示在这里", relief="solid", background="white")
        self.image_label.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
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
            self.display_image(file_path)
            self.result_var.set("图片已加载，点击'检测道路方向'进行分析")
            self.status_var.set(f"已加载: {os.path.basename(file_path)}")
    
    def display_image(self, file_path):
        """在UI中显示图片"""
        try:
            # 使用PIL打开图片并调整大小以适应显示区域
            image = Image.open(file_path)
            
            # 获取显示区域的大小
            width, height = 700, 400  # 固定显示区域大小
            
            # 调整图片大小，保持宽高比
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            
            # 转换为Tkinter可用的格式
            self.photo = ImageTk.PhotoImage(image)
            
            # 更新标签
            self.image_label.configure(image=self.photo)
            self.image_label.image = self.photo  # 保持引用
            
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片: {str(e)}")
    
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
            result = self._detect_road_direction(file_path)
            
            # 在主线程中更新UI
            self.root.after(0, self._update_result, result)
            
        except Exception as e:
            self.root.after(0, self._show_error, str(e))
    
    def _update_result(self, result):
        """更新检测结果"""
        self.progress.stop()
        self.result_var.set(f"检测结果: {result}")
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
            return "错误: 无法读取图像"
        
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
            return "未知 - 未检测到车道线"
        
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
            return "未知 - 未检测到有效的车道线"
        elif not left_lines:
            return "右转 - 只检测到右侧车道线"
        elif not right_lines:
            return "左转 - 只检测到左侧车道线"
        
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
            return "直行"
        elif deviation > 0:
            return "右转"
        else:
            return "左转"

def main():
    """主函数"""
    root = tk.Tk()
    app = LaneDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()