import cv2
import numpy as np
import threading
import tkinter as tk
from PIL import Image, ImageTk

class VideoStreamFrame(tk.Frame):
    def __init__(self, parent, client, airsim, width=320, height=240):
        super().__init__(parent)
        self.client = client
        self.airsim = airsim
        self.width = width
        self.height = height
        self.running = False
        self.current_frame = None
        self.image_id = None
        
        self.label = tk.Label(self, text="视频加载中...", width=width//10, height=height//20)
        self.label.pack(fill=tk.BOTH, expand=True)
    
    def start(self):
        """开始视频流"""
        if self.running:
            return
        self.running = True
        self.update_frame()
    
    def stop(self):
        """停止视频流"""
        self.running = False
    
    def get_frame(self):
        """从AirSim获取视频帧"""
        if not self.client:
            return None
        
        try:
            responses = self.client.simGetImages([self.airsim.ImageRequest(0, self.airsim.ImageType.Scene, False, False)])
            response = responses[0]
            
            if response.height <= 0 or response.width <= 0:
                print(f"无效的图像尺寸: {response.height}x{response.width}")
                return None
            
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            if img1d.size == 0:
                print("图像数据为空")
                return None
            
            img_rgb = img1d.reshape(response.height, response.width, 3)
            
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img_resized = cv2.resize(img_bgr, (self.width, self.height))
            
            return img_resized
        except Exception as e:
            print(f"获取视频帧失败: {e}")
            return None
    
    def update_frame(self):
        """更新视频帧"""
        if not self.running:
            return
        
        try:
            frame = self.get_frame()
            if frame is not None:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.current_frame = imgtk
                self.label.configure(image=imgtk)
                self.label.image = imgtk
        except Exception as e:
            print(f"更新视频帧失败: {e}")
        
        self.after(33, self.update_frame)

def create_video_window(client, airsim_module):
    """创建一个独立的视频流窗口"""
    window = tk.Toplevel()
    window.title("无人机视频流")
    window.geometry("640x480")
    
    video_frame = VideoStreamFrame(window, client, airsim_module, width=640, height=480)
    video_frame.pack(fill=tk.BOTH, expand=True)
    video_frame.start()
    
    def on_close():
        video_frame.stop()
        window.destroy()
    
    window.protocol("WM_DELETE_WINDOW", on_close)
    
    # 返回窗口，供外部控制
    return window

if __name__ == "__main__":
    import airsim
    print("测试视频流显示...")
    
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    create_video_window(client, airsim)
