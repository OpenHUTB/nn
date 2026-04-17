import cv2
import numpy as np
import threading
import tkinter as tk
from PIL import Image, ImageTk
import time
import queue

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
        self.frame_queue = queue.Queue(maxsize=2)
        self.capture_thread = None
        self.stop_capture = threading.Event()

        self.label = tk.Label(self, text="视频加载中...", width=width//10, height=height//20)
        self.label.pack(fill=tk.BOTH, expand=True)

    def start(self):
        """开始视频流"""
        if self.running:
            return
        self.running = True
        self.stop_capture.clear()
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        self.update_frame()

    def stop(self):
        """停止视频流"""
        self.running = False
        self.stop_capture.set()
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)

    def _capture_frames(self):
        """在单独线程中捕获帧"""
        while not self.stop_capture.is_set():
            try:
                if not self.client:
                    time.sleep(0.5)
                    continue

                responses = self.client.simGetImages([self.airsim.ImageRequest(0, self.airsim.ImageType.Scene, False, False)])
                response = responses[0]

                if not hasattr(response, 'height') or not hasattr(response, 'width'):
                    time.sleep(0.1)
                    continue

                if response.height <= 0 or response.width <= 0:
                    time.sleep(0.1)
                    continue

                if not hasattr(response, 'image_data_uint8') or response.image_data_uint8 is None:
                    time.sleep(0.1)
                    continue

                image_data = bytes(response.image_data_uint8)
                img1d = np.frombuffer(image_data, dtype=np.uint8)

                if img1d.size == 0:
                    time.sleep(0.1)
                    continue

                if len(img1d) != response.height * response.width * 3:
                    time.sleep(0.1)
                    continue

                img_rgb = img1d.reshape(response.height, response.width, 3)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                img_resized = cv2.resize(img_bgr, (self.width, self.height))

                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(img_resized, block=False)

            except (BufferError, AssertionError):
                time.sleep(0.5)
                continue
            except Exception as e:
                time.sleep(0.5)
                continue

            time.sleep(0.1)

    def update_frame(self):
        """更新视频帧"""
        if not self.running:
            return

        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img_rgb)
                imgtk = ImageTk.PhotoImage(image=img)

                self.current_frame = imgtk
                self.label.configure(image=imgtk)
                self.label.image = imgtk
        except queue.Empty:
            pass
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
