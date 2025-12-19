import cv2

class Camera:
    """摄像头工具类"""
    
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        
    def initialize(self):
        """初始化摄像头"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            # 尝试其他摄像头索引
            for i in range(1, 5):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    self.camera_index = i
                    break
            else:
                raise Exception("无法找到可用的摄像头")
        
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return True
    
    def read_frame(self):
        """读取一帧"""
        if self.cap is None:
            raise Exception("摄像头未初始化")
        
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("无法读取摄像头帧")
        
        # 水平翻转帧（镜像效果）
        frame = cv2.flip(frame, 1)
        return frame
    
    def release(self):
        """释放摄像头资源"""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    """主函数：显示摄像头画面"""
    camera = Camera()
    
    try:
        camera.initialize()
        print("摄像头初始化成功，按 'q' 键退出")
        
        while True:
            # 读取帧
            frame = camera.read_frame()
            
            # 显示帧
            cv2.imshow('Camera', frame)
            
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"错误: {e}")
    finally:
        camera.release()


if __name__ == "__main__":
    main()
