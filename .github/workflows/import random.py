class CameraManager:
    def __init__(self, camera_index=0):
        """
        初始化摄像头管理器
        :param camera_index: 摄像头设备索引，默认为0
        """
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError("无法打开摄像头，请检查设备连接。")
        
        # 设置摄像头参数（可选）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 设置帧宽度
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 设置帧高度
        self.cap.set(cv2.CAP_PROP_FPS, 30)           # 设置帧率

    def get_frame(self):
        """
        从摄像头获取一帧图像
        :return: (bool, numpy.ndarray) 成功标志和BGR格式的图像帧
        """
        ret, frame = self.cap.read()
        if ret:
            # 可选：进行镜头畸变校正（如有校准参数）
            # frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
            pass
        return ret, frame

    def release(self):
        """释放摄像头资源"""
        if self.cap.isOpened():
            self.cap.release()
