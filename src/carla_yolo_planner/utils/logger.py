import os
import time
from torch.utils.tensorboard import SummaryWriter


# ==========================================
# [Fix] 之前报错是因为 class 行末尾少了一个冒号 ':'
# ==========================================
class PerformanceLogger:
    """
    系统性能日志记录器 (基于 TensorBoard)
    用于记录 FPS、检测目标数量、平均置信度等指标
    """

    def __init__(self, log_dir='runs/experiment_1'):
        # 确保日志目录存在
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0
        print(f"[Logger] TensorBoard 日志已启动: {log_dir}")
        print(f"[Logger] 请运行命令查看: tensorboard --logdir={log_dir}")

    def log_step(self, fps, detection_count, avg_confidence=0):
        """
        记录每一步的系统状态
        """
        # 记录 FPS
        self.writer.add_scalar('Performance/FPS', fps, self.step)

        # 记录检测到的物体数量
        self.writer.add_scalar('Detection/Object_Count', detection_count, self.step)

        # 记录平均置信度 (如果有检测到物体)
        if detection_count > 0:
            self.writer.add_scalar('Detection/Avg_Confidence', avg_confidence, self.step)

        self.step += 1

    def close(self):
        """
        关闭写入器
        """
        self.writer.close()
        print("[Logger] 日志写入器已关闭")