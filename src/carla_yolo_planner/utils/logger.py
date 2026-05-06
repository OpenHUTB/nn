from pathlib import Path

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
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.step = 0
        self.closed = False
        print(f"[Logger] TensorBoard 日志已启动: {self.log_dir}")
        print(f"[Logger] 请运行命令查看: tensorboard --logdir={self.log_dir}")

    def log_step(self, fps, detection_count, avg_confidence=0):
        """
        记录每一步的系统状态
        """
        if self.closed:
            raise RuntimeError("PerformanceLogger 已关闭，无法继续写入日志")

        fps = float(fps)
        detection_count = max(0, int(detection_count))
        avg_confidence = float(avg_confidence) if detection_count > 0 else 0.0

        # 记录 FPS
        self.writer.add_scalar('Performance/FPS', fps, self.step)

        # 记录检测到的物体数量
        self.writer.add_scalar('Detection/Object_Count', detection_count, self.step)

        # 始终写入该序列，避免 TensorBoard 曲线因缺测点中断
        self.writer.add_scalar('Detection/Avg_Confidence', avg_confidence, self.step)

        self.step += 1

    def close(self):
        """
        关闭写入器
        """
        if self.closed:
            return

        self.writer.flush()
        self.writer.close()
        self.closed = True
        print("[Logger] 日志写入器已关闭")
