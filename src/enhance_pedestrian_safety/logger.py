import logging
import sys
import os
from datetime import datetime


class Logger:
    """日志管理器"""

    @staticmethod
    def setup_logger(name, log_dir=None, level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(level)

        if not logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

        return logger

    @staticmethod
    def get_progress_bar(iteration, total, length=50):
        percent = iteration / total
        filled = int(length * percent)
        bar = '█' * filled + '░' * (length - filled)
        return f"[{bar}] {percent:.1%}"