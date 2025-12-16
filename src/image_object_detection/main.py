# main.py
import matplotlib
matplotlib.use('Agg')  # 必须在最前，禁用 GUI 后端

from ui_handler import UIHandler
from config import Config

def main():
    config = Config()
    ui_handler = UIHandler(config)
    ui_handler.run()

if __name__ == "__main__":
    main()
