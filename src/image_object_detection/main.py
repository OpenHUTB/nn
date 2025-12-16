# main.py
from ui_handler import UIHandler
from config import Config

def main():
    config = Config()
    ui_handler = UIHandler(config)
    ui_handler.run()

if __name__ == "__main__":
    main()