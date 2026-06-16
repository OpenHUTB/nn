import subprocess
import time
import os
import platform  # 用于判断操作系统
import logging

# 初始化日志，修复 NameError
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)



# NOTE: Modify this to the appropriate Carla server launch command depending on your method of installation
# 优先读取环境变量，若未设置则回退到默认的 D 盘路径

LAUNCH_STRING = os.getenv("CARLA_SERVER_PATH", r"D:\WindowsNoEditor\CarlaUE4.exe")

def stop_carla_server():
    log.info("[INFO] Shut down existing Carla servers")
    if platform.system() == "Windows":
        # Windows 使用 taskkill 强制关闭进程
        os.system("taskkill /f /im CarlaUE4.exe >nul 2>&1")
    else:
        # Linux 保持原有的 killall
        os.system("killall -9 CarlaUE4-Optional")

def start_carla_server(carla_sh_str=LAUNCH_STRING):
    stop_carla_server()
    # 修复：移除 Windows 不支持的 preexec_fn，并根据系统选择参数
    if platform.system() == "Windows":
        subprocess.Popen(carla_sh_str, shell=True)
    else:
        subprocess.Popen(carla_sh_str, shell=True, preexec_fn=os.setsid)
    
    log.info("[INFO] Waiting for Carla server to start...")
    # 修复：将 self._t_sleep 改为具体的数字 10
    time.sleep(10)