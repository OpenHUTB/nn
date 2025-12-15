import os,psutil,time
import subprocess
from gym_carla.setting import CARLA_PATH

operating_system='windows' if os.name=='nt' else 'linux'

def get_binary():
    return 'CarlaUE4.exe' if operating_system=='windows' else 'CarlaUE4.sh'

def get_exec_command():
    binary=get_binary()
    exec_command=binary if operating_system=='windows' else ('./'+binary)

    return binary,exec_command
#
def kill_process():
    binary=get_binary()
    for process in psutil.process_iter():
        if process.name().lower().startswith(binary.split('.')[0].lower()):
            try:
                process.terminate()
            except:
                pass

    still_alive=[]
    for process in psutil.process_iter():
        if process.name().lower().startswith(binary.split('.')[0].lower()):
            still_alive.append(process)

    if len(still_alive):
        for process in still_alive:
            try:
                process.kill()
            except:
                pass
        psutil.wait_procs(still_alive)

# Starts Carla simulator
def start_process():
    # Kill Carla processes if there are any and start simulator
    print('Starting Carla...')
    kill_process()
    subprocess.Popen(get_exec_command()[1],cwd=CARLA_PATH, shell=True)
    time.sleep(5)