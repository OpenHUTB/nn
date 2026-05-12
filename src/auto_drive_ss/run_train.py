import os
import sys

# 把src加入环境路径
sys.path.append(os.path.join(os.getcwd(), "src"))

# 这里改成直接运行脚本，而不是导入不存在的函数
if __name__ == "__main__":
    print("===== 开始启动 CARLA 强化学习训练 =====")
    print("请确保 CarlaUE4 模拟器已提前开启")
    # 直接执行训练脚本
    exec(open(os.path.join("src", "rl_training_with_ssl.py")).read())