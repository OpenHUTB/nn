import argparse
from simulator import Simulator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--model", default="simulation.xml", help="MuJoCo模型路径")
    args = parser.parse_args()

    # 初始化仿真器
    sim = Simulator(config_path=args.config, model_path=args.model)
    # 运行仿真循环
    while sim.step():
        pass
    # 关闭资源
    sim.close()

if __name__ == "__main__":
    main()