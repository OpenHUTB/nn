from simulator import IndexSimulator

def main():
    # 1. 配置路径
    config_path = "config.yaml"
    model_path = "simulation.xml"

    # 2. 初始化仿真器+运行
    sim = IndexSimulator(config_path, model_path)
    sim.run_simulation()  # 所有打印都在run_simulation里，这里不用额外打印

if __name__ == "__main__":
    main()