import argparse
import importlib
import os
import sys

# 定义项目各模块信息
MODULES = {
    "embodied_robot": "双轮机器人仿真",
    "mujoco01": "人形机器人仿真",
    "humantest": "机械臂力控仿真",
    "chap06_RNN": "唐诗生成RNN模型",
    "manual_control": "CARLA车辆手动控制",
    "drone_perception": "无人机感知模型训练"
}

def print_banner():
    """打印项目横幅"""
    print(r"""
    ==================================================
           仿真与深度学习项目集合
    ==================================================
    """)

def list_modules():
    """列出所有可用模块"""
    print("可用模块:")
    for name, desc in MODULES.items():
        print(f"  {name:15} - {desc}")
    print()

def run_module(module_name):
    """运行指定模块"""
    if module_name not in MODULES:
        print(f"错误: 模块 '{module_name}' 不存在")
        return

    try:
        # 构建模块路径
        module_path = f"nn.src.{module_name}.main"
        
        # 动态导入模块
        module = importlib.import_module(module_path)
        
        # 检查是否有main函数
        if hasattr(module, "main"):
            print(f"启动模块: {module_name} - {MODULES[module_name]}")
            module.main()
        else:
            print(f"错误: 模块 '{module_name}' 中未找到 main 函数")
            
    except ImportError as e:
        print(f"导入模块失败: {e}")
    except Exception as e:
        print(f"模块运行出错: {e}")

def main():
    print_banner()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="项目主入口，用于运行各个子模块")
    parser.add_argument("module", nargs='?', help="要运行的模块名称")
    parser.add_argument("--list", action="store_true", help="列出所有可用模块")
    
    args = parser.parse_args()
    
    if args.list:
        list_modules()
    elif args.module:
        run_module(args.module)
    else:
        # 如果没有提供参数，显示帮助信息
        parser.print_help()
        print()
        list_modules()

if __name__ == "__main__":
    # 添加项目根目录到Python路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()