#!/usr/bin/env python3
"""
run.py 位于 scripts 文件夹下
功能：修正路径，使其能找到同级模块(yolo_vehicle_counter)和上级资源
"""
import sys
from pathlib import Path

# 1. 获取 scripts 目录的路径（即当前文件所在目录）
scripts_dir = Path(__file__).parent

# 2. 关键点1：将 scripts 目录加入路径（确保能导入同目录的 yolo_vehicle_counter）
sys.path.insert(0, str(scripts_dir))

# 3. 关键点2：将项目根目录加入路径（因为 models 等文件夹在根目录，相对于 scripts 是 ../）
root_dir = scripts_dir.parent
sys.path.insert(0, str(root_dir))

# 4. 导入模块（此时结构同级，可直接导入）
try:
    from yolo_vehicle_counter import main
except ImportError as e:
    print(f"导入失败，请检查路径。当前搜索路径: {sys.path}")
    raise e

# 5. 执行主函数
# 注意：路径需要从根目录开始拼接，或者使用 root_dir 辅助
if __name__ == "__main__":
    # 方式A：基于根目录构建路径（推荐，最稳定）
    model_path = str(root_dir / "models" / "yolo11n.pt")
    input_video_path = str(root_dir / "dataset" / "sample.mp4")
    output_video_path = str(root_dir / "res" / "sample_res.mp4")
    
    # 方式B：如果运行报错，可尝试改为绝对路径测试
    # model_path = r"D:\github\nn\src\yolo11n_vehicle_counter\models\yolo11n.pt"
    
    main(model_path, input_video_path, output_video_path)