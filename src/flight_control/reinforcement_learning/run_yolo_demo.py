#!/usr/bin/env python3
"""
YOLO 目标检测演示脚本

该脚本展示如何启用 YOLO 目标检测并可视化检测结果
"""

import os
import sys
import time
import cv2
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reinforcement_learning.drone_env import DroneEnv

def run_yolo_demo():
    """运行 YOLO 目标检测演示"""
    print("=== YOLO 目标检测演示 ===")
    print("正在初始化环境...")
    
    try:
        # 创建启用 YOLO 的环境
        env = DroneEnv(use_yolo=True)
        print("环境初始化成功！")
        print(f"YOLO 状态: {'已启用' if env.use_yolo else '未启用'}")
        
        print("\n开始检测...")
        print("按 'q' 键退出")
        
        # 运行检测循环
        frame_count = 0
        start_time = time.time()
        
        while True:
            # 获取带标注的图像
            annotated_frame = env.get_annotated_frame()
            
            if annotated_frame is not None:
                # 计算 FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # 在图像上显示 FPS
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # 显示图像
                cv2.imshow("YOLO 目标检测演示", annotated_frame)
                
                # 等待按键
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            else:
                print("获取图像失败，跳过当前帧")
                time.sleep(0.5)
                
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        if 'env' in locals():
            env.close()
        cv2.destroyAllWindows()
        print("\n演示结束")

def run_with_custom_model(model_path):
    """使用自定义模型运行演示"""
    print(f"=== 使用自定义模型演示 ===")
    print(f"模型路径: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    try:
        env = DroneEnv(use_yolo=True, yolo_model_path=model_path)
        print("环境初始化成功！")
        
        # 运行一个简单的检测
        print("正在执行检测...")
        annotated = env.get_annotated_frame()
        
        if annotated is not None:
            print("检测完成，显示结果...")
            cv2.imshow("自定义模型检测结果", annotated)
            print("按任意键继续...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("获取图像失败")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()

def main():
    """主函数"""
    print("YOLO 目标检测脚本")
    print("1. 运行默认 YOLOv8 演示")
    print("2. 使用自定义模型")
    print("3. 退出")
    
    choice = input("请选择 (1-3): ")
    
    if choice == '1':
        run_yolo_demo()
    elif choice == '2':
        model_path = input("请输入模型路径: ")
        run_with_custom_model(model_path)
    elif choice == '3':
        print("退出")
    else:
        print("无效选择")

if __name__ == "__main__":
    main()
