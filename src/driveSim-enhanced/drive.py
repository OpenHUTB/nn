#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepGTAV 车辆控制客户端
用于连接DeepGTAV服务器并发送控制指令控制车辆
"""

from deepgtav.messages import Start, Stop, Scenario, Commands, frame2numpy
from deepgtav.client import Client

import argparse
import time
import cv2
import numpy as np
from typing import Tuple, Optional


class Model:
    """
    简单的车辆控制模型类
    可根据实际需求替换为复杂的深度学习模型
    """
    
    def __init__(self):
        """初始化模型"""
        # 可以在此处加载预训练模型
        # 例如：self.model = load_model('path/to/model.h5')
        pass
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        图像预处理函数
        
        Args:
            frame: 原始输入图像
            
        Returns:
            预处理后的图像，适用于模型输入
        """
        # 示例：调整大小、归一化等操作
        # processed = cv2.resize(frame, (224, 224))
        # processed = processed / 255.0
        # return processed
        return frame
    
    def run(self, frame: np.ndarray) -> Tuple[float, float, float]:
        """
        模型推理函数，根据输入图像计算控制指令
        
        Args:
            frame: 输入图像（numpy数组格式）
            
        Returns:
            (throttle, brake, steering): 油门、刹车、转向角度的元组
            - throttle: 0-1之间的浮点数，表示油门开度
            - brake: 0-1之间的浮点数，表示刹车力度  
            - steering: -1到1之间的浮点数，负值左转，正值右转
        """
        # 预处理图像
        processed_frame = self.preprocess(frame)
        
        # 在这里实现实际的模型推理
        # 示例：使用简单的规则控制
        # output = self.model.predict(processed_frame)
        
        # 返回控制指令 [油门, 刹车, 转向]
        # 当前为示例值：不加速、不刹车、直行
        return [0.0, 0.0, 0.0]
    
    def run_with_visualization(self, frame: np.ndarray) -> Tuple[float, float, float]:
        """
        带可视化的模型推理（可选）
        可以在图像上绘制控制信息
        """
        # 获取控制指令
        throttle, brake, steering = self.run(frame)
        
        # 在图像上绘制控制信息
        h, w = frame.shape[:2]
        
        # 显示控制值
        cv2.putText(frame, f"Throttle: {throttle:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Brake: {brake:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Steering: {steering:.2f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示转向指示器
        center_x = w // 2
        steering_x = center_x + int(steering * 100)
        cv2.arrowedLine(frame, (center_x, h-50), (steering_x, h-50), 
                       (0, 0, 255), 3)
        
        # 显示图像
        cv2.imshow('DeepGTAV Control', frame)
        cv2.waitKey(1)  # 1ms延迟，允许窗口更新
        
        return throttle, brake, steering


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='DeepGTAV 车辆控制客户端')
    parser.add_argument('--ip', type=str, default='localhost', 
                       help='DeepGTAV服务器IP地址 (默认: localhost)')
    parser.add_argument('--port', type=int, default=8000, 
                       help='DeepGTAV服务器端口 (默认: 8000)')
    parser.add_argument('--duration', type=float, default=80.0, 
                       help='运行时长（小时） (默认: 80小时)')
    parser.add_argument('--fps', type=int, default=10, 
                       help='帧率 (默认: 10 FPS)')
    parser.add_argument('--width', type=int, default=320, 
                       help='图像宽度 (默认: 320)')
    parser.add_argument('--height', type=int, default=160, 
                       help='图像高度 (默认: 160)')
    parser.add_argument('--driving_mode', type=int, default=-1, 
                       help='驾驶模式 (-1: 手动, 0: 正常, 等)')
    parser.add_argument('--visualize', action='store_true', 
                       help='启用实时可视化')
    return parser.parse_args()


def main():
    """
    主函数：连接DeepGTAV服务器并控制车辆
    """
    # 解析命令行参数
    args = parse_arguments()
    
    print(f"正在连接到 DeepGTAV 服务器: {args.ip}:{args.port}")
    
    # 创建与DeepGTAV的连接
    # 可选参数：dataset_path, compression_level 可用于保存数据
    client = Client(args.ip, args.port)
    
    # 配置驾驶场景
    # drivingMode = -1 表示手动驾驶模式
    # 其他选项可参考 deepgtav/messages.py 中的 Scenario 类
    scenario = Scenario(
        drivingMode=args.driving_mode,  # 驾驶模式
        # 可选参数：
        # weather='EXTRASUNNY',     # 天气设置
        # vehicle='blista',         # 车辆类型
        # time=[12, 0],             # 时间设置
        # location=[x, y, z]        # 位置设置
    )
    
    # 发送启动请求到DeepGTAV
    # dataset参数可配置数据采集频率和图像尺寸
    client.sendMessage(Start(
        scenario=scenario,
        # dataset=Dataset(rate=args.fps, frame=[args.width, args.height])
    ))
    
    print(f"DeepGTAV客户端已启动")
    print(f"运行时长: {args.duration} 小时")
    print(f"帧率: {args.fps} FPS")
    print("按 Ctrl+C 停止程序\n")
    
    # 创建车辆控制模型实例
    model = Model()
    
    # 计算停止时间
    stop_time = time.time() + args.duration * 3600
    
    # 统计变量
    frame_count = 0
    start_time = time.time()
    
    try:
        # 主控制循环
        while time.time() < stop_time:
            # 接收DeepGTAV发送的消息（字典格式）
            message = client.recvMessage()
            
            # 如果消息为空，跳过处理
            if message is None:
                print("警告：接收到空消息")
                continue
            
            # 将帧数据转换为numpy数组
            # 指定图像尺寸 (width, height)
            image = frame2numpy(message['frame'], (args.width, args.height))
            
            # 使用模型计算控制指令
            if args.visualize:
                # 启用可视化模式
                throttle, brake, steering = model.run_with_visualization(image)
            else:
                # 普通模式
                throttle, brake, steering = model.run(image)
            
            # 将控制指令发送回DeepGTAV以控制车辆
            client.sendMessage(Commands(throttle, brake, steering))
            
            # 更新统计
            frame_count += 1
            
            # 每100帧打印一次性能信息
            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                print(f"处理帧数: {frame_count}, 平均FPS: {fps:.2f}, "
                      f"控制指令: [油门={throttle:.2f}, 刹车={brake:.2f}, 转向={steering:.2f}]")
                
    except KeyboardInterrupt:
        # 处理用户中断（Ctrl+C）
        print("\n\n用户中断程序")
    except Exception as e:
        # 处理其他异常
        print(f"\n发生错误: {type(e).__name__}: {e}")
    finally:
        # 清理资源
        print("\n正在关闭连接...")
        
        # 可选：发送停止命令
        # client.sendMessage(Stop())
        
        # 关闭客户端连接
        client.close()
        
        # 如果有可视化窗口，关闭它
        cv2.destroyAllWindows()
        
        # 打印最终统计
        if frame_count > 0:
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time
            print(f"\n最终统计:")
            print(f"总处理帧数: {frame_count}")
            print(f"总运行时间: {total_time:.2f} 秒")
            print(f"平均帧率: {avg_fps:.2f} FPS")
        
        print("程序正常退出")


if __name__ == '__main__':
    main()
