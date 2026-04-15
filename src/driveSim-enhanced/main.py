from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

import cv2 
from tensorflow.keras.models import load_model
from common.tools.lib.parser import parser
import sys

#------------------------- 检测Ctrl+C并关闭客户端
#!/usr/bin/env python
import signal
import sys

def signal_handler(sig, frame):
  client.close()
  sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


#-------------------------------- 导入Vpilot模块

from deepgtav.messages import Start, Stop, Scenario, Commands, frame2numpy, Dataset
from deepgtav.client import Client

import argparse
import time

#------------------------------- 导入结束


# 连接到DeepGTAV服务器
client = Client(ip="localhost", port=8000)

#camerafile = "sample.hevc"
# 加载预训练的supercombo模型
supercombo = load_model('/home/idir/Bureau/modeld-master/models/supercombo.keras')

# 常量定义
MAX_DISTANCE = 140.
LANE_OFFSET = 1.8
MAX_REL_V = 10.

LEAD_X_SCALE = 10
LEAD_Y_SCALE = 10

#cap = cv2.VideoCapture(camerafile)

# 处理的帧数
NBFRAME = 10000

def frame_to_tensorframe(frame):                                                                                               
  """将帧转换为张量格式，为模型输入做准备"""
  H = (frame.shape[0]*2)//3                                                                                                
  W = frame.shape[1]                                                                                                       
  in_img1 = np.zeros((6, H//2, W//2), dtype=np.uint8)                                                      
                                                                                                                            
  in_img1[0] = frame[0:H:2, 0::2]                                                                                    
  in_img1[1] = frame[1:H:2, 0::2]                                                                                    
  in_img1[2] = frame[0:H:2, 1::2]                                                                                    
  in_img1[3] = frame[1:H:2, 1::2]                                                                                    
  in_img1[4] = frame[H:H+H//4].reshape((-1, H//2,W//2))                                                              
  in_img1[5] = frame[H+H//4:H+H//2].reshape((-1, H//2,W//2))
  return in_img1

def vidframe2img_yuv_reshaped():
  """从客户端接收消息并将帧转换为YUV格式"""
  message = client.recvMessage()  
                
  # 帧是可以传递给CNN处理的数据格式     
  frame = frame2numpy(message['frame'], (1164,874))
  #ret, frame = cap.read()
  img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
  return frame, img_yuv.reshape((874 * 3//2, 1164))

def vidframe2frame_tensors():
  """从视频帧获取张量格式的数据"""
  frame, img = vidframe2img_yuv_reshaped()
  imgs_med_model = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                    output_size=(512,256))
  f2t = frame_to_tensorframe(np.array(imgs_med_model)).astype(np.float32)/128.0 - 1.0
  return frame, f2t

# 初始化状态和期望
state = np.zeros((1,512))
desire = np.zeros((1,8))


#----------------------- 配置Vpilot

# 创建场景：设置驾驶模式、天气、车辆、时间、位置
scenario = Scenario(drivingMode=[786603,100.0], weather='EXTRASUNNY',vehicle='blista',time=[12,0],location=[-2573.13916015625, 3292.256103515625, 13.241103172302246]) # 手动驾驶 drivingMode=-1
dataset = Dataset(rate=20, frame=[1164,874])
client.sendMessage(Start(scenario=scenario, dataset = dataset)) 

#------------------------ 配置结束


# frame_tensors = np.zeros((NBFRAME,6,128,256))
# for i in tqdm(range(NBFRAME)):
#     frame_tensors[i] = vidframe2frame_tensors()[1]

# cap2 = cv2.VideoCapture("sample.hevc")

# 主循环：处理视频帧并进行推理
for i in tqdm(range(NBFRAME-1)):
  if i == 0:
    # 第一帧处理
    frame, frame_tensors1 = vidframe2frame_tensors()
  else :
    # 获取下一帧
    frame, frame_tensors2 = vidframe2frame_tensors()
    # 准备模型输入
    inputs = [np.vstack([frame_tensors1, frame_tensors2])[None], desire, state]
    # inputs = [np.vstack(frame_tensors[i:i+2])[None], desire, state]
    
    # 模型推理
    outs = supercombo.predict(inputs)
    print(outs)
    
    # 解析模型输出
    parsed = parser(outs)
    
    # 重要：重新设置状态
    state = outs[-1]
    pose = outs[-2]
    
    # ret, frame = cap2.read()
    # frame = cv2.resize(frame, (640, 420))
    
    # 可视化：显示原始摄像头图像
    plt.clf()
    plt.subplot(1, 2, 1) # 第1行，第2列，索引1
    plt.title("原始视频画面")
    plt.imshow(frame, aspect="auto")
    
    # 清除绘图，准备下一帧
    plt.subplot(1, 2, 2) # 第1行，第2列，索引2
    plt.title("预测结果")
    
    # lll = 左车道线
    plt.plot(parsed["lll"][0], range(0,192), "b-", linewidth=1)
    
    # rll = 右车道线
    plt.plot(parsed["rll"][0], range(0, 192), "r-", linewidth=1)
    
    # path = 预测路径
    plt.plot(parsed["path"][0], range(0, 192), "g-", linewidth=1)

    # 由于标准坐标系中左车道线是正的，右车道线是负的，需要反转X轴
    plt.gca().invert_xaxis()
    plt.pause(0.001)

plt.show()
