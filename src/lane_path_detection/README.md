我选择的是Self driving car lane and path detection
自动驾驶汽车车道和路径检测
1. 环境要求
Python版本：3.6 或更高
2. 安装依赖包
bash
pip3 install -r requirements.txt
这将安装运行所需的所有库和依赖项。
3. 下载示例数据
从https://drive.google.com/file/d/1hP-v8lLn1g1jEaJUBYJhv1mEb32hkMvG/view下载示例数据文件。
（注意：更多数据将陆续添加）
4. 运行程序
bash
python3 main.py <你的hevc文件路径>
后续功能规划（即将推出）
使用YOLOv3进行物体检测，包括：
交通信号灯
车辆（汽车、卡车、摩托车）
自行车
行人
停车标志
实时或近实时的语义分割
快速SLAM（同步定位与地图构建）