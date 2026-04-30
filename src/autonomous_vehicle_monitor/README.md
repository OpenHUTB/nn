CARLA 自动驾驶仿真平台
基于 CARLA 的自动驾驶可视化与安全验证仿真系统，集成多传感器感知、BEV 鸟瞰图、环境切换、数据记录、危险预警等功能，适用于算法验证、教学演示、毕业设计。
项目结构
plaintext
carla-rl-project-main/
├─ main.py                # 主运行文件（启动入口）
├─ blackbox.csv           # 车辆黑匣子数据记录
├─ recording.pkl          # 录制回放数据文件
└─ core/                  # 核心功能模块文件夹
   ├─ __init__.py
   ├─ sensors.py          # 相机+激光雷达管理
   ├─ npc_manager.py      # NPC车辆生成与管理
   ├─ recorder.py         # 数据录制
   ├─ player.py           # 回放控制
   ├─ blackbox.py         # 黑匣子记录
   ├─ map_drawer.py       # 车道线+BEV绘制
   ├─ ui_dashboard.py     # 虚拟仪表盘
   └─ traffic_light_monitor.py  # 红绿灯监测
已实现功能
🚗 基础功能
自动生成自车与 NPC 车辆，开启自动巡航
前 / 后 / 左 / 右 四路 RGB 相机实时显示
激光雷达点云在 BEV 视图可视化
BEV 鸟瞰图 + 车道线 + 可行驶区域绘制
独立双窗口：主监控画面 + 车辆仪表盘
🌤 环境切换（按键控制）
N：切换 白天 / 夜间
R：开启 雨天
F：开启 雾天
C：恢复 晴天
📊 数据与回放
黑匣子记录：时间、位置、车速、航向（保存为 CSV）
场景录制 / 回放：R 开始 / S 保存 / P 回放
⚠️ 安全预警功能
近距离车辆 / 行人检测
碰撞风险预警
急刹行为检测
危险时 BEV 变红 + 屏幕闪烁告警
🚦 辅助驾驶
实时红绿灯状态监测
车辆速度、转向、控制状态显示
安全距离验证、制动逻辑验证、避障效果验证
运行环境
Python 3.7+
CARLA 0.9.12 / 0.9.13 / 0.9.14
依赖库
bash
运行
pip install opencv-python numpy
启动方式
启动 CARLA 模拟器
运行主程序
bash
运行
python main.py
按键说明
表格
按键	功能
N	切换 白天 / 夜间
R	雨天模式
F	雾天模式
C	清空天气，恢复晴天
R	开始录制场景
S	停止并保存录制
P	回放录制数据
ESC	退出程序
适用场景
自动驾驶感知算法验证
安全距离与碰撞预警策略测试
不同天气环境下的感知鲁棒性测试
毕业设计 / 课程设计 / 教学演示
BEV 感知与可视化开发