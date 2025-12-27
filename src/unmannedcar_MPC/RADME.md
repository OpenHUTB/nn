\#!/usr/bin/env python3

"""

PyGameDrawer - 绘图显示模块完整说明文档

运行此文件查看drawer.py中所有函数说明

"""

一、模块概述

PyGameDrawer是基于Pygame的图形绘制模块，用于在Carla仿真环境中显示各种驾驶信息。

模块提供了丰富的可视化功能，包括速度显示、转向显示、驾驶评分、辅助线等。



二、坐标转换函数

1\. \_\_w\_locs\_2\_camera\_locs(w\_locs)

&nbsp;  - 功能：将世界坐标转换为相机屏幕坐标

&nbsp;  - 参数：w\_locs - 世界坐标列表

&nbsp;  - 返回：屏幕坐标列表



2\. draw\_camera\_text(location, color, text)

&nbsp;  - 功能：在相机视角下绘制文本

&nbsp;  - 参数：location - 世界坐标，color - 颜色，text - 文本内容



3\. draw\_camera\_circles(w\_locs, color, radius)

&nbsp;  - 功能：在相机视角下绘制圆形

&nbsp;  - 参数：w\_locs - 世界坐标列表，color - 颜色，radius - 半径



4\. draw\_camera\_polygon(w\_locs, color)

&nbsp;  - 功能：在相机视角下绘制多边形

&nbsp;  - 参数：w\_locs - 世界坐标列表，color - 颜色



5\. draw\_camera\_lines(color, w\_locs, width=1)

&nbsp;  - 功能：在相机视角下绘制线条

&nbsp;  - 参数：color - 颜色，w\_locs - 世界坐标列表，width - 线宽



6\. \_\_draw\_camera\_line\_safe(color, pt1, pt2, width=1)

&nbsp;  - 功能：安全绘制线条（检查边界）

&nbsp;  - 参数：color - 颜色，pt1 - 起点，pt2 - 终点，width - 线宽



7\. draw\_point(location, color, radius=3)

&nbsp;  - 功能：在相机视角下绘制点

&nbsp;  - 参数：location - 世界坐标，color - 颜色，radius - 半径



三、信息显示函数

1\. display\_speed(speed\_kmh)

&nbsp;  - 功能：显示当前速度（右上角）

&nbsp;  - 参数：speed\_kmh - 速度值（km/h）

&nbsp;  - 特性：颜色编码（绿/黄/橙/红），速度条可视化



2\. display\_brake\_status(is\_braking, brake\_history, target\_speed, frame\_count)

&nbsp;  - 功能：显示刹车状态（左上角）

&nbsp;  - 参数：

&nbsp;    - is\_braking: 当前是否刹车

&nbsp;    - brake\_history: 刹车历史记录

&nbsp;    - target\_speed: 目标速度

&nbsp;    - frame\_count: 帧计数

&nbsp;  - 特性：闪烁效果，测试模式，颜色编码



3\. display\_speed\_history(speed\_history, target\_speed)

&nbsp;  - 功能：显示速度历史图表（左下角）

&nbsp;  - 参数：

&nbsp;    - speed\_history: 速度历史记录

&nbsp;    - target\_speed: 目标速度

&nbsp;  - 特性：趋势线，目标线，当前速度标注



4\. display\_steering(steer\_angle)

&nbsp;  - 功能：显示转向角度（右下角）

&nbsp;  - 参数：steer\_angle - 转向角度值（-1到1）

&nbsp;  - 特性：角度可视化，颜色编码，方向箭头



5\. display\_throttle\_info(throttle\_value, brake\_value)

&nbsp;  - 功能：显示油门和刹车信息（右侧中部）

&nbsp;  - 参数：

&nbsp;    - throttle\_value: 油门值（0-1）

&nbsp;    - brake\_value: 刹车值（0-1）

&nbsp;  - 特性：条形图，实时更新



6\. display\_control\_mode(control\_mode)

&nbsp;  - 功能：显示控制模式（顶部中央）

&nbsp;  - 参数：control\_mode - 控制模式（AUTO/MANUAL）

&nbsp;  - 特性：不同颜色标识



7\. display\_frame\_info(frame\_count, dt)

&nbsp;  - 功能：显示帧信息（左下角）

&nbsp;  - 参数：

&nbsp;    - frame\_count: 帧计数

&nbsp;    - dt: 时间间隔

&nbsp;  - 特性：FPS计算，运行时间



8\. display\_collision\_warning(collision\_warning, collision\_history)

&nbsp;  - 功能：显示碰撞警告（中央上方）

&nbsp;  - 参数：

&nbsp;    - collision\_warning: 当前碰撞警告状态

&nbsp;    - collision\_history: 碰撞警告历史

&nbsp;  - 特性：闪烁效果，脉冲动画，渐变背景



9\. display\_driving\_score(score, score\_factors, score\_history)

&nbsp;  - 功能：显示驾驶评分（左侧中部）

&nbsp;  - 参数：

&nbsp;    - score: 综合评分

&nbsp;    - score\_factors: 各项评分因子

&nbsp;    - score\_history: 评分历史记录

&nbsp;  - 特性：等级评定（A/B/C/D），趋势图，五项评分指标



10\. display\_waypoint\_navigation(current\_index, waypoints, distance\_to\_waypoint, reached\_count, progress)

&nbsp;   - 功能：显示航点导航信息（中央下方）

&nbsp;   - 参数：

&nbsp;     - current\_index: 当前航点索引

&nbsp;     - waypoints: 航点位置列表

&nbsp;     - distance\_to\_waypoint: 到当前航点的距离

&nbsp;     - reached\_count: 已到达航点计数

&nbsp;     - progress: 航点进度（0-1）

&nbsp;   - 特性：进度条，航点指示器，脉冲效果



11\. draw\_waypoint\_indicators(waypoints, current\_index)

&nbsp;   - 功能：绘制航点指示器（屏幕顶部）

&nbsp;   - 参数：

&nbsp;     - waypoints: 航点位置列表

&nbsp;     - current\_index: 当前航点索引

&nbsp;   - 特性：不同状态不同颜色，当前航点脉冲效果



四、驾驶辅助系统函数

1\. display\_driving\_assist\_lines(vehicle\_location, vehicle\_transform, steer\_angle, path=None)

&nbsp;  - 功能：显示驾驶辅助线和预期路径（屏幕中央）

&nbsp;  - 参数：

&nbsp;    - vehicle\_location: 车辆位置

&nbsp;    - vehicle\_transform: 车辆变换

&nbsp;    - steer\_angle: 转向角度

&nbsp;    - path: 预期路径（可选）

&nbsp;  - 特性：

&nbsp;    - 中心参考线

&nbsp;    - 转向辅助线（弧线）

&nbsp;    - 安全距离线

&nbsp;    - 车道保持辅助线

&nbsp;    - 预期路径显示

&nbsp;    - 车辆图标和前进方向



2\. display\_simple\_radar(vehicle\_location, obstacles=None)

&nbsp;  - 功能：显示简单雷达图（检测周围环境）（右下角）

&nbsp;  - 参数：

&nbsp;    - vehicle\_location: 车辆位置

&nbsp;    - obstacles: 障碍物列表（可选）

&nbsp;  - 特性：

&nbsp;    - 雷达网格

&nbsp;    - 障碍物显示（不同颜色表示不同类型）

&nbsp;    - 扫描线旋转效果

&nbsp;    - 近距离障碍物警告



五、静态工具函数

1\. get\_location\_bbox(location, camera)

&nbsp;  - 功能：获取位置在相机中的边界框

&nbsp;  - 参数：location - 世界坐标，camera - 相机对象



2\. location\_to\_sensor\_cords(cords, location, sensor)

&nbsp;  - 功能：将坐标转换为传感器坐标

&nbsp;  - 参数：cords - 坐标，location - 位置，sensor - 传感器



3\. location\_to\_world\_cords(cords, location)

&nbsp;  - 功能：将坐标转换为世界坐标

&nbsp;  - 参数：cords - 坐标，location - 位置



4\. \_create\_vehicle\_bbox\_points(vehicle)

&nbsp;  - 功能：创建车辆3D边界框点

&nbsp;  - 参数：vehicle - 车辆对象



5\. \_vehicle\_to\_sensor\_cords(cords, vehicle, sensor)

&nbsp;  - 功能：将车辆边界框坐标转换为传感器坐标

&nbsp;  - 参数：cords - 坐标，vehicle - 车辆，sensor - 传感器



6\. \_vehicle\_to\_world\_cords(cords, vehicle)

&nbsp;  - 功能：将车辆边界框坐标转换为世界坐标

&nbsp;  - 参数：cords - 坐标，vehicle - 车辆



7\. \_world\_to\_sensor\_cords(cords, sensor)

&nbsp;  - 功能：将世界坐标转换为传感器坐标

&nbsp;  - 参数：cords - 坐标，sensor - 传感器



8\. get\_matrix(transform)

&nbsp;  - 功能：从Carla变换创建矩阵

&nbsp;  - 参数：transform - Carla变换对象



六、使用说明

1\. 初始化：在main.py中通过PyGameDrawer(main)创建实例

2\. 调用：每帧调用相应的显示函数

3\. 坐标：大多数函数使用世界坐标，模块内部处理坐标转换

4\. 颜色：使用RGBA格式，支持透明度



七、可视化效果特性

1\. 颜色编码：根据数值大小使用不同颜色（绿/黄/橙/红）

2\. 动画效果：闪烁、脉冲、渐变等

3\. 历史趋势：速度、评分等历史数据显示

4\. 实时更新：所有显示内容每帧更新

5\. 边界检查：确保绘制内容在屏幕内



================================================================================

"""

