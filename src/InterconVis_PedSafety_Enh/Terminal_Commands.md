# 1. 默认配置：在 Town04 生成 10 辆车辆，天气为晴天，时间为中午
```shell
python generate_vehicles_only.py --town Town04
```
# 2. 自定义车辆数量：在 Town04 生成 15 辆车辆
```shell
python generate_vehicles_only.py --town Town04 --num_vehicles 15
```
# 3. 自定义天气：在 Town04 生成 10 辆车辆，天气为雨天
```shell
python generate_vehicles_only.py --town Town04 --weather rainy
```
# 4. 自定义时间：在 Town04 生成 10 辆车辆，时间为夜晚
```shell
python generate_vehicles_only.py --town Town04 --time_of_day night
```
# 5. 完整自定义：在 Town07 生成 20 辆车辆，天气为多云，时间为日落
```shell
python generate_vehicles_only.py --town Town07 --num_vehicles 20 --weather cloudy --time_of_day sunset
```
# 6. 设置随机种子（保证场景可复现）：在 Town10HD 生成 12 辆车辆，使用种子 123
```shell
python generate_vehicles_only.py --town Town10HD --num_vehicles 12 --seed 123
```