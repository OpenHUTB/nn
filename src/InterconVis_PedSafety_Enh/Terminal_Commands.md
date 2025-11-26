## 命令格式
```shell
python cvips_generation.py --town <城镇名称> [--num_vehicles < 数量 >] [--num_pedestrians < 数量 >] [--weather < 天气类型 >] [--time_of_day < 时段 >] [--seed < 种子值 >]
```
## 参数说明
--town: (必填) CARLA 城镇地图名称 (例如: Town01, Town04)--num_vehicles: (可选) 生成车辆数量，默认值为 20--num_pedestrians:(可选) 生成行人数量，默认值为 100--weather: (可选) 天气类型，可选值: clear (晴天), rainy (雨天), cloudy (多云)，默认值为 clear--time_of_day: (可选) 时段，可选值: noon (中午), sunset (日落), night (夜晚)，默认值为 noon--seed: (可选) 随机种子，用于复现相同场景
## 一、基础场景命令 (核心参数覆盖)
### 1. Town01 + 晴天 + 中午 (默认配置)
```shell
python cvips_generation.py --town Town01
```
### 2. Town01 + 雨天 + 夜晚
```shell
python cvips_generation.py --town Town01 --weather rainy --time_of_day night
```
### 3. Town04 + 多云 + 日落
```shell
python cvips_generation.py --town Town04 --weather cloudy --time_of_day sunset
```
### 4. Town01 + 晴天 + 夜晚
```shell
python cvips_generation.py --town Town01 --time_of_day night
```
### 5. Town04 + 雨天 + 中午
```shell
python cvips_generation.py --town Town04 --weather rainy
```
## 二、不同密度场景命令
### 6. Town01 + 低密度 (10 辆车，50 个行人)
```shell
python cvips_generation.py --town Town01 --num_vehicles 10 --num_pedestrians 50
```
### 7. Town01 + 中密度 (25 辆车，150 个行人)
```shell
python cvips_generation.py --town Town01 --num_vehicles 25 --num_pedestrians 150
```
### 8. Town04 + 高密度 (40 辆车，250 个行人)
```shell
python cvips_generation.py --town Town04 --num_vehicles 40 --num_pedestrians 250
```
## 三、随机种子与场景复现命令

### 9. Town01 + 种子 123 (可复现)
```shell
python cvips_generation.py --town Town01 --seed 123
```
### 10. Town04 + 种子 456 (可复现)
```shell
python cvips_generation.py --town Town04 --seed 456
```
### 11. Town01 + 雨天夜晚 + 种子 789 (可复现)
```shell
python cvips_generation.py --town Town01 --weather rainy --time_of_day night --seed 789
```
## 四、多参数组合场景命令
### 12. Town01 + 15 辆车 + 80 个行人 + 雨天 + 日落 + 种子 111
```shell
python cvips_generation.py --town Town01 --num_vehicles 15 --num_pedestrians 80 --weather rainy --time_of_day sunset --seed 111
```
### 13. Town04 + 30 辆车 + 200 个行人 + 多云 + 夜晚 + 种子 222
```shell
python cvips_generation.py --town Town04 --num_vehicles 30 --num_pedestrians 200 --weather cloudy --time_of_day night --seed 222
```
## 五、边缘场景命令 (极限参数)
### 14. Town01 + 极限高密度 (35 辆车，220 个行人) + 雨天 + 夜晚
```shell
python cvips_generation.py --town Town01 --num_vehicles 35 --num_pedestrians 220 --weather rainy --time_of_day night
```
### 15. Town04 + 极限低密度 (5 辆车，20 个行人) + 晴天 + 中午
```shell
python cvips_generation.py --town Town04 --num_vehicles 5 --num_pedestrians 20 --weather clear --time_of_day noon
```