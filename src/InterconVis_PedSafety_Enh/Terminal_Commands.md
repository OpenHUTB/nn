## 命令格式
```shell
python cvips_generation.py --town <城镇名称> [--num_vehicles < 数量 >] [--num_pedestrians < 数量 >] [--weather < 天气类型 >] [--time_of_day < 时段 >] [--seed < 种子值 >]
```
## 参数说明
|参数名|类型|默认值|可选值|作用说明|
|:---:|:---:|:---:|:---:|:---:|
|--town|字符串|town01|town01,town04等|指定要加载的CARLA地图|
|--num_vehicles|整数|20|>=0|要生成的自动驾驶车辆数量|
|--num_pedestrians|整数|100|>=0|要生成的自主行走的行人数量|
|--seed|整数|None|任意整数|随机种子，设置后可保证每次运行完全相同的场景，便于复现。|
|--weather|字符串|clear|clear，rainy，cloudy|设置天气，rainy会有明显的降雨和地面反光效果|
|--time_of_day|字符串|noon|noon，sunset，night|设置时间，night会切换到夜晚并开启月光|
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