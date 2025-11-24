# 1. 基础场景：各地图+路口类型验证
```shell
python cvips_generation.py --town Town04 --intersection 4way --weather clear --time_of_day noon
python cvips_generation.py --town Town10HD --intersection 3way --weather clear --time_of_day sunset
python cvips_generation.py --town Town07 --intersection 4way --weather cloudy --time_of_day noon
```
# 2. 天气+时段组合场景（覆盖所有环境组合）
```shell
python cvips_generation.py --town Town04 --intersection 3way --weather rainy --time_of_day night
python cvips_generation.py --town Town10HD --intersection 4way --weather cloudy --time_of_day sunset
python cvips_generation.py --town Town07 --intersection 3way --weather clear --time_of_day night
python cvips_generation.py --town Town04 --intersection 4way --weather rainy --time_of_day noon
```
# 3. 边界场景（极限环境测试）
```shell
python cvips_generation.py --town Town10HD --intersection 4way --weather rainy --time_of_day night
python cvips_generation.py --town Town07 --intersection 3way --weather cloudy --time_of_day night
```
# 4. 常用场景快捷指令
```shell
python cvips_generation.py --town Town04 --intersection 3way --weather clear --time_of_day noon
python cvips_generation.py --town Town10HD --intersection 4way --weather clear --time_of_day sunset
```