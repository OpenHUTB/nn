# 测试1：基础场景（Town04+4岔路口+晴天+中午）
```shell
python cvips_generation.py --town Town04 --intersection 4way --weather clear --time_of_day noon
```
# 测试2：复杂场景（Town10HD+3岔路口+雨天+日落+高行人密度）
```shell
python cvips_generation.py --town Town10HD --intersection 3way --weather rainy --time_of_day sunset --crowd_level high --crossing_percentage 0.3
```
# 测试3：夜间场景（Town07+4岔路口+阴天+夜间+自定义随机种子）
```shell
python cvips_generation.py --town Town07 --intersection 4way --weather cloudy --time_of_day night --seed_value 123 --scenario scenario00005
```
# 测试4：最简场景（Town04+3岔路口+晴天+日落）
```shell
python cvips_generation.py --town Town04 --intersection 3way --weather clear --time_of_day sunset
```