# 最简测试脚本，只做打印和CARLA连接
print("✅ 脚本已启动！")
import carla
print("✅ 导入 carla 成功！")
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.get_world()
print("✅ 连接 CARLA 成功！")
print("✅ 所有步骤完成！")