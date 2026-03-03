# test_connection.py
import airsim
import time

print("=" * 60)
print("无人机连接测试")
print("=" * 60)

print("\n请确保：")
print("1. AbandonedPark.exe 已运行")
print("2. 模拟器完全加载（看到公园场景）")
print("3. 切换到无人机模式（如果需要）")

input("\n按回车键开始测试...")

try:
    # 1. 连接到模拟器
    print("\n连接到模拟器...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✅ 连接成功！")

    # 2. 检查无人机状态
    print("\n检查无人机状态...")
    state = client.getMultirotorState()
    print(f"位置: X={state.kinematics_estimated.position.x_val:.1f}, "
          f"Y={state.kinematics_estimated.position.y_val:.1f}, "
          f"Z={state.kinematics_estimated.position.z_val:.1f}")

    # 3. 解锁无人机
    print("\n解锁无人机...")
    client.enableApiControl(True)
    client.armDisarm(True)
    print("✅ 无人机已解锁")

    # 4. 起飞测试
    print("\n起飞测试...")
    client.takeoffAsync().join()
    time.sleep(2)
    print("✅ 起飞成功")

    # 5. 捕获图像测试
    print("\n捕获图像测试...")
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])

    if responses:
        import numpy as np

        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)

        # 保存测试图像
        import cv2

        cv2.imwrite("test_image.jpg", img_rgb)
        print(f"✅ 图像捕获成功，保存为 test_image.jpg")
        print(f"   图像尺寸: {responses[0].width}x{responses[0].height}")

    # 6. 简单移动测试
    print("\n移动测试...")
    print("向前移动...")
    client.moveByVelocityAsync(2, 0, 0, 2).join()
    time.sleep(1)

    print("向右移动...")
    client.moveByVelocityAsync(0, 2, 0, 2).join()
    time.sleep(1)

    print("向左移动...")
    client.moveByVelocityAsync(0, -2, 0, 2).join()
    time.sleep(1)

    # 7. 返回起点并降落
    print("\n返回起点...")
    client.moveToPositionAsync(0, 0, -10, 3).join()

    print("降落...")
    client.landAsync().join()
    print("✅ 降落成功")

    # 8. 锁定无人机
    client.armDisarm(False)
    client.enableApiControl(False)

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("你的环境已准备就绪，可以开始项目开发。")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    print("\n可能的解决方案：")
    print("1. 确保模拟器已运行")
    print("2. 在模拟器中切换到无人机模式：")
    print("   - 按 ~ 键打开控制台")
    print("   - 输入: vehicle change Drone")
    print("   - 或尝试: vehicle Drone")
    print("3. 等待模拟器完全加载")

input("\n按回车键退出...")
