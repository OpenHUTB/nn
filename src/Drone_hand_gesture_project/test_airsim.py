# -*- coding: utf-8 -*-
"""
AirSim 连接测试
"""

def test_connection():
    """测试 AirSim 连接"""
    print("=" * 60)
    print("AirSim 连接测试")
    print("=" * 60)
    
    try:
        from airsim_controller import AirSimController
        
        controller = AirSimController()
        
        print("\n[1/3] 尝试连接 AirSim...")
        if controller.connect():
            print("✅ AirSim 连接成功")
            
            print("\n[2/3] 获取无人机状态...")
            state = controller.get_state()
            print(f"  位置：{state['position']}")
            print(f"  速度：{state['velocity']}")
            print(f"  姿态：{state['orientation']}")
            
            print("\n[3/3] 测试相机...")
            img = controller.get_camera_image()
            if img is not None:
                print(f"✅ 相机工作正常，图像尺寸：{img.shape}")
            else:
                print("⚠️ 相机未返回图像")
            
            controller.disconnect()
            
            print("\n" + "=" * 60)
            print("✅ 所有测试通过！")
            return True
        else:
            print("❌ AirSim 连接失败")
            print("\n请检查:")
            print("  1. AirSim 模拟器是否运行")
            print("  2. settings.json 是否正确配置")
            print("  3. 防火墙设置")
            return False
            
    except ImportError as e:
        print(f"❌ 导入错误：{e}")
        print("请运行：pip install airsim")
        return False
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_connection()
