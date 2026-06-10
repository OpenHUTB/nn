import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

AVAILABLE_MAPS = [
    "Town01", "Town02", "Town03", "Town04", "Town05",
    "Town06", "Town07", "Town10HD", "Town11", "Town12",
    "Town13", "Town15"
]

MAP_DESCRIPTION = {
    "Town01": "基础小镇，适合入门测试",
    "Town02": "乡村道路为主，交通设施较少",
    "Town03": "城市核心区，红绿灯+限速牌最密集（推荐）",
    "Town04": "高速公路场景，长直路多",
    "Town05": "大型城市，复杂路口多",
    "Town06": "山区道路，弯道较多",
    "Town07": "环岛密集的城市",
    "Town10HD": "高清大地图，场景丰富",
    "Town11": "工业区域，仓库和厂房多",
    "Town12": "现代都市，高楼林立",
    "Town13": "港口区域，集装箱码头",
    "Town15": "最新地图，综合场景",
}

CONFIG_FILE = os.path.join(SCRIPT_DIR, "map_config.txt")

def get_current_map():
    try:
        import carla
        client = carla.Client("localhost", 2000)
        client.set_timeout(5.0)
        return client.get_world().get_map().name
    except Exception:
        return None

def save_map_choice(map_name):
    with open(CONFIG_FILE, 'w') as f:
        f.write(map_name)

def load_map_choice():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return f.read().strip()
    return None

def interactive_menu():
    current = get_current_map()
    
    print("=" * 50)
    print("  CARLA 地图切换工具")
    print("=" * 50)
    if current:
        print(f"  当前地图: {current}")
    print()
    
    for i, m in enumerate(AVAILABLE_MAPS):
        desc = MAP_DESCRIPTION.get(m, "")
        marker = " *" if m == current else ""
        print(f"  [{i+1:2d}] {m:<12s} - {desc}{marker}")
    
    print()
    print(f"  [ 0] 退出")
    print()
    
    saved = load_map_choice()
    if saved and saved in AVAILABLE_MAPS:
        default_idx = AVAILABLE_MAPS.index(saved) + 1
        print(f"  上次选择: {saved} (直接回车使用)")
    
    try:
        choice = input("\n  请输入编号: ").strip()
        
        if choice == "" and saved and saved in AVAILABLE_MAPS:
            map_name = saved
        elif choice == "0":
            print("  已退出")
            return
        else:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(AVAILABLE_MAPS):
                print(f"  错误: 无效编号 {choice}")
                return
            map_name = AVAILABLE_MAPS[idx]
        
        print(f"\n  正在切换到 {map_name} ...")
        
        import carla
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        world = client.load_world(map_name)
        actual = world.get_map().name
        
        save_map_choice(map_name)
        print(f"  已加载: {actual}")
        print(f"  已保存选择，运行 main.py 时将自动使用此地图")
        
    except ValueError:
        print("  错误: 请输入数字编号")
    except Exception as e:
        print(f"  错误: {e}")
        print("  请确保CARLA模拟器正在运行")

if __name__ == "__main__":
    interactive_menu()
