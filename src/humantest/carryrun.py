import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from carryzhan import Robot, Grid, CellType

def test_robot_movement():
    """测试机器人移动功能"""
    print("测试机器人移动功能...")
    
    # 创建一个简单的网格
    grid = Grid()
    
    # 创建机器人
    robot = Robot(0, 0)
    
    # 测试A*算法
    start = (0, 0)
    goal = (4, 4)
    
    # 添加一些障碍物
    for x in range(1, 4):
        grid.add_obstacle(x, 1)
    
    # 计算路径
    path = robot.a_star(grid.cells, start, goal)
    
    print(f"起点: {start}, 终点: {goal}")
    print(f"找到的路径: {path}")
    
    if path:
        print("路径规划成功!")
    else:
        print("路径规划失败!")
    
    return path

def test_brick_pickup_and_drop():
    """测试机器人拾取和放置砖块功能"""
    print("\n测试机器人拾取和放置砖块功能...")
    
    # 创建一个简单的网格
    grid = Grid()
    
    # 添加一个砖块和一个目标位置
    grid.add_brick(2, 2)
    grid.add_target(4, 4)
    
    # 创建机器人
    robot = Robot(0, 0)
    
    print(f"初始机器人位置: ({robot.x}, {robot.y})")
    print(f"初始状态: {robot.state}")
    print(f"携带砖块: {robot.carrying_brick}")
    
    # 手动动更新机器人状态，直到完成一次搬运
    steps = 0
    max_steps = 100
    
    while steps < max_steps:
        robot.update(grid.cells)
        steps += 1
        
        print(f"步骤 {steps}: 位置({robot.x}, {robot.y}), 状态: {robot.state}, 携带砖块: {robot.carrying_brick}")
        
        # 检查是否完成一次搬运
        if robot.state == State.IDLE and robot.carrying_brick is False:
            # 检查砖块是否被移动到目标位置
            if grid.cells[4][4] == CellType.BRICK_ON_TARGET:
                print("砖块成功搬运到目标位置!")
                return True
            else:
                print("砖块未被正确放置到目标位置")
                return False
    
    print(f"超过最大步数({max_steps})，测试失败")
    return False

def test_multiple_bricks():
    """测试多个机器人人搬运多个砖块"""
    print("\n测试机器人人搬运多个砖块...")
    
    # 创建一个简单的网格
    grid = Grid()
    
    # 添加多个砖块和目标位置
    brick_positions = [(1, 1), (1, 3), (3, 1), (3, 3)]
    target_positions = [(6, 6), (6, 7), (7, 6), (7, 7)]
    
    for x, y in brick_positions:
        grid.add_brick(x, y)
    
    for x, y in target_positions:
        grid.add_target(x, y)
    
    # 创建机器人
    robot = Robot(0, 0)
    
    print(f"初始机器人位置: ({robot.x}, {robot.y})")
    print(f"初始砖块数量: {grid.brick_count}")
    
    # 手动更新机器人状态，直到所有砖块都被搬运
    steps = 0
    max_steps = 500
    
    while steps < max_steps:
        robot.update(grid.cells)
        steps += 1
        
        # 每50步打印一次状态
        if steps % 50 == 0:
            print(f"步骤 {steps}: 剩余砖块: {grid.brick_count}, 已放置: {grid.bricks_placed}")
        
        # 检查是否所有砖块都被搬运
        if grid.brick_count == 0 and grid.bricks_placed == len(brick_positions):
            print(f"所有{len(brick_positions)}个砖块都成功搬运到目标位置!")
            print(f"总共用了{steps}步")
            return True
    
    print(f"超过最大步数({max_steps})，测试失败")
    print(f"最终状态: 剩余砖块: {grid.brick_count}, 已放置: {grid.bricks_placed}")
    return False

def test_obstacle_avoidance():
    """测试机器人避障功能"""
    print("\n测试机器人避障功能...")
    
    # 创建一个有障碍物的网格
    grid = Grid()
    
    # 添加一个砖块和一个目标位置，中间有障碍物
    grid.add_brick(5, 1)
    grid.add_target(5, 5)
    
    # 添加障碍物墙
    for y in range(2, 4):
        grid.add_obstacle(4, y)
        grid.add_obstacle(5, y)
        grid.add_obstacle(6, y)
    
    # 创建机器人
    robot = Robot(1, 1)
    
    print(f"初始机器人位置: ({robot.x}, {robot.y})")
    
    # 计算到砖块的路径
    path_to_brick = robot.a_star(grid.cells, (robot.x, robot.y), (5, 1))
    
    print(f"到砖块的路径: {path_to_brick}")
    
    if path_to_brick:
        print("机器人成功规划绕过障碍物的路径!")
        
        # 测试机器人是否能找到从砖块到目标的路径
        path_to_target = robot.a_star(grid.cells, (5, 1), (5, 5))
        print(f"从砖块到目标的路径: {path_to_target}")
        
        if path_to_target:
            print("机器人成功规划从砖块到目标的路径!")
            return True
        else:
            print("机器人无法规划从砖块到目标的路径!")
            return False
    else:
        print("机器人无法规划到砖块的路径!")
        return False

def main():
    print("=" * 60)
    print("机器人搬砖模拟测试")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("机器人移动功能", test_robot_movement),
        ("机器人避障功能", test_obstacle_avoidance),
        ("砖块拾取和放置", test_brick_pickup_and_drop),
        ("多砖块搬运", test_multiple_bricks)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"测试: {test_name}")
        print(f"{'=' * 60}")
        
        try:
            result = test_func()
            if result or result is None:  # None表示测试不返回布尔值
                print(f"✓ 测试通过")
                passed += 1
            else:
                print(f"✗ 测试失败")
                failed += 1
        except Exception as e:
            print(f"✗ 测试出错: {e}")
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"测试结果: {passed} 个通过, {failed} 个失败")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    # 导入State枚举
    from carryzhan import State
    main()
