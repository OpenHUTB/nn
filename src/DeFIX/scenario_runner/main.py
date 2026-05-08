def main():
    print("="*50)
    print("🚀 DeFIX 主入口")
    print("0 → 场景0：动态车辆超车碰撞")
    print("1 → 场景1：行人礼让（前停/侧减/后不理）")
    print("2 → 场景2：前方多车拥堵自动刹车")
    print("3 → 场景3：车辆闯红灯")
    print("4 → 场景4：信号灯交叉口（红灯停车）")
    print("5 → 场景5：穿越无信号灯路口（减速观察）")
    print("="*50)

    choice = input("请输入场景编号：").strip()

    if choice == "0":
        print("正在运行：场景0")
        from run_scenario_0 import run
        run()
    elif choice == "1":
        print("正在运行：场景1")
        from run_scenario_1 import run
        run()
    elif choice == "2":
        print("正在运行：场景2")
        from run_scenario_2 import run
        run()
    elif choice == "3":
        print("正在运行：场景3")
        from run_scenario_3 import run
        run()
    elif choice == "4":
        print("正在运行：场景4")
        from run_scenario_4 import run
        run()
    elif choice == "5":
        print("正在运行：场景5")
        from run_scenario_5 import run
        run()
    else:
        print("输入错误，默认运行场景0")
        from run_scenario_0 import run
        run()

if __name__ == "__main__":
    main()
