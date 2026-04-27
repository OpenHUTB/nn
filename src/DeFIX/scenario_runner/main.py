def main():
    print("="*50)
    print("🚀 DeFIX 主入口")
    print("0 → 场景0：动态车辆碰撞（超车）")
    print("1 → 场景1：突发行人碰撞")
    print("="*50)

    choice = input("请输入场景编号：").strip()

    if choice == "0":
        print("正在运行：场景0 动态车辆碰撞")
        from run_scenario_0 import run
        run()
    elif choice == "1":
        print("正在运行：场景1 突发行人碰撞")
        from run_scenario_1 import run
        run()
    else:
        print("输入错误，默认运行场景0")
        from run_scenario_0 import run
        run()

if __name__ == "__main__":
    main()
