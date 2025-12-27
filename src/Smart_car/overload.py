import time
import random

class UnmannedVehicleOverloadWarningSystem:
    """无人车超载预警系统"""
    def __init__(self, max_load=1000):
        # 初始化系统参数
        self.max_load = max_load  # 车辆最大载重（单位：kg，可自定义）
        self.current_load = 0     # 当前载重
        self.warning_level = 0    # 预警等级：0-正常，1-轻度过载，2-中度过载，3-严重超载

    def simulate_weight_collection(self):
        """模拟无人车重量采集（模拟传感器数据，可替换为真实硬件接口）"""
        # 模拟载重变化：每次在当前基础上小幅波动（模拟乘客/货物上下车）
        weight_change = random.randint(-50, 100)
        self.current_load = max(0, self.current_load + weight_change)  # 载重不能为负数
        return self.current_load

    def judge_overload(self):
        """超载判断逻辑，划分预警等级"""
        load_ratio = self.current_load / self.max_load  # 载重占比

        if load_ratio <= 0.8:
            self.warning_level = 0  # 正常：载重≤80%最大载重
            return "正常", "绿色", "当前载重未超出安全范围，无需处理"
        elif 0.8 < load_ratio <= 0.95:
            self.warning_level = 1  # 轻度过载：80%<载重≤95%
            return "轻度过载预警", "黄色", "当前载重接近上限，建议停止加载"
        elif 0.95 < load_ratio <= 1.1:
            self.warning_level = 2  # 中度过载：95%<载重≤110%
            return "中度过载警告", "橙色", "当前载重已超出安全上限，立即停止运行并卸载部分货物"
        else:
            self.warning_level = 3  # 严重超载：载重>110%
            return "严重超载警报", "红色", "极度危险！立即紧急制动，联系工作人员处理"

    def display_warning(self, status, color, desc):
        """可视化输出预警信息（模拟车载终端显示）"""
        print("=" * 50)
        print(f"【无人车超载预警系统 - 实时监测】")
        print(f"当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"最大载重：{self.max_load} kg")
        print(f"当前载重：{self.current_load:.2f} kg")
        print(f"载重占比：{self.current_load/self.max_load*100:.2f}%")
        print(f"预警状态：【{status}】（{color}）")
        print(f"处理建议：{desc}")
        print("=" * 50)
        print()

    def run(self, monitor_times=10):
        """运行系统，持续监测载重并输出预警"""
        print("无人车超载预警系统已启动...")
        print(f"开始持续监测（共监测{monitor_times}次，每次间隔2秒）\n")

        for i in range(monitor_times):
            # 1. 采集当前载重（模拟）
            self.simulate_weight_collection()
            # 2. 判断超载等级
            status, color, desc = self.judge_overload()
            # 3. 显示预警信息
            self.display_warning(status, color, desc)
            # 4. 严重超载时紧急停止
            if self.warning_level == 3:
                print("❗❗❗ 检测到严重超载，系统紧急停止运行！ ❗❗")
                break
            # 5. 间隔2秒进行下一次监测
            time.sleep(2)

if __name__ == "__main__":
    # 初始化系统：设置无人车最大载重为1000kg（可根据需求修改）
    overload_system = UnmannedVehicleOverloadWarningSystem(max_load=1000)
    # 运行系统，持续监测10次（可修改监测次数）
    overload_system.run(monitor_times=10)