import time
import random

class UnmannedVehicleOverloadWarningSystem:
    """无人车超载预警系统

    核心功能：
    1. 实时监测无人车载重状态数据
    2. 根据载重占比自动划分预警等级
    3. 可视化输出预警信息及处理建议
    4. 检测到严重超载时触发系统紧急停止机制
    """
    def __init__(self, max_load=1000):
        """系统初始化方法，配置核心参数

        参数:
            max_load (int/float): 车辆最大载重限额，单位为千克(kg)，可根据实际车型自定义配置，默认值1000
        实例属性:
            self.max_load: 存储车辆最大载重
            self.current_load: 当前实时载重，初始值设为0
            self.warning_level: 预警等级标识，0-正常，1-轻度过载，2-中度过载，3-严重超载
        """
        self.max_load = max_load  # 车辆最大载重（kg）
        self.current_load = 0     # 当前实时载重，初始化为0
        self.warning_level = 0    # 预警等级：0-正常，1-轻度过载，2-中度过载，3-严重超载

    def simulate_weight_collection(self):
        """模拟无人车重量数据采集（传感器数据模拟接口）

        逻辑说明：
        在当前载重基础上进行随机小幅波动，模拟乘客上下车或货物装卸带来的载重变化
        避免载重出现负数，最低载重限制为0

        返回:
            float: 更新后的当前载重数值
        """
        # 模拟载重变化范围：卸载最多50kg，加载最多100kg
        weight_change = random.randint(-50, 100)
        # 更新当前载重，确保载重不低于0（最低载重为0）
        self.current_load = max(0, self.current_load + weight_change)
        return self.current_load

    def judge_overload(self):
        """超载等级判断核心方法，基于载重占比划分预警等级

        计算规则：
        载重占比 = 当前载重 / 最大载重

        返回值:
            tuple: 包含三个元素的元组，依次为：
                1. 预警状态名称（字符串）
                2. 预警颜色标识（字符串）
                3. 处理建议描述（字符串）
        """
        # 计算载重占比，作为预警等级判定的核心依据
        load_ratio = self.current_load / self.max_load

        # 正常状态：载重≤80%最大载重，无需任何干预措施
        if load_ratio <= 0.8:
            self.warning_level = 0
            return "正常", "绿色", "当前载重未超出安全范围，无需处理"
        # 轻度过载预警：80%<载重≤95%，接近上限需停止继续加载
        elif 0.8 < load_ratio <= 0.95:
            self.warning_level = 1
            return "轻度过载预警", "黄色", "当前载重接近上限，建议停止加载"
        # 中度过载警告：95%<载重≤110%，超出上限需卸载并停止运行
        elif 0.95 < load_ratio <= 1.1:
            self.warning_level = 2
            return "中度过载警告", "橙色", "当前载重已超出安全上限，立即停止运行并卸载部分货物"
        # 严重超载警报：载重>110%，极度危险需紧急制动并联系工作人员
        else:
            self.warning_level = 3
            return "严重超载警报", "红色", "极度危险！立即紧急制动，联系工作人员处理"

    def display_warning(self, status, color, desc):
        """可视化输出预警信息（模拟车载终端显示界面）

        参数:
            status (str): 预警状态名称（如"正常"、"严重超载警报"等）
            color (str): 预警颜色标识（如"绿色"、"红色"等，对应车载终端显示颜色）
            desc (str): 针对当前状态的具体处理建议描述
        """
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
        """系统主运行方法，执行持续循环监测流程

        参数:
            monitor_times (int): 预设监测总次数，默认值10次，可根据实际需求修改

        运行流程:
        1. 采集当前载重数据（模拟传感器更新）
        2. 判断超载预警等级，获取预警相关信息
        3. 可视化输出预警详情（模拟车载终端展示）
        4. 检测到严重超载时，触发紧急停止机制
        5. 间隔指定时间，进行下一次循环监测
        """
        print("无人车超载预警系统已启动...")
        print(f"开始持续监测（共监测{monitor_times}次，每次间隔2秒）\n")
        # 循环执行监测流程，满足以下任一条件即终止：
        # 1. 完成预设的监测次数
        # 2. 检测到严重超载（预警等级3）
        for i in range(monitor_times):
            # 步骤1：采集当前载重（模拟传感器数据实时更新）
            self.simulate_weight_collection()
            # 步骤2：判断超载等级，获取预警状态、颜色及处理建议
            status, color, desc = self.judge_overload()
            # 步骤3：可视化输出预警详情（模拟车载终端显示效果）
            self.display_warning(status, color, desc)
            # 步骤4：严重超载（等级3）时，立即紧急停止系统运行
            if self.warning_level == 3:
                print("❗❗❗ 检测到严重超载，系统紧急停止运行！❗")
                break
            # 步骤5：间隔2秒进行下一次监测，模拟实时周期性数据检测
            time.sleep(2)

if __name__ == "__main__":
    # 初始化系统：设置无人车最大载重为1000kg
    # 备注：可根据不同车型、应用场景修改max_load参数值
    overload_system = UnmannedVehicleOverloadWarningSystem(max_load=1000)
    # 运行系统：持续监测10次
    # 备注：可修改monitor_times参数调整监测总次数
    overload_system.run(monitor_times=10)