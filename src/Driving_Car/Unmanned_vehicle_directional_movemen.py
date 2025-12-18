import tkinter as tk

class UnmannedCarSimulator:
    def __init__(self, root):
        # 初始化主窗口
        self.root = root
        self.root.title("无人车平地移动模拟")
        self.root.resizable(False, False)

        # 无人车参数
        self.car_x = 250  # 初始x坐标
        self.car_y = 250  # 初始y坐标
        self.car_size = 30  # 车的大小（正方形边长）
        self.move_step = 10  # 每次移动的步长

        # 创建画布（模拟平地）
        self.canvas = tk.Canvas(root, width=500, height=500, bg="white")
        self.canvas.pack()

        # 绘制初始无人车（矩形）
        self.car = self.canvas.create_rectangle(
            self.car_x - self.car_size/2, self.car_y - self.car_size/2,
            self.car_x + self.car_size/2, self.car_y + self.car_size/2,
            fill="blue"
        )

        # 绑定方向键事件
        self.root.bind("<Up>", self.move_up)
        self.root.bind("<Down>", self.move_down)
        self.root.bind("<Left>", self.move_left)
        self.root.bind("<Right>", self.move_right)

        # 聚焦窗口以接收键盘事件
        self.canvas.focus_set()

    def move_up(self, event):
        """向上移动（y坐标减小）"""
        new_y = self.car_y - self.move_step
        # 边界检测：不超出画布上沿
        if new_y - self.car_size/2 >= 0:
            self.car_y = new_y
            self.canvas.move(self.car, 0, -self.move_step)

    def move_down(self, event):
        """向下移动（y坐标增大）"""
        new_y = self.car_y + self.move_step
        # 边界检测：不超出画布下沿
        if new_y + self.car_size/2 <= 500:
            self.car_y = new_y
            self.canvas.move(self.car, 0, self.move_step)

    def move_left(self, event):
        """向左移动（x坐标减小）"""
        new_x = self.car_x - self.move_step
        # 边界检测：不超出画布左沿
        if new_x - self.car_size/2 >= 0:
            self.car_x = new_x
            self.canvas.move(self.car, -self.move_step, 0)

    def move_right(self, event):
        """向右移动（x坐标增大）"""
        new_x = self.car_x + self.move_step
        # 边界检测：不超出画布右沿
        if new_x + self.car_size/2 <= 500:
            self.car_x = new_x
            self.canvas.move(self.car, self.move_step, 0)

if __name__ == "__main__":
    root = tk.Tk()
    simulator = UnmannedCarSimulator(root)
    root.mainloop()