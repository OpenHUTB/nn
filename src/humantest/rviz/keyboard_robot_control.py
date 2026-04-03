import keyboard
import time

class Robot:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.speed = 1

    def move_forward(self):
        self.y -= self.speed
        print(f"Robot moved forward. Position: ({self.x}, {self.y})")

    def move_backward(self):
        self.y += self.speed
        print(f"Robot moved backward. Position: ({self.x}, {self.y})")

    def move_left(self):
        self.x -= self.speed
        print(f"Robot moved left. Position: ({self.x}, {self.y})")

    def move_right(self):
        self.x += self.speed
        print(f"Robot moved right. Position: ({self.x}, {self.y})")

def main():
    robot = Robot()
    print("Press ESC to stop.")
    print("Use arrow keys to control the robot:")
    print("Up: Move forward")
    print("Down: Move backward")
    print("Left: Turn left")
    print("Right: Turn right")

    keyboard.add_hotkey('up', robot.move_forward)
    keyboard.add_hotkey('down', robot.move_backward)
    keyboard.add_hotkey('left', robot.move_left)
    keyboard.add_hotkey('right', robot.move_right)

    keyboard.wait('esc')

if __name__ == "__main__":
    main()