
import cv2
import numpy as np
import os


def create_test_left_turn_image():
    """创建左转测试图片"""
    # 创建空白图像
    img = np.ones((600, 800, 3), dtype=np.uint8) * 120

    # 天空（蓝色）
    sky_color = (135, 206, 235)
    cv2.rectangle(img, (0, 0), (800, 300), sky_color, -1)

    # 草地（绿色）
    grass_color = (34, 139, 34)
    cv2.rectangle(img, (0, 300), (800, 600), grass_color, -1)

    # 绘制左转道路（从底部向右上方弯曲，然后向左）
    road_color = (50, 50, 50)

    # 使用贝塞尔曲线绘制左转道路
    points = []
    for t in np.linspace(0, 1, 50):
        # 左转曲线：先向右再向左
        x = int(400 + 200 * t - 150 * t ** 2)
        y = int(580 - 350 * t)
        points.append((x, y))

    points = np.array(points, dtype=np.int32)

    # 绘制道路（宽度变化）
    for i in range(len(points) - 1):
        width = int(60 + 20 * i / len(points))
        cv2.line(img,
                 (points[i][0] - width, points[i][1]),
                 (points[i][0] + width, points[i][1]),
                 road_color, 1)

    # 绘制车道线（白色虚线）
    line_color = (255, 255, 255)
    for i in range(0, len(points), 4):
        cv2.line(img,
                 (points[i][0] - 2, points[i][1]),
                 (points[i][0] + 2, points[i][1]),
                 line_color, 2)

    # 绘制道路边缘线
    for side in [-1, 1]:
        edge_points = []
        for i, pt in enumerate(points):
            edge_x = pt[0] + side * (40 + 10 * i / len(points))
            edge_points.append((int(edge_x), pt[1]))
        edge_points = np.array(edge_points, dtype=np.int32)
        cv2.polylines(img, [edge_points], False, (255, 255, 255), 2)

    # 添加一些护栏（左侧）
    for i in range(0, len(points), 3):
        if i < len(points) - 1:
            pt = points[i]
            guard_x = pt[0] - 50
            cv2.line(img, (guard_x, pt[1]), (guard_x, pt[1] + 10), (100, 100, 100), 2)

    # 保存图像
    output_path = os.path.join(os.path.dirname(__file__), 'test_left_turn.png')
    cv2.imwrite(output_path, img)
    print(f"已保存左转测试图片: {output_path}")
    return output_path


def create_test_right_turn_image():
    """创建右转测试图片"""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 120

    # 天空
    sky_color = (135, 206, 235)
    cv2.rectangle(img, (0, 0), (800, 300), sky_color, -1)

    # 草地
    grass_color = (34, 139, 34)
    cv2.rectangle(img, (0, 300), (800, 600), grass_color, -1)

    # 绘制右转道路（从底部向左上方弯曲，然后向右）
    road_color = (50, 50, 50)

    points = []
    for t in np.linspace(0, 1, 50):
        # 右转曲线：先向左再向右
        x = int(400 - 200 * t + 150 * t ** 2)
        y = int(580 - 350 * t)
        points.append((x, y))

    points = np.array(points, dtype=np.int32)

    # 绘制道路
    for i in range(len(points) - 1):
        width = int(60 + 20 * i / len(points))
        cv2.line(img,
                 (points[i][0] - width, points[i][1]),
                 (points[i][0] + width, points[i][1]),
                 road_color, 1)

    # 车道线
    line_color = (255, 255, 255)
    for i in range(0, len(points), 4):
        cv2.line(img,
                 (points[i][0] - 2, points[i][1]),
                 (points[i][0] + 2, points[i][1]),
                 line_color, 2)

    # 边缘线
    for side in [-1, 1]:
        edge_points = []
        for i, pt in enumerate(points):
            edge_x = pt[0] + side * (40 + 10 * i / len(points))
            edge_points.append((int(edge_x), pt[1]))
        edge_points = np.array(edge_points, dtype=np.int32)
        cv2.polylines(img, [edge_points], False, (255, 255, 255), 2)

    # 保存图像
    output_path = os.path.join(os.path.dirname(__file__), 'test_right_turn.png')
    cv2.imwrite(output_path, img)
    print(f"已保存右转测试图片: {output_path}")
    return output_path


if __name__ == "__main__":
    create_test_left_turn_image()
    create_test_right_turn_image()
    print("测试图片已生成！请在主程序中测试这两个图片。")
