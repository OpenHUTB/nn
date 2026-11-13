#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务3：激光建图 + A* 规划 + 航点跟踪（全程不需要图形界面）。

流程:
    1. 用 /uav/scan 在 2D 栅格上累积占据/空闲（ray casting）
    2. 收到目标点后跑 A*（障碍膨胀 2 格）
    3. 用纯追踪（pure pursuit）沿路径飞，到达后停

输出（便于离线画图）:
    ~/nav_out/grid.csv    每个格子: x,y,occ(0/1)
    ~/nav_out/path.csv    规划路径点
    ~/nav_out/track.csv   实际飞行轨迹
"""

import csv
import heapq
import math
import os

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

RES = 0.1
W = 100            # 10m x 10m
H = 100
OX, OY = -5.0, -5.0


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class GridNav(object):
    def __init__(self):
        self.scan_topic = rospy.get_param("~scan_topic", "/uav/scan")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")
        goal = rospy.get_param("~goal", [2.5, -2.5])
        if isinstance(goal, str):
            goal = [float(v) for v in goal.strip("[] ").split(",")]
        self.goal = (float(goal[0]), float(goal[1]))
        self.out_dir = rospy.get_param("~out_dir", os.path.expanduser("~/nav_out"))
        self.map_seconds = rospy.get_param("~map_seconds", 12.0)
        self.v = rospy.get_param("~v", 0.7)

        self.occ = [[0] * W for _ in range(H)]
        self.free = [[0] * W for _ in range(H)]
        self.scan = None
        self.odom = None
        self.path = []
        self.track = []
        self.state = "mapping"
        self.t_map = None
        self.t0 = None

        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

        rospy.Subscriber(self.scan_topic, LaserScan, self.on_scan, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.on_odom, queue_size=1)
        self.pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        rospy.loginfo("网格导航启动，目标 %s", self.goal)

    # ---------------- 地图 ----------------
    def cell(self, x, y):
        i = int((x - OX) / RES)
        j = int((y - OY) / RES)
        if 0 <= i < W and 0 <= j < H:
            return i, j
        return None

    def on_odom(self, msg):
        self.odom = msg
        p = msg.pose.pose.position
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        if self.t0 is None:
            self.t0 = msg.header.stamp
        self.track.append([(msg.header.stamp - self.t0).to_sec(), p.x, p.y, yaw])
        self.pos = (p.x, p.y, yaw)

    def on_scan(self, msg):
        self.scan = msg

    def map_scan(self):
        if self.scan is None or not hasattr(self, "pos"):
            return
        px, py, yaw = self.pos
        s = self.scan
        for i in range(0, len(s.ranges), 2):
            r = s.ranges[i]
            a = yaw + s.angle_min + i * s.angle_increment
            if not (r == r) or r <= 0.05 or r > 11.0:
                continue
            steps = int(r / RES)
            for k in range(steps):
                x = px + math.cos(a) * k * RES
                y = py + math.sin(a) * k * RES
                c = self.cell(x, y)
                if c:
                    self.free[c[1]][c[0]] = 1
            ex = px + math.cos(a) * r
            ey = py + math.sin(a) * r
            c = self.cell(ex, ey)
            if c:
                self.occ[c[1]][c[0]] += 1

    # ---------------- A* ----------------
    def blocked(self, i, j):
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < W and 0 <= nj < H and self.occ[nj][ni] > 0:
                    return True
        return False

    def astar(self, start, goal):
        s = self.cell(*start)
        g = self.cell(*goal)
        if not s or not g:
            return []
        openq = [(0.0, s)]
        came = {}
        cost = {s: 0.0}
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == g:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return path[::-1]
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                ni, nj = cur[0] + di, cur[1] + dj
                if not (0 <= ni < W and 0 <= nj < H) or self.blocked(ni, nj):
                    continue
                nc = cost[cur] + math.hypot(di, dj)
                if nc < cost.get((ni, nj), 1e9):
                    cost[(ni, nj)] = nc
                    came[(ni, nj)] = cur
                    h = math.hypot(g[0] - ni, g[1] - nj)
                    heapq.heappush(openq, (nc + h, (ni, nj)))
        return []

    def world(self, c):
        return OX + (c[0] + 0.5) * RES, OY + (c[1] + 0.5) * RES

    # ---------------- 控制 ----------------
    def follow(self, cmd):
        if not self.path:
            return False
        px, py, yaw = self.pos
        tx, ty = self.path[0]
        d = math.hypot(tx - px, ty - py)
        if d < 0.35:
            self.path.pop(0)
            return bool(self.path)
        err = math.atan2(ty - py, tx - px) - yaw
        while err > math.pi:
            err -= 2 * math.pi
        while err < -math.pi:
            err += 2 * math.pi
        cmd.angular.z = max(-1.0, min(1.0, 1.6 * err))
        cmd.linear.x = self.v * max(0.0, 1.0 - abs(err) / 1.6)
        return True

    def run(self):
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            cmd = Twist()
            if self.state == "mapping":
                self.map_scan()
                if self.pos_ready() and (rospy.Time.now() - self.t_start).to_sec() > self.map_seconds:
                    self.plan()
                    self.state = "driving"
            elif self.state == "driving":
                if not self.follow(cmd):
                    self.state = "done"
                    self.save()
                    rospy.loginfo("到达目标，已保存数据")
            self.pub.publish(cmd)
            r.sleep()

    def pos_ready(self):
        return hasattr(self, "pos")

    def plan(self):
        start = (self.pos[0], self.pos[1])
        cells = self.astar(start, self.goal)
        self.path = [self.world(c) for c in cells]
        rospy.loginfo("A* 规划完成: %d 个航点", len(self.path))
        with open(os.path.join(self.out_dir, "path.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "y"])
            w.writerows(self.path)

    def save(self):
        with open(os.path.join(self.out_dir, "grid.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["i", "j", "x", "y", "occ"])
            for j in range(H):
                for i in range(W):
                    if self.occ[j][i] > 0:
                        x, y = self.world((i, j))
                        w.writerow([i, j, round(x, 3), round(y, 3), 1])
        with open(os.path.join(self.out_dir, "track.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "x", "y", "yaw"])
            w.writerows(self.track)
        rospy.loginfo("已保存 grid / path / track 到 %s", self.out_dir)


def main():
    rospy.init_node("uav_grid_nav")
    node = GridNav()
    node.t_start = rospy.Time.now()
    rospy.on_shutdown(node.save)
    node.run()


if __name__ == "__main__":
    main()
