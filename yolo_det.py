

# 👇 新增优化参数
SKIP_FRAMES = 3  # 每 4 帧跑一次 YOLO (极大提升 FPS)
WARNING_RATIO = 0.20  # 当车宽占画面 20% 时，变红预警


# ==========================================

class LaneDetector:
    def __init__(self):
        self.left_lines_buffer = deque(maxlen=10)
        self.right_lines_buffer = deque(maxlen=10)
        self.vertices = None

    def region_of_interest(self, img):
        mask = np.zeros_like(img)
        if self.vertices is None:
            height, width = img.shape
            top_w = width * ROI_TOP_WIDTH
            top_x = width * 0.5
            self.vertices = np.array([[
                (0, height),
                (int(top_x - top_w / 2), int(height * ROI_HEIGHT_POS)),
                (int(top_x + top_w / 2), int(height * ROI_HEIGHT_POS)),
                (width, height)
            ]], dtype=np.int32)
        cv2.fillPoly(mask, self.vertices, 255)
        return cv2.bitwise_and(img, mask)

    def process_lane(self, frame):
        if frame is None: return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
        roi = self.region_of_interest(edges)
        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, HOUGH_THRESH,
                                minLineLength=MIN_LINE_LEN, maxLineGap=MAX_LINE_GAP)

        left_raw, right_raw = self.average_slope_intercept(lines)
        left_avg, right_avg = self.smooth_lines(left_raw, right_raw)

        h = frame.shape[0]
        y_min, y_max = int(h * ROI_HEIGHT_POS) + 50, h
        left_pts = self.make_points(left_avg, y_min, y_max)
        right_pts = self.make_points(right_avg, y_min, y_max)

        return self.draw_lane(frame, left_pts, right_pts)

    def average_slope_intercept(self, lines):
        lefts, rights = [], []
        if lines is None: return None, None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1: continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if abs(slope) < 0.3 or abs(slope) > 10: continue
            if slope < 0:
                lefts.append((slope, intercept))
            else:
                rights.append((slope, intercept))
        return (np.average(lefts, axis=0) if lefts else None,
                np.average(rights, axis=0) if rights else None)

    def smooth_lines(self, l, r):
        if l is not None: self.left_lines_buffer.append(l)
        if r is not None: self.right_lines_buffer.append(r)
        return (np.average(self.left_lines_buffer, axis=0) if self.left_lines_buffer else None,
                np.average(self.right_lines_buffer, axis=0) if self.right_lines_buffer else None)

    def make_points(self, line, y1, y2):
        if line is None: return None
        s, i = line
        if abs(s) < 1e-3: return None
        try:
            return ((int((y1 - i) / s), y1), (int((y2 - i) / s), y2))
        except:
            return None

    def draw_lane(self, img, l, r):
        overlay = np.zeros_like(img)
        if l and r:
            pts = np.array([l[0], l[1], r[1], r[0]], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
        if l: cv2.line(overlay, l[0], l[1], (255, 0, 0), 10)
        if r: cv2.line(overlay, r[0], r[1], (0, 0, 255), 10)
        return overlay


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "sample.hevc"
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): return

    lane_det = LaneDetector()
    yolo_det = ObjectDetector()
    print("🚀 系统优化版启动 - 跳帧检测 & 碰撞预警")

    frame_count = 0
    current_detections = []  # 缓存当前的检测结果

    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        display = frame.copy()
        h, w = frame.shape[:2]

        # --- 1. 跳帧检测逻辑 ---
        # 只有当 帧数 能被 (SKIP_FRAMES + 1) 整除时才跑 YOLO
        if frame_count % (SKIP_FRAMES + 1) == 0:
            _, current_detections = yolo_det.detect(frame)

        # --- 2. 绘制检测结果 (每一帧都画，利用缓存) ---
        for det in current_detections:
            x1, y1, x2, y2 = det['box']
            width_ratio = det['width'] / w

            # 智能预警逻辑
            color = (0, 255, 0)  # 默认绿色
            msg = ""

            if width_ratio > WARNING_RATIO:
                color = (0, 0, 255)  # 危险红色
                msg = " WARNING!"
                cv2.putText(display, "BRAKE!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, f"{det['class']}{msg}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # --- 3. 车道线叠加 ---
        lane_layer = lane_det.process_lane(frame)
        if lane_layer is not None:
            display = cv2.addWeighted(display, 1, lane_layer, 0.4, 0)

        # --- 4. 仪表盘 ---
        fps = 1.0 / (time.time() - t_start)
        # 显示当前是在检测(Detect)还是在追踪(Track/Skip)
        status = "Detecting" if frame_count % (SKIP_FRAMES + 1) == 0 else "Skipping"

        cv2.putText(display, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display, f"Mode: {status}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('AutoPilot V2.5', display)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()