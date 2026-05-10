#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight lane detection visualizer for driving videos."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output"

CANNY_LOW = 40
CANNY_HIGH = 120
WHITE_LOWER = np.array([0, 120, 0], dtype=np.uint8)
WHITE_UPPER = np.array([180, 255, 170], dtype=np.uint8)
YELLOW_LOWER = np.array([15, 70, 70], dtype=np.uint8)
YELLOW_UPPER = np.array([40, 255, 255], dtype=np.uint8)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class LaneResult:
    frame: np.ndarray
    line_count: int
    has_left: bool
    has_right: bool
    confidence: float
    deviation: float
    departure_status: str


@dataclass
class LaneSmoother:
    """Keep fitted lane lines stable across frames."""

    history_len: int = 8
    left_history: list[np.ndarray] = field(default_factory=list)
    right_history: list[np.ndarray] = field(default_factory=list)

    def update(self, left_fit: np.ndarray | None, right_fit: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
        if left_fit is not None:
            self.left_history.append(left_fit)
            self.left_history = self.left_history[-self.history_len :]
        if right_fit is not None:
            self.right_history.append(right_fit)
            self.right_history = self.right_history[-self.history_len :]

        left = np.mean(self.left_history, axis=0) if self.left_history else None
        right = np.mean(self.right_history, axis=0) if self.right_history else None
        return left, right


class RunReporter:
    """Write output video, screenshots, and per-frame metrics."""

    def __init__(self, source: Path, output_dir: Path, fps: float, frame_size: tuple[int, int], save_every: int):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir = output_dir / "screenshots"
        self.screenshot_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.video_path = output_dir / f"{source.stem}_lane_detection_{timestamp}.mp4"
        self.report_path = output_dir / f"{source.stem}_lane_metrics_{timestamp}.csv"
        self.save_every = max(0, int(save_every))
        self.first_screenshot: Path | None = None
        self.rows: list[dict[str, str | int]] = []
        self.writer = cv2.VideoWriter(
            str(self.video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(float(fps), 1.0),
            frame_size,
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot create output video writer: {self.video_path}")

    def add_frame(self, frame: np.ndarray, frame_idx: int, fps: float, result: LaneResult) -> None:
        self.writer.write(frame)
        self.rows.append(
            {
                "frame": frame_idx,
                "fps": f"{fps:.2f}",
                "hough_lines": result.line_count,
                "left_lane": int(result.has_left),
                "right_lane": int(result.has_right),
                "confidence": f"{result.confidence:.3f}",
                "lane_deviation": f"{result.deviation:.4f}",
                "departure_status": result.departure_status,
            }
        )

        if self.first_screenshot is None:
            self.save_screenshot(frame, frame_idx, "preview")
        if self.save_every and frame_idx % self.save_every == 0:
            self.save_screenshot(frame, frame_idx, "auto")

    def save_screenshot(self, frame: np.ndarray, frame_idx: int, suffix: str = "manual") -> Path:
        path = self.screenshot_dir / f"frame_{frame_idx:06d}_{suffix}.jpg"
        cv2.imwrite(str(path), frame)
        if self.first_screenshot is None:
            self.first_screenshot = path
        return path

    def close(self) -> None:
        self.writer.release()
        with open(self.report_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "frame",
                "fps",
                "hough_lines",
                "left_lane",
                "right_lane",
                "confidence",
                "lane_deviation",
                "departure_status",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


class FrameSource:
    def __init__(self, source: Path, carla_test: bool):
        self.source = source
        self.carla_test = carla_test
        self.cap: cv2.VideoCapture | None = None
        self.images: list[Path] = []
        self.image_index = 0
        self.width = 0
        self.height = 0
        self.fps = 20.0 if carla_test else 30.0
        self.total_frames = 0

        if source.is_dir():
            self.images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
            if not self.images:
                raise RuntimeError(f"No image frames found in directory: {source}")
            first = cv2.imread(str(self.images[0]))
            if first is None:
                raise RuntimeError(f"Cannot read first image frame: {self.images[0]}")
            self.height, self.width = first.shape[:2]
            self.total_frames = len(self.images)
        else:
            self.cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video: {source}")
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or self.fps
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.width <= 0 or self.height <= 0:
            raise RuntimeError(f"Invalid input frame size: {self.width}x{self.height}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.images:
            if self.image_index >= len(self.images):
                return False, None
            path = self.images[self.image_index]
            self.image_index += 1
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"Warning: skip unreadable image frame: {path}")
                return self.read()
            return True, frame

        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()

    @property
    def mode_label(self) -> str:
        if self.carla_test:
            return "CARLA TEST"
        return "VIDEO"


def color_filter(frame: np.ndarray) -> np.ndarray:
    """Prefer white and yellow lane markings in HLS color space."""
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white_mask = cv2.inRange(hls, WHITE_LOWER, WHITE_UPPER)
    yellow_mask = cv2.inRange(hls, YELLOW_LOWER, YELLOW_UPPER)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.dilate(mask, kernel, iterations=1)


def trapezoid_roi(width: int, height: int) -> np.ndarray:
    top_y = int(height * 0.58)
    top_half_width = int(width * 0.12)
    bottom_margin = int(width * 0.04)
    center_x = width // 2
    return np.array(
        [
            [
                (bottom_margin, height),
                (center_x - top_half_width, top_y),
                (center_x + top_half_width, top_y),
                (width - bottom_margin, height),
            ]
        ],
        dtype=np.int32,
    )


def fit_lane_lines(lines: np.ndarray | None, width: int, height: int) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    if lines is None:
        return None, None, 0

    left_lines: list[tuple[float, float]] = []
    right_lines: list[tuple[float, float]] = []
    left_weights: list[float] = []
    right_weights: list[float] = []
    mid_x = width / 2
    y_top = int(height * 0.62)
    y_bottom = height

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue

        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.35 or abs(slope) > 3.5:
            continue

        intercept = y1 - slope * x1
        x_at_top = (y_top - intercept) / slope
        x_at_bottom = (y_bottom - intercept) / slope
        weight = float(np.hypot(x2 - x1, y2 - y1))

        if slope < 0 and x_at_top < mid_x and x_at_bottom < width * 0.75:
            left_lines.append((slope, intercept))
            left_weights.append(weight)
        elif slope > 0 and x_at_top > mid_x and x_at_bottom > width * 0.55:
            right_lines.append((slope, intercept))
            right_weights.append(weight)

    left = np.average(left_lines, axis=0, weights=left_weights) if left_lines else None
    right = np.average(right_lines, axis=0, weights=right_weights) if right_lines else None
    return left, right, len(lines)


def line_points(fit: np.ndarray | None, y_top: int, y_bottom: int, width: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if fit is None:
        return None
    slope, intercept = fit
    if abs(slope) < 1e-3:
        return None

    x_top = int((y_top - intercept) / slope)
    x_bottom = int((y_bottom - intercept) / slope)
    x_top = max(-width, min(width * 2, x_top))
    x_bottom = max(-width, min(width * 2, x_bottom))
    return (x_top, y_top), (x_bottom, y_bottom)


def departure_status(deviation: float, confidence: float) -> str:
    if confidence < 0.45:
        return "LOW_CONF"
    if deviation <= -0.08:
        return "LEFT_DEPARTURE"
    if deviation >= 0.08:
        return "RIGHT_DEPARTURE"
    if abs(deviation) >= 0.05:
        return "CAUTION"
    return "CENTERED"


def lane_confidence(line_count: int, left_points, right_points, width: int) -> tuple[float, float, str]:
    has_left = left_points is not None
    has_right = right_points is not None
    detection_score = 0.25 * int(has_left) + 0.25 * int(has_right)
    line_score = min(line_count / 24.0, 1.0) * 0.25

    deviation = 0.0
    geometry_score = 0.0
    if has_left and has_right:
        left_bottom = left_points[1][0]
        right_bottom = right_points[1][0]
        lane_width = max(1, right_bottom - left_bottom)
        lane_center = (left_bottom + right_bottom) / 2.0
        deviation = (lane_center - width / 2.0) / width
        width_ratio = lane_width / width
        geometry_score = 0.25 * max(0.0, 1.0 - abs(width_ratio - 0.55) / 0.45)
    elif has_left or has_right:
        deviation = -0.12 if has_left else 0.12
        geometry_score = 0.08

    confidence = max(0.0, min(1.0, detection_score + line_score + geometry_score))
    return confidence, deviation, departure_status(deviation, confidence)


def lane_detection(frame: np.ndarray, smoother: LaneSmoother) -> LaneResult:
    h, w = frame.shape[:2]
    mask = color_filter(frame)
    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    roi_vertices = trapezoid_roi(w, h)
    roi_mask = np.zeros_like(edges)
    cv2.fillPoly(roi_mask, roi_vertices, 255)
    roi_edges = cv2.bitwise_and(edges, roi_mask)

    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=18, minLineLength=25, maxLineGap=120)
    left_fit, right_fit, line_count = fit_lane_lines(lines, w, h)
    left_fit, right_fit = smoother.update(left_fit, right_fit)

    detected = frame.copy()
    overlay = np.zeros_like(frame)
    y_top = int(h * 0.62)
    y_bottom = h
    left_points = line_points(left_fit, y_top, y_bottom, w)
    right_points = line_points(right_fit, y_top, y_bottom, w)

    cv2.polylines(detected, roi_vertices, True, (0, 255, 255), 2)
    if left_points is not None:
        cv2.line(overlay, left_points[0], left_points[1], (0, 0, 255), 10)
    if right_points is not None:
        cv2.line(overlay, right_points[0], right_points[1], (255, 0, 0), 10)
    if left_points is not None and right_points is not None:
        lane_area = np.array([left_points[1], left_points[0], right_points[0], right_points[1]], dtype=np.int32)
        cv2.fillPoly(overlay, [lane_area], (0, 180, 0))

    detected = cv2.addWeighted(detected, 1.0, overlay, 0.35, 0)
    confidence, deviation, status = lane_confidence(line_count, left_points, right_points, w)
    return LaneResult(
        frame=detected,
        line_count=line_count,
        has_left=left_points is not None,
        has_right=right_points is not None,
        confidence=confidence,
        deviation=deviation,
        departure_status=status,
    )


def draw_hud(frame: np.ndarray, frame_idx: int, total_frames: int, fps: float, output_path: Path, result: LaneResult, mode_label: str) -> None:
    h, w = frame.shape[:2]
    progress = frame_idx / total_frames if total_frames > 0 else 0.0
    frame_text = f"{frame_idx}/{total_frames}" if total_frames > 0 else f"{frame_idx}/?"
    progress_text = f"{progress * 100:.1f}%" if total_frames > 0 else "N/A"

    cv2.rectangle(frame, (0, 0), (w, 76), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_text}", (150, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Progress: {progress_text}", (18, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
    cv2.putText(frame, f"Hough lines: {result.line_count}", (220, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 220, 255), 1)
    cv2.putText(frame, f"{mode_label} | {output_path.name}", (430, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 220, 255), 1)

    status_color = (0, 255, 0)
    if result.departure_status in {"LEFT_DEPARTURE", "RIGHT_DEPARTURE"}:
        status_color = (0, 0, 255)
    elif result.departure_status in {"CAUTION", "LOW_CONF"}:
        status_color = (0, 215, 255)

    cv2.rectangle(frame, (0, h - 72), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"Confidence: {result.confidence:.2f}", (18, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2)
    cv2.putText(frame, f"Lane offset: {result.deviation:+.3f}", (260, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {result.departure_status}", (520, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.68, status_color, 2)
    cv2.line(frame, (w // 2, h - 12), (w // 2, h - 32), (180, 180, 180), 2)
    marker_x = int(w // 2 + result.deviation * w)
    marker_x = max(20, min(w - 20, marker_x))
    cv2.circle(frame, (marker_x, h - 22), 8, status_color, -1)

    bar_w = 220
    bar_x = max(20, w - bar_w - 24)
    bar_y = 24
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (80, 80, 80), 1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 12), (0, 190, 0), -1)


def process_video(video_path: Path, output_dir: Path, display: bool, save_every: int, max_frames: int, carla_test: bool) -> tuple[Path, Path, Path | None, int]:
    source = FrameSource(video_path, carla_test=carla_test)

    width = source.width
    height = source.height
    fps = source.fps
    total_frames = source.total_frames
    progress_total = max_frames if max_frames > 0 else total_frames

    reporter = RunReporter(video_path, output_dir, fps, (width * 2, height), save_every)
    smoother = LaneSmoother()
    frame_idx = 0
    fps_values: list[float] = []

    print("=== Lane Detection Split Screen ===")
    print(f"Mode: {source.mode_label}")
    print(f"Input: {video_path}")
    print(f"Size: {width}x{height} | FPS: {fps:.2f} | Total frames: {progress_total if progress_total > 0 else 'unknown'}")
    print(f"Output video: {reporter.video_path}")
    print(f"Metrics CSV: {reporter.report_path}")

    try:
        while True:
            start = time.perf_counter()
            ret, frame = source.read()
            if not ret:
                break

            try:
                result = lane_detection(frame, smoother)
                split_frame = np.hstack((frame, result.frame))
            except Exception as exc:
                print(f"Warning: skip frame {frame_idx + 1} because lane detection failed: {exc}")
                continue

            elapsed_fps = 1.0 / max(time.perf_counter() - start, 1e-6)
            fps_values.append(elapsed_fps)

            cv2.putText(split_frame, "Original", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(split_frame, "Lane Detection", (width + 20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            frame_idx += 1
            draw_hud(split_frame, frame_idx, progress_total, elapsed_fps, reporter.video_path, result, source.mode_label)
            reporter.add_frame(split_frame, frame_idx, elapsed_fps, result)

            if progress_total > 0 and frame_idx % max(1, int(fps)) == 0:
                print(f"\rProcessed {frame_idx}/{progress_total} frames ({frame_idx / progress_total * 100:.1f}%)", end="")

            if display:
                cv2.imshow("Lane Detection - Split Screen", split_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    reporter.save_screenshot(split_frame, frame_idx)

            if max_frames > 0 and frame_idx >= max_frames:
                break
    finally:
        print()
        source.release()
        reporter.close()
        if display:
            cv2.destroyAllWindows()

    avg_fps = float(np.mean(fps_values)) if fps_values else 0.0
    print(f"Done. Processed frames: {frame_idx}")
    print(f"Average processing FPS: {avg_fps:.1f}")
    print(f"Saved video: {reporter.video_path}")
    print(f"Saved metrics: {reporter.report_path}")
    if reporter.first_screenshot is not None:
        print(f"Saved screenshot: {reporter.first_screenshot}")

    return reporter.video_path, reporter.report_path, reporter.first_screenshot, frame_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect lane lines in a driving video.")
    parser.add_argument("video", type=Path, help="Input video path, or CARLA RGB image directory when --carla-test is enabled")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for video, screenshots, and CSV metrics")
    parser.add_argument("--no-display", action="store_true", help="Run without opening an OpenCV window")
    parser.add_argument("--save-every", type=int, default=0, help="Save a screenshot every N frames")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means process the whole video")
    parser.add_argument("--carla-test", action="store_true", help="Enable CARLA test mode for videos or RGB image sequences")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        process_video(
            args.video,
            args.output_dir,
            display=not args.no_display,
            save_every=args.save_every,
            max_frames=args.max_frames,
            carla_test=args.carla_test,
        )
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
