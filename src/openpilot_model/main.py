#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight lane detection visualizer for driving videos."""

from __future__ import annotations

import argparse
import csv
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

    def add_frame(self, frame: np.ndarray, frame_idx: int, fps: float, line_count: int, has_left: bool, has_right: bool) -> None:
        self.writer.write(frame)
        self.rows.append(
            {
                "frame": frame_idx,
                "fps": f"{fps:.2f}",
                "hough_lines": line_count,
                "left_lane": int(has_left),
                "right_lane": int(has_right),
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
            fieldnames = ["frame", "fps", "hough_lines", "left_lane", "right_lane"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


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


def lane_detection(frame: np.ndarray, smoother: LaneSmoother) -> tuple[np.ndarray, int, bool, bool]:
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
    return detected, line_count, left_points is not None, right_points is not None


def draw_hud(frame: np.ndarray, frame_idx: int, total_frames: int, fps: float, output_path: Path, line_count: int) -> None:
    h, w = frame.shape[:2]
    progress = frame_idx / total_frames if total_frames > 0 else 0.0
    frame_text = f"{frame_idx}/{total_frames}" if total_frames > 0 else f"{frame_idx}/?"
    progress_text = f"{progress * 100:.1f}%" if total_frames > 0 else "N/A"

    cv2.rectangle(frame, (0, 0), (w, 76), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_text}", (150, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Progress: {progress_text}", (18, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
    cv2.putText(frame, f"Hough lines: {line_count}", (220, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 220, 255), 1)
    cv2.putText(frame, output_path.name, (430, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 220, 255), 1)

    bar_w = 220
    bar_x = max(20, w - bar_w - 24)
    bar_y = 24
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (80, 80, 80), 1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 12), (0, 190, 0), -1)


def process_video(video_path: Path, output_dir: Path, display: bool, save_every: int, max_frames: int) -> tuple[Path, Path, Path | None, int]:
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_total = max_frames if max_frames > 0 else total_frames

    reporter = RunReporter(video_path, output_dir, fps, (width * 2, height), save_every)
    smoother = LaneSmoother()
    frame_idx = 0
    fps_values: list[float] = []

    print("=== Lane Detection Split Screen ===")
    print(f"Input: {video_path}")
    print(f"Size: {width}x{height} | FPS: {fps:.2f} | Total frames: {progress_total if progress_total > 0 else 'unknown'}")
    print(f"Output video: {reporter.video_path}")
    print(f"Metrics CSV: {reporter.report_path}")

    while True:
        start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        detected, line_count, has_left, has_right = lane_detection(frame, smoother)
        split_frame = np.hstack((frame, detected))
        elapsed_fps = 1.0 / max(time.perf_counter() - start, 1e-6)
        fps_values.append(elapsed_fps)

        cv2.putText(split_frame, "Original", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(split_frame, "Lane Detection", (width + 20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        frame_idx += 1
        draw_hud(split_frame, frame_idx, progress_total, elapsed_fps, reporter.video_path, line_count)
        reporter.add_frame(split_frame, frame_idx, elapsed_fps, line_count, has_left, has_right)

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

    print()
    cap.release()
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
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for video, screenshots, and CSV metrics")
    parser.add_argument("--no-display", action="store_true", help="Run without opening an OpenCV window")
    parser.add_argument("--save-every", type=int, default=0, help="Save a screenshot every N frames")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means process the whole video")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_video(args.video, args.output_dir, display=not args.no_display, save_every=args.save_every, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
