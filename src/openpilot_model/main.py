#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lane detection demo with stable Hough lane fitting."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
WHITE_LOWER = np.array([0, 120, 0], dtype=np.uint8)
WHITE_UPPER = np.array([180, 255, 170], dtype=np.uint8)
YELLOW_LOWER = np.array([15, 70, 70], dtype=np.uint8)
YELLOW_UPPER = np.array([40, 255, 255], dtype=np.uint8)


@dataclass
class LaneSmoother:
    max_history: int = 8
    left_history: list[np.ndarray] = field(default_factory=list)
    right_history: list[np.ndarray] = field(default_factory=list)

    def update(self, left_fit: np.ndarray | None, right_fit: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
        if left_fit is not None:
            self.left_history.append(left_fit)
            self.left_history = self.left_history[-self.max_history :]
        if right_fit is not None:
            self.right_history.append(right_fit)
            self.right_history = self.right_history[-self.max_history :]

        left = np.mean(self.left_history, axis=0) if self.left_history else None
        right = np.mean(self.right_history, axis=0) if self.right_history else None
        return left, right


def build_trapezoid_roi(width: int, height: int) -> np.ndarray:
    """Return a road-focused trapezoid region of interest."""
    top_y = int(height * 0.58)
    bottom_y = height
    top_half_width = int(width * 0.11)
    bottom_margin = int(width * 0.05)
    center_x = width // 2

    return np.array(
        [
            [
                (bottom_margin, bottom_y),
                (center_x - top_half_width, top_y),
                (center_x + top_half_width, top_y),
                (width - bottom_margin, bottom_y),
            ]
        ],
        dtype=np.int32,
    )


def color_filter(frame: np.ndarray) -> np.ndarray:
    """Prefer white and yellow lane markings in HLS color space."""
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white_mask = cv2.inRange(hls, WHITE_LOWER, WHITE_UPPER)
    yellow_mask = cv2.inRange(hls, YELLOW_LOWER, YELLOW_UPPER)
    mask = cv2.bitwise_or(white_mask, yellow_mask)

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.dilate(mask, kernel, iterations=1)


def weighted_lane_fit(lines: Iterable[np.ndarray] | None, width: int, height: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fit scattered Hough segments into one left and one right lane line."""
    if lines is None:
        return None, None

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
        length = float(np.hypot(x2 - x1, y2 - y1))
        x_at_top = (y_top - intercept) / slope
        x_at_bottom = (y_bottom - intercept) / slope

        if slope < 0 and x_at_top < mid_x and x_at_bottom < width * 0.72:
            left_lines.append((slope, intercept))
            left_weights.append(length)
        elif slope > 0 and x_at_top > mid_x and x_at_bottom > width * 0.55:
            right_lines.append((slope, intercept))
            right_weights.append(length)

    left_fit = np.average(left_lines, axis=0, weights=left_weights) if left_lines else None
    right_fit = np.average(right_lines, axis=0, weights=right_weights) if right_lines else None
    return left_fit, right_fit


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


def lane_detection(frame: np.ndarray, smoother: LaneSmoother) -> np.ndarray:
    h, w = frame.shape[:2]
    color_mask = color_filter(frame)
    blur = cv2.GaussianBlur(color_mask, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)

    roi_vertices = build_trapezoid_roi(w, h)
    roi_mask = np.zeros_like(edges)
    cv2.fillPoly(roi_mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, roi_mask)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=18,
        minLineLength=25,
        maxLineGap=120,
    )

    left_fit, right_fit = weighted_lane_fit(lines, w, h)
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

    return cv2.addWeighted(detected, 1.0, overlay, 0.35, 0)


def draw_hud(frame: np.ndarray, frame_idx: int, total_frames: int, fps: float, output_path: Path) -> None:
    h, w = frame.shape[:2]
    progress = (frame_idx / total_frames) if total_frames > 0 else 0.0
    progress_text = f"{progress * 100:5.1f}%" if total_frames > 0 else "  N/A"

    cv2.rectangle(frame, (0, 0), (w, 74), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:5.1f}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_idx}/{total_frames if total_frames > 0 else '?'}", (180, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(frame, f"Progress: {progress_text}", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2)
    cv2.putText(frame, f"Output: {output_path.name}", (330, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (180, 220, 255), 1)

    bar_x, bar_y, bar_w, bar_h = w - 260, 24, 220, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (0, 200, 0), -1)


def process_video(video_path: Path, output_dir: Path, display: bool, save_every: int, max_frames: int) -> tuple[Path, Path | None, int]:
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_total = max_frames if max_frames > 0 else total_frames
    output_path = output_dir / f"{video_path.stem}_lane_detection.mp4"
    screenshot_dir = output_dir / "screenshots"
    screenshot_dir.mkdir(exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width * 2, height),
    )

    smoother = LaneSmoother()
    frame_idx = 0
    last_time = time.perf_counter()
    first_screenshot: Path | None = None

    print("=== Lane Detection Split Screen ===")
    print(f"Input: {video_path}")
    print(f"Size: {width}x{height} | FPS: {source_fps:.2f} | Total frames: {progress_total if progress_total > 0 else 'unknown'}")
    print(f"Output video: {output_path}")
    print("Keys: q=quit, p=pause, s=save screenshot")

    paused = False
    current_frame: np.ndarray | None = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            detected_frame = lane_detection(frame, smoother)
            split_frame = np.hstack((frame, detected_frame))

            now = time.perf_counter()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now

            cv2.putText(split_frame, "Original", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(split_frame, "Lane Detection", (width + 20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            draw_hud(split_frame, frame_idx, progress_total, fps, output_path)

            writer.write(split_frame)
            current_frame = split_frame

            if save_every > 0 and frame_idx % save_every == 0:
                screenshot_path = screenshot_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(screenshot_path), split_frame)
                first_screenshot = first_screenshot or screenshot_path

            if progress_total > 0 and frame_idx % max(1, int(source_fps)) == 0:
                print(f"\rProcessed {frame_idx}/{progress_total} frames ({frame_idx / progress_total * 100:.1f}%)", end="")

            if max_frames > 0 and frame_idx >= max_frames:
                break

        if display and current_frame is not None:
            cv2.imshow("Lane Detection - Split Screen", current_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused
            if key == ord("s"):
                screenshot_path = screenshot_dir / f"frame_{frame_idx:06d}_manual.jpg"
                cv2.imwrite(str(screenshot_path), current_frame)
                first_screenshot = first_screenshot or screenshot_path
                print(f"\nScreenshot saved: {screenshot_path}")
        elif not display and current_frame is not None:
            # Headless mode still writes one representative screenshot for reports.
            if first_screenshot is None and frame_idx == 1:
                screenshot_path = screenshot_dir / "frame_000001_auto.jpg"
                cv2.imwrite(str(screenshot_path), current_frame)
                first_screenshot = screenshot_path

    print()
    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    print(f"Done. Processed frames: {frame_idx}")
    print(f"Saved video: {output_path}")
    if first_screenshot is not None:
        print(f"Saved screenshot: {first_screenshot}")

    return output_path, first_screenshot, frame_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect lane lines in a driving video.")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for video and screenshots")
    parser.add_argument("--no-display", action="store_true", help="Run without opening an OpenCV window")
    parser.add_argument("--save-every", type=int, default=0, help="Save a screenshot every N frames")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means process the whole video")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_video(args.video, args.output_dir, display=not args.no_display, save_every=args.save_every, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
