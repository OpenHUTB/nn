"""Visualisation helpers: HUD overlays, composite frames and GIF export."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\consola.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _font(size: int = 14):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_hud(img: np.ndarray, lines: list[str], origin=(10, 8), size: int = 15, color=(255, 255, 255)) -> np.ndarray:
    """Draw a list of text lines with a translucent background box."""
    base = Image.fromarray(img).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font(size)
    pad = 6
    widths, heights = [], []
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        widths.append(box[2] - box[0])
        heights.append(box[3] - box[1])
    w = max(widths) + 2 * pad if widths else 0
    h = sum(heights) + (len(lines) + 1) * pad
    draw.rectangle([origin[0] - pad, origin[1] - pad, origin[0] + w, origin[1] + h], fill=(0, 0, 0, 110))
    y = origin[1]
    for line, hh in zip(lines, heights):
        draw.text((origin[0], y), line, font=font, fill=color)
        y += hh + pad
    return np.asarray(Image.alpha_composite(base, overlay).convert("RGB"))


def composite_demo(chase: np.ndarray, ego: np.ndarray, lines: list[str], inset: int = 176) -> np.ndarray:
    """Chase view with an ego-camera inset (top-right) and a HUD (top-left)."""
    h, w = chase.shape[:2]
    canvas = Image.fromarray(chase)
    ego_img = Image.fromarray(ego).resize((inset, int(inset * ego.shape[0] / ego.shape[1])), Image.BILINEAR)
    canvas.paste(ego_img, (w - ego_img.width - 8, 8))
    img = draw_hud(np.asarray(canvas), lines)
    return img


def save_gif(frames: list[np.ndarray], path: str, fps: int = 8, colors: int = 64, scale: float = 1.0) -> int:
    """Save frames as an optimised, palette-quantised GIF. Returns size in bytes."""
    if not frames:
        raise ValueError("no frames")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imgs = [Image.fromarray(f).convert("RGB") for f in frames]
    if scale != 1.0:
        imgs = [im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS) for im in imgs]
    imgs = [im.convert("P", palette=Image.ADAPTIVE, colors=colors) for im in imgs]
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
        disposal=2,
    )
    return os.path.getsize(path)


@dataclass
class MapView:
    resolution: float
    origin: tuple[float, float]
    width: int
    height: int
    scale: int = 3


def render_map(
    grid: np.ndarray,
    view: MapView,
    trajectory: list[tuple[float, float]],
    car_pose: tuple[float, float, float],
    scan_points: np.ndarray | None = None,
    title: str = "",
) -> np.ndarray:
    """Render an occupancy grid + trajectory + car + scan into an RGB image."""
    # occupancy colours: free = white, unknown = grey, occupied = black
    prob = 1.0 / (1.0 + np.exp(-grid))
    img = np.full((view.height, view.width, 3), 128, dtype=np.uint8)
    free = prob < 0.40
    occ = prob > 0.60
    img[free] = (248, 248, 248)
    img[occ] = (25, 25, 28)

    pil = Image.fromarray(img).resize((view.width * view.scale, view.height * view.scale), Image.NEAREST)
    draw = ImageDraw.Draw(pil)

    def to_px(x: float, y: float):
        px = (x - view.origin[0]) / view.resolution
        py = (y - view.origin[1]) / view.resolution
        return px * view.scale, (view.height - 1 - py) * view.scale

    if len(trajectory) > 1:
        pts = [to_px(x, y) for x, y in trajectory]
        draw.line(pts, fill=(0, 200, 255), width=2)
    if scan_points is not None and len(scan_points):
        for x, y in scan_points[::2]:
            px, py = to_px(x, y)
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(255, 90, 60))
    x, y, yaw = car_pose
    px, py = to_px(x, y)
    r = 7
    draw.ellipse([px - r, py - r, px + r, py + r], fill=(80, 160, 255), outline=(255, 255, 255))
    draw.line([px, py, px + r * np.cos(yaw), py - r * np.sin(yaw)], fill=(255, 255, 255), width=2)
    if title:
        draw.text((10, 8), title, font=_font(16), fill=(255, 255, 255))
    return np.asarray(pil)
