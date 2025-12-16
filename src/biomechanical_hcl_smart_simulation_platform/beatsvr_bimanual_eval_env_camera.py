from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from PIL import Image


def mp4_to_gif(mp4_path: Path, gif_path: Path, target_width: int = 640, fps: int = 15) -> None:
    """Convert one MP4 to GIF with fps downsampling and width resize."""
    reader = imageio.get_reader(str(mp4_path))
    meta = reader.get_meta_data()
    src_fps = float(meta.get("fps", 20.0))

    # Downsample frames to target fps
    step = max(1, int(round(src_fps / float(fps))))

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(gif_path), mode="I", fps=fps, loop=0)

    try:
        for i, frame in enumerate(reader):
            if i % step != 0:
                continue

            im = Image.fromarray(frame)
            w, h = im.size
            if target_width and w != target_width:
                new_h = int(h * target_width / w)
                im = im.resize((target_width, new_h), resample=Image.Resampling.LANCZOS)

            writer.append_data(np.array(im))
    finally:
        writer.close()
        reader.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="Input mp4 files (paths).")
    p.add_argument("--out_dir", default=".", help="Output folder.")
    p.add_argument("--fps", type=int, default=15, help="GIF FPS.")
    p.add_argument("--width", type=int, default=640, help="Output GIF width (keep aspect).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    for in_path_str in args.inputs:
        in_path = Path(in_path_str)
        if not in_path.exists():
            raise FileNotFoundError(f"Input not found: {in_path}")

        gif_name = in_path.with_suffix(".gif").name
        out_path = out_dir / gif_name
        mp4_to_gif(in_path, out_path, target_width=args.width, fps=args.fps)
        print(f"[OK] {in_path} -> {out_path}")


if __name__ == "__main__":
    main()
