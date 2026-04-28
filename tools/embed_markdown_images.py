#!/usr/bin/env python3
"""Embed local Markdown image references as base64 data URIs.

This keeps a Markdown file self-contained so it can still display images
after the original image files are moved or deleted.
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import re
from pathlib import Path


IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def is_remote_path(path_text: str) -> bool:
    lowered = path_text.lower()
    return lowered.startswith(("http://", "https://", "data:"))


def clean_target(raw_target: str) -> str:
    target = raw_target.strip().strip("<>").strip()
    if " " in target and target.startswith(("\"", "'")) and target.endswith(("\"", "'")):
        target = target[1:-1]
    return target


def embed_images(markdown_path: Path, inplace: bool) -> tuple[str, int]:
    text = markdown_path.read_text(encoding="utf-8")
    replaced = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replaced

        alt_text, raw_target = match.groups()
        target = clean_target(raw_target)

        if is_remote_path(target):
            return match.group(0)

        asset_path = (markdown_path.parent / target).resolve()
        if not asset_path.is_file():
            return match.group(0)

        mime_type, _ = mimetypes.guess_type(asset_path.name)
        mime_type = mime_type or "application/octet-stream"
        encoded = base64.b64encode(asset_path.read_bytes()).decode("ascii")
        replaced += 1
        return f'![{alt_text}](data:{mime_type};base64,{encoded})'

    new_text = IMAGE_PATTERN.sub(replace, text)

    if inplace:
        markdown_path.write_text(new_text, encoding="utf-8")

    return new_text, replaced


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed local Markdown images as base64 data URIs."
    )
    parser.add_argument("markdown_file", type=Path, help="Path to the Markdown file")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the converted Markdown instead of overwriting the file",
    )
    args = parser.parse_args()

    markdown_path = args.markdown_file.resolve()
    if not markdown_path.is_file():
        raise SystemExit(f"Markdown file not found: {markdown_path}")

    new_text, replaced = embed_images(markdown_path, inplace=not args.stdout)

    if args.stdout:
        print(new_text, end="")
    else:
        print(f"Embedded {replaced} image(s) into {markdown_path}")


if __name__ == "__main__":
    main()
