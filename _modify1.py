
import sys

md_path = "D:\\github\\nn\\docs\\carla_traffic_sign_recognition\\carla_traffic_sign_recognition.md"

with open(md_path, "rb") as f:
    raw = f.read()
text = raw.decode("utf-8")

# Marker: "## 摘要" followed by newline
marker = "## \u6458\u8981\r\n\r\n"
start = text.find(marker)
if start < 0:
    # Try Unix line endings
    marker = "## \u6458\u8981\n\n"
    start = text.find(marker)
if start < 0:
    print("FAIL: marker not found")
    sys.exit(1)

content_start = start + len(marker)
end_marker = "\r\n\r\n------"
end = text.find(end_marker, content_start)
if end < 0:
    end_marker = "\n\n------"
    end = text.find(end_marker, content_start)
if end < 0:
    print("FAIL: end marker not found")
    sys.exit(2)

old_abstract = text[content_start:end]
# Replace "填补了" with "填补了" (same), just add small modifications
new_abstract = old_abstract.replace("（Human-in-the-loop）", "（Human-in-the-Loop）")
new_abstract = new_abstract.replace("，涵盖了", "，覆盖了")
# Make a small change
if new_abstract == old_abstract:
    # Fallback: change first sentence slightly
    new_abstract = old_abstract.replace("本文档详细阐述了为", "本文档旨在阐述为")

text = text[:content_start] + new_abstract + text[end:]

with open(md_path, "wb") as f:
    f.write(text.encode("utf-8"))

print("OK")
