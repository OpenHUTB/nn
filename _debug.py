
import re, sys

md_path = "D:\\github\\nn\\docs\\carla_traffic_sign_recognition\\carla_traffic_sign_recognition.md"
with open(md_path, "rb") as f:
    raw = f.read()
text = raw.decode("utf-8")

# Find sections by byte patterns
# Section 1.1
idx = text.find("### 1.1")
if idx > 0:
    print(f"1.1 found at {idx}")
    end = text.find("### 1.2", idx)
    if end > 0:
        section = text[idx:end]
        print(f"1.1 section: {repr(section[:150])}")
else:
    print("1.1 NOT FOUND")
