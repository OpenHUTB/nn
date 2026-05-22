
import sys
md_path = "D:\\github\\nn\\docs\\carla_traffic_sign_recognition\\carla_traffic_sign_recognition.md"
with open(md_path, "rb") as f:
    raw = f.read()
text = raw.decode("utf-8")

# Extract exact text for each section and write to a debug file
sections = {
    "s1_1_para": (text.find("\n\n\u672c\u9879\u76ee\u6e90\u4e8e"), text.find("\n\n### 1.2")),
    "s1_2_para1": (text.find("\u5728\u73b0\u4ee3\u81ea"), text.find("\u4ea4\u901a\u6807\u5fd7\u8bc6\u522b") + 200),
    "s1_3_list": (text.find("**\u6a21\u62df\u73af\u5883**"), text.find("\u64cd\u4f5c\u754c\u9762\u3002")),
}

out = []
for name, (start, end) in sections.items():
    if start >= 0:
        out.append(f"=== {name} [{start}:{end}] ===")
        out.append(text[start:end])
    else:
        out.append(f"=== {name} NOT FOUND ===")

with open("D:\\github\\nn\\_sections.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(out))
print("Extracted")
