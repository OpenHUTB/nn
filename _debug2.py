import sys

md_path = "D:\\github\\nn\\docs\\carla_traffic_sign_recognition\\carla_traffic_sign_recognition.md"
with open(md_path, "rb") as f:
    raw = f.read()
text = raw.decode("utf-8")

# Find section header positions
s1_1 = text.find("### 1.1")
s1_2 = text.find("### 1.2") 
s1_3 = text.find("### 1.3")
s2 = text.find("## 2.")
s3 = text.find("## 3.")
s4 = text.find("## 4.")
s5 = text.find("## 5.")
s6 = text.find("## 6.")
s7 = text.find("## 7.")

print(f"1.1: {s1_1}, 1.2: {s1_2}, 1.3: {s1_3}")
print(f"2: {s2}, 3: {s3}, 4: {s4}")
print(f"5: {s5}, 6: {s6}, 7: {s7}")
print(f"Total length: {len(text)}")
