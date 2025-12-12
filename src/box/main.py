import moblARMslndex 
import argparse
from certifi import where
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--contents", action="store_true", help="查看证书文件内容")
args = parser.parse_args()

if args.contents:
    # 读取 certifi 证书文件内容
    cert_path = where()
    with open(cert_path, "r", encoding="utf-8") as f:
        print(f.read())
else:
    # 打印证书文件路径
    print(where())