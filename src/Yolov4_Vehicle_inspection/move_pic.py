import os
import random
import shutil

# xml路径的地址
XmlPath = r'xml_train'
# 原图片的地址
pictureBasePath = r"E:\Insight-MVT_Annotation_Train"
# 保存图片的地址
saveBasePath = r"E:\picture_train"

total_xml = os.listdir(XmlPath)
num = len(total_xml)
list = range(num)
if os.path.exists(saveBasePath) == False:  # 判断文件夹是否存在
    os.makedirs(saveBasePath)
from tqdm import tqdm
for xml in tqdm(total_xml):
    xml_temp = xml.split("__")
    folder = xml_temp[0]
    filename = xml_temp[1].split(".")[0] + ".jpg"
    # print(folder)
    # print(filename)
    temp_pictureBasePath = os.path.join(pictureBasePath, folder)
    filePath = os.path.join(temp_pictureBasePath, filename)
    # print(filePath)
    newfile = xml.split(".")[0] + ".jpg"
    newfile_path = os.path.join(saveBasePath, newfile)
    # print(newfile_path)
    shutil.copyfile(filePath, newfile_path)
print("xml file total number", num)