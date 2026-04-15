# ----------------------------------------------------#
#   对视频中的predict.py进行了修改，
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import time  # 导入时间模块，用于计时和FPS计算
import cv2  # 导入OpenCV库，用于视频/图像的读取、处理和保存
import numpy as np  # 导入numpy库，用于数组格式转换
from PIL import Image  # 导入PIL库的Image模块，用于图像格式转换

from yolo import YOLO  # 导入自定义的YOLO检测类

if __name__ == "__main__":
    # 初始化YOLO检测器，设置输入图像尺寸为640x640
    yolo = YOLO(size=640)

    # -------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测
    #   'video'表示视频检测
    #   'fps'表示测试fps
    # -------------------------------------------------------------------------#
    mode = "video"  # 当前运行模式设置为视频检测

    # -------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出才会完成完整的保存步骤，不可直接结束程序。
    # -------------------------------------------------------------------------#
    video_name = "333"  # 视频文件名称（不含后缀）
    video_path = r'vedio\int\%s.mp4' % video_name  # 输入视频的路径
    video_save_path = r"vedio\out\%s.mp4" % video_name  # 输出视频的保存路径
    video_fps = 30  # 保存视频的帧率

    if mode == "predict":
        '''
        1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
        具体流程可以参考get_dr_txt.py，在get_dr_txt.py即实现了遍历还实现了目标信息的保存。
        2、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        3、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        4、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        5、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        # 循环读取用户输入的图片路径，实现多次单张图片预测
        while True:
            img = input('Input image filename:')  # 提示用户输入图片文件路径
            start = time.time()  # 记录图片检测开始时间
            try:
                # 尝试打开指定路径的图片文件
                image = Image.open(img)
            except:
                # 打开失败时提示错误并重新输入
                print('Open Error! Try again!')
                continue
            else:
                # 图片打开成功，调用YOLO的检测函数进行目标检测
                r_image = yolo.detect_image(image)
                # 打印单张图片检测耗时
                print("time : ", time.time() - start)
                # 显示检测后的图片
                r_image.show()

    elif mode == "video":
        # 初始化视频读取器，读取指定路径的视频文件
        capture = cv2.VideoCapture(video_path)
        start = time.time()  # 记录视频检测总开始时间

        # 如果设置了视频保存路径，则初始化视频写入器
        if video_save_path != "":
            # 设置视频编码格式为mp4v（兼容mp4格式）
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 获取原视频的分辨率（宽度和高度）
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            # 创建视频写入器，指定保存路径、编码格式、帧率和分辨率
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        # 初始化FPS相关参数
        fps = 0.0  # 实时帧率（滑动平均）
        time_cnt = 0  # 帧计数，记录已处理的帧数
        cars = []  # 存储当前可见的车辆目标信息
        car_not_vis = []  # 存储不可见的车辆目标信息

        # 循环读取视频帧，直到视频结束或用户退出
        while (True):
            t1 = time.time()  # 记录当前帧处理开始时间

            # 读取视频的一帧，ref为读取成功标志，frame为帧数据
            ref, frame = capture.read()

            # 若读取失败（如视频结束、帧为空），跳出循环
            if not ref:
                break

            # ====================== 图像预处理：对比度和亮度调整 ======================
            # alpha>1增强对比度，alpha<1降低对比度；beta>0增亮，beta<0调暗
            alpha = 1.2  # 对比度增强20%（可根据实际视频调整1.1~1.5）
            beta = 8  # 亮度提升8（可根据实际视频调整5~15）
            # 调整帧的对比度和亮度，提升检测效果
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

            # ====================== 格式转换：适配YOLO检测要求 ======================
            # OpenCV读取的帧是BGR格式，转换为RGB格式（YOLO检测要求RGB）
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 将numpy数组格式的帧转换为PIL的Image格式（YOLO检测函数入参要求）
            frame = Image.fromarray(np.uint8(frame))

            # ====================== 目标检测：调用YOLO检测函数 ======================
            # 传入帧数据和车辆跟踪列表，执行目标检测并返回标注后的帧
            frame = np.array(yolo.detect_image(frame, cars, car_not_vis))

            # ====================== 格式还原：适配OpenCV显示/保存 ======================
            # 将RGB格式转换回BGR格式（OpenCV显示/保存要求BGR）
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # ====================== FPS计算与信息标注 ======================
            # 滑动平均计算实时FPS：(上一帧FPS + 当前帧FPS) / 2
            fps = (fps + (1. / (time.time() - t1))) / 2
            # 打印实时帧号、FPS和总耗时（\r实现单行覆盖打印）
            print("\rframe_id = %d\tfps = %.2f\ttime = %.2f" % (time_cnt, fps, time.time() - start), end="")
            time_cnt += 1  # 帧计数+1

            # 在帧上标注实时FPS（位置：左上角，字体：默认，大小：1，颜色：绿色，线宽：2）
            frame = cv2.putText(frame, "fps = %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # 在帧上标注当前检测到的车辆数量（位置：FPS下方，样式同上）
            frame = cv2.putText(frame, "cars_num = %d" % (len(cars)), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                2)

            # 可选：显示视频窗口（注释掉以提升处理速度）
            # cv2.imshow("video",frame)

            # 等待1ms，检测键盘输入（ESC键退出）
            c = cv2.waitKey(1) & 0xff

            # 如果设置了保存路径，将标注后的帧写入视频文件
            if video_save_path != "":
                out.write(frame)

            # 按ESC键（ASCII码27）退出视频检测循环
            if c == 27:
                break

        # 换行，避免打印信息重叠
        print()

        # ====================== 资源释放 ======================
        capture.release()  # 释放视频读取器资源
        # 若初始化了视频写入器，释放其资源
        if video_save_path != "":
            out.release()
        cv2.destroyAllWindows()  # 关闭所有OpenCV创建的窗口

    elif mode == "fps":
        # FPS测试模式：计算模型处理单张图片的平均耗时和帧率
        test_interval = 100  # 测试次数（连续检测100次取平均）
        img = Image.open('img/street.jpg')  # 读取测试用图片
        # 调用YOLO的FPS测试函数，返回单次检测平均耗时（秒）
        tact_time = yolo.get_FPS(img, test_interval)
        # 打印测试结果：单次耗时、帧率（1/耗时）、批量大小为1
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')
    else:
        # 模式指定错误时抛出断言异常，提示正确的模式选项
        raise AssertionError("Please specify the correct mode: 'predict', 'video' or 'fps'.")