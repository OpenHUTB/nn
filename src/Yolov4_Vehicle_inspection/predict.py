# ----------------------------------------------------#
#   对视频中的predict.py进行了修改，
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO(size=640)
    # -------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测
    #   'video'表示视频检测
    #   'fps'表示测试fps
    # -------------------------------------------------------------------------#
    mode = "video"
    # -------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出才会完成完整的保存步骤，不可直接结束程序。
    # -------------------------------------------------------------------------#
    video_name = "333"
    video_path = r'vedio\int\%s.mp4' % video_name
    video_save_path = r"vedio\out\%s.mp4" % video_name
    video_fps = 30

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
        while True:
            img = input('Input image filename:')
            start = time.time()
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                print("time : ", time.time() - start)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        start = time.time()
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        time_cnt = 0
        cars = []
        car_not_vis = []
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()


            if not ref:  # 防止视频读取完/帧为空导致后续报错
                break


            # 调整对比度（alpha）和亮度（beta），可根据实际视频微调：
            # alpha>1增强对比度，alpha<1降低对比度；beta>0增亮，beta<0调暗
            alpha = 1.2  # 对比度增强20%（可改1.1~1.5）
            beta = 8  # 亮度提升8（可改5~15）
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 原try-except可删除，因为已提前判断ref

            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(yolo.detect_image(frame, cars, car_not_vis))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("\rframe_id = %d\tfps = %.2f\ttime = %.2f" % (time_cnt, fps, time.time() - start), end="")
            time_cnt += 1
            frame = cv2.putText(frame, "fps = %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame = cv2.putText(frame, "cars_num = %d" % (len(cars)), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                2)

            # cv2.imshow("video",frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:  # 按ESC退出
                break
        # print("time: ", time.time() - start)
        print()
        capture.release()
        if video_save_path != "":  # 防止未保存视频时out未定义
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        test_interval = 100
        img = Image.open('img/street.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video' or 'fps'.")