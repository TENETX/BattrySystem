import queue
import cv2
import numpy as np
import myyolo as yolov5
import time as t
import mainwindow
from setting import pt, rmax
# 图片流
stream = queue.Queue()
whole = queue.Queue()
ss = queue.Queue()
fina = []


def pro():
    # 此处为读取图片，在摄像头中依靠read来获取图片——修改source地址
    s = False
    # 开始整个程序的标志,也就是开始时间间隔截图
    # yolov5.load()
    print("检测开始")
    while not mainwindow.stopexplore:
        if mainwindow.pauseexplore:
            continue
        imgt = cv2.imread("1.png")
        img_gray = cv2.cvtColor(imgt, cv2.COLOR_BGR2GRAY)
        # 霍夫圆检测
        circles = cv2.HoughCircles(img_gray,
                                   cv2.HOUGH_GRADIENT,
                                   1,
                                   800,
                                   param1=400,
                                   param2=20,
                                   minRadius=250,
                                   maxRadius=400)
        circles = np.uint16(np.around(circles))
        # timer = t.time()
        if s is False and len(circles) != 0:
            s = True
            # 检测到圆开始工作
        if s:
            # if timer % 5 == 0 and timer != 0:
            stream.put([imgt, circles])
            while len(fina) != len(circles[0, :]):  # 当标签数据出来后在继续
                continue
                # 调整坐标
            last = []
            for length in range(0, len(fina) - 1):
                f = False
                for i in circles[0, :]:
                    if abs(fina[length][1][0] - i[0]) < 10 and fina[length][1][1] > i[1]:
                        fina[length][1][0] = i[0]
                        fina[length][1][1] = i[1]
                        f = True
                        break
                if f:
                    last.append(length)  # 如果没找到对应的,说明此时盖帽已经移除屏幕,需要进行清除
            if not last:
                for i in last:
                    del fina[last[i]]
            for length in range(0, len(fina) - 1):
                yolov5.plot_one_box(2,
                                    [fina[length][1][0] - rmax, fina[length][1][1] - rmax,
                                     fina[length][1][0] + rmax, fina[length][1][1] + rmax],
                                    imgt,
                                    color=yolov5.z,
                                    label=fina[length][0],
                                    line_thickness=3)
            whole.put(imgt)
    print("检测结束")


def det():
    print("推理开始")
    while not mainwindow.stopexplore:
        if mainwindow.pauseexplore:
            continue
        # 先得到label结果
        if not stream.empty():
            hs = yolov5.start_process(
                stream.get()[0], pt)  # 实际测试改成stream.get()
            circles = next(hs)
            for i in circles[0, :]:
                label = next(hs)
                fina.append([label, [i[0], i[1], i[2]]])
    print("推理结束")
