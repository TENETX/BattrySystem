# C++调用代码入口
import mythread
import login
import mainwindow
import threading
import numpy as np


def Cget():
    t1 = threading.Thread(target=login.showout)
    t2 = threading.Thread(target=mythread.pro)
    t3 = threading.Thread(target=mythread.det)
    t1.start()
    while mainwindow.startexplore is False:
        continue
    t2.start()
    t3.start()
    return True


def arrayreset(array):
    a = array[:, 0:len(array[0] - 2):3]
    b = array[:, 1:len(array[0] - 2):3]
    c = array[:, 2:len(array[0] - 2):3]
    a = a[:, :, None]
    b = b[:, :, None]
    c = c[:, :, None]
    m = np.concatenate((a, b, c), axis=2)
    return m


def CgetSecond(image):
    if mainwindow.pauseexplore is False:
        img = arrayreset(image)
        mythread.ss.put(img)
    return mainwindow.stopexplore


if __name__ == '__main__':
    Cget()
