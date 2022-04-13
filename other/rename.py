import os
from xml.dom.minidom import Document
import cv2 as cv


num = 274
path1 = 'D:/yolov5-4.0/battry_test/labels/train/'
path2 = 'D:/yolov5-4.0/battry_test/img/train/'
def rename():
    for i in range(1, num + 1):
        if i < 64:
            k = 0
        elif i >= 64 & i < 121:
            k = 1
        else:
            k = 2
    f = open(path1 + '%s.txt' % str(i))
    n = path2 + '%s.txt' % str(i)
    img = cv.imread(n)
    x = img.shape[1]
    y = img.shape[0]
    d = img.shape[2]

