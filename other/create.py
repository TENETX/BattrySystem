import os
from xml.dom.minidom import Document
import cv2 as cv
import random
import glob
import xml.etree.ElementTree as ET

# 批量批注图片，放在Yolov5根目录，并在data中创建name文件夹，内部创建JPEGImages文件夹，内部放入图片即可。
# 带有自动标号和统计图片数量功能功能，只用放图片即可
# 错误类型
classes = ['zhadian', 'podian', 'liangping']
# 图片路径,里面放好样本图片
dataset = 'D:/yolov5-4.0/battry_test/images/train/'
path_photos = 'D:/yolov5-4.0/battry_test/images/train/'
# 以下为测试集和测试集的比例
trainval_percent = 0.5
train_percent = 0.5

num = len(glob.glob(path_photos + '*.jpg'))
path_xml = 'D:/yolov5-4.0/battry_test/images/train/Annotations'
path_txt = 'D:/yolov5-4.0/battry_test/labels/train'


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def renewname():
    i = 1
    for filename in os.listdir(path_photos):
        newname = str(i) + ".jpg"
        os.rename(path_photos + filename, path_photos + newname)
        i += 1


def create():
    for i in range(1, num + 1):
        if i < 64:
            k = 'zhadian'
        elif i >= 64 and i < 121:
            k = 'podian'
        else:
            k = 'liangping'
        filename = path_xml + '/' + str(i) + '.xml'
        n = 'D:/yolov5-4.0/battry_test/images/train/' + str(i) + '.jpg'
        f = open(filename, 'w')
        f.write(Document().toprettyxml(indent="  "))
        img = cv.imread(n)
        x = img.shape[1]
        y = img.shape[0]
        d = img.shape[2]
        f.writelines('<annotation>\n')
        f.writelines('                <folder>%s</folder>\n' % k)
        f.writelines('                <filename>%s</filename>\n' % i)
        f.writelines('                <path>%s</path>\n' % n)
        f.writelines('                <source>\n')
        f.writelines('                                <database>Unknown</database>\n')
        f.writelines('                </source>\n')
        f.writelines('                <size>\n')
        f.writelines('                                <width>%s</width>\n' % x)
        f.writelines('                                <height>%s</height>\n' % y)
        f.writelines('                                <depth>%s</depth>\n' % d)
        f.writelines('                </size>\n')
        f.writelines('                <segmented>1</segmented>\n')
        f.writelines('                <object>\n')
        f.writelines('                                <name>%s</name>\n' % k)
        f.writelines('                                <pose>Unspecified</pose>\n')
        f.writelines('                                <truncated>1</truncated>\n')
        f.writelines('                                <difficult>0</difficult>\n')
        f.writelines('                <bndbox>\n')
        f.writelines('                                <xmin>1</xmin>\n')
        f.writelines('                                <ymin>1</ymin>\n')
        f.writelines('                                <xmax>%s</xmax>\n' % x)
        f.writelines('                                <ymax>%s</ymax>\n' % y)
        f.writelines('                </bndbox>\n')
        f.writelines('                </object>\n')
        f.writelines('</annotation>')
        f.close()
        n = str(i) + '.xml'
        in_file = open(path_xml + '/' + n)  # 读取xml文件路径
        nn = str(i) + '.txt'
        out_file = open('D:/yolov5-4.0/battry_test/labels/train/' + nn, 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:  # 检索xml中的缺陷名称
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text),
                 float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(
                str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def txt():
    folder = os.path.exists(path_txt)
    if not folder:
        os.makedirs(path_txt)
    total_xml = os.listdir(path_xml)
    num = len(total_xml)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)
    ftrainval = open(path_txt + '/trainval.txt', 'w')
    ftest = open(path_txt + '/test.txt', 'w')
    ftrain = open(path_txt + '/train.txt', 'w')
    fval = open(path_txt + '/val.txt', 'w')
    for i in list:
        na = path_photos + total_xml[i][:-4] + '.jpg\n'
        if i in trainval:
            ftrainval.write(na)
            if i in train:
                ftrain.write(na)
            else:
                fval.write(na)
        else:
            ftest.write(na)
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


"""def new_yaml():
    f = open('data/' + name + '.yaml', 'w')
    f.writelines('train: %s/train.txt\n' % path_txt)
    f.writelines('val: %s/val.txt\n' % path_txt)
    f.writelines('test: %s/test.txt\n' % path_txt)
    f.writelines('\n')
    f.writelines('nc: 1\n')
    f.writelines('\n')
    f.writelines("names: ['%s']\n" % name)
    f.writelines('\n')
    f.close()
"""

if __name__ == '__main__':
    # renewname()
    create()
    txt()
    # new_yaml()
