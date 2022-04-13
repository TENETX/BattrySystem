# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 22:48:20 2021

@author: 86133
"""
import os
import random
from pathlib import Path
import argparse
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import sys
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier
import SQL

root_path = os.getcwd()
sys.path.insert(0, "D:/yolov5-4.0")

# 设置参数
rmax = 360
rmin = 80
save = "D:/photos/"


def letterbox(img,
              new_shape=(640, 640),
              color=(114, 114, 114),
              auto=True,
              scaleFill=False,
              scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
        1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[
            0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img,
                             top,
                             bottom,
                             left,
                             right,
                             cv2.BORDER_CONSTANT,
                             value=color)  # add border
    return img, ratio, (dw, dh)


def detect(save_img=False):
    # 原来detect的参数预加载
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith(
        '.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(
        increment_path(Path(opt.project) / opt.name,
                       exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load("D:/yolov5-4.0/runs/train/exp/weights/best.pt",
                         map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(
            torch.load('weights/resnet101.pt',
                       map_location=device)['model']).to(device).eval()

    # Set Dataloader

    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        # dataset = LoadImages(source, img_size=imgsz)
        cudnn.benchmark = True

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img
              ) if device.type != 'cpu' else None  # run once
    # 此处为读取图片，在摄像头中依靠read来获取图片——修改source地址
    imgt = cv2.imread(source + '/003.jpg')
    img_gray = cv2.cvtColor(imgt, cv2.COLOR_BGR2GRAY)
    img_back = cv2.imread(source + '/003.jpg')
    circles = cv2.HoughCircles(img_gray,
                               cv2.HOUGH_GRADIENT,
                               1,
                               800,
                               param1=400,
                               param2=30,
                               minRadius=200,
                               maxRadius=400)
    circles = np.uint16(np.around(circles))
    numbers = 0
    for i in circles[0, :]:
        imgl = img_back[i[1] - rmax:i[1] + rmax, i[0] - rmax:i[0] + rmax]
        img_save = img_back[i[1] - rmin:i[1] + rmin, i[0] - rmin:i[0] + rmin]
        imgq = cv2.resize(img_save, (imgsz, imgsz))
        imgq = letterbox(imgq, new_shape=imgsz)[0]
        x = i[0]
        y = i[1]
        r = i[2]
        # 图片操作
        imgq = imgq[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        imgq = np.ascontiguousarray(imgq)
        imgq = torch.from_numpy(imgq).to(device)
        imgq = imgq.half() if half else imgq.float()  # uint8 to fp16/32
        imgq /= 255.0  # 0 - 255 to 0.0 - 1.0
        if imgq.ndimension() == 3:
            imgq = imgq.unsqueeze(0)
        # 预测
        pred = model(imgq, augment=opt.augment)[0]
        pred = non_max_suppression(pred,
                                   opt.conf_thres,
                                   opt.iou_thres,
                                   classes=opt.classes,
                                   agnostic=opt.agnostic_nms)
        # 处理结果
        for j, det in enumerate(pred):
            numr = ''
            ret = ''
            if len(det):
                det[:, :4] = scale_coords(imgq.shape[2:], det[:, :4],
                                          img_save.shape).round()
            # Print results
            for c in det[:, -1].unique():
                numr = f'{names[int(c)]}'
                ret = int(c)
                # numr储存结果名字，ret储存结果序号
            for *xyxy, conf, cls in reversed(det):
                if save_img or view_img:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(1,
                                 xyxy,
                                 img_save,
                                 label=label,
                                 color=colors[int(cls)],
                                 line_thickness=3)
            print(f'{ret}  {numr}')
            # 标记结果
            plot_one_box(2, [x - rmax, y - rmax, x + rmax, y + rmax], imgt, label=numr, color=colors[int(ret)],
                         line_thickness=3)
            url = save + str(numbers) + ".jpg"
            cv2.imwrite(url, imgl)  # 保存单个盖帽结果
            numbers += 1
            SQL.insert(numr, "user", url, ret)

    cv2.imwrite(save + "result.jpg", imgt)  # 保存整个图片结果


if __name__ == '__main__':
    SQL.createone()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights',
        nargs='+',
        type=str,
        default='D:/yolov5-4.0/runs/train/exp/weights/best.pt',
        help='model.pt path(s)')
    parser.add_argument('--source',
                        type=str,
                        default="D:/yp",
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size',
                        type=int,
                        default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.25,
                        help='object confidence threshold')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.45,
                        help='IOU threshold for NMS')
    parser.add_argument('--device',
                        default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img',
                        action='store_true',
                        help='display results')
    parser.add_argument('--save-txt',
                        action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf',
                        action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--classes',
                        nargs='+',
                        type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms',
                        action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment',
                        action='store_true',
                        help='augmented inference')
    parser.add_argument('--update',
                        action='store_true',
                        help='update all models')
    parser.add_argument('--project',
                        default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name',
                        default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok',
                        action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in [
                    'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt'
            ]:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
