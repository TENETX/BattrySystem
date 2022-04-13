import glob

# 各项设置
rmax = 360  # 单个盖帽圆大小
rmin = 80  # 中心特征区域圆大小
save = "./photos/"  # 单个盖帽保存路径
pt = "./runs/train/exp/weights/best.pt"  # 模型地址
database_set = ['localhost', 'root', '123456', 'battry']  # 数据库设置

# 下面是批量标记样本设置
# 错误类型
name = "cid"
classes = [name]
# 图片路径,里面放好样本图片
path_photos = 'D:/Yolov5/data/' + name + '/JPEGImages/'
# 以下为测试集和测试集的比例
trainval_percent = 0.5
train_percent = 0.5

num = len(glob.glob(path_photos + '*.jpg'))
path_xml = 'data/' + name + '/Annotations'
path_txt = 'data/' + name + '/ImageSets/Main'
