import numpy as np

import os


NMS_THRESHOLD = 0.3  # nms（非极大值抑制）的阙值

DATA_PATH = '../VOC2012'  # 数据集路径

ImageSets_PATH = os.path.join(DATA_PATH, 'ImageSets')

BLOCKS = ['block4', 'block7', 'block8',

          'block9', 'block10', 'block11', 'block12']  # 需要抽出的特征层

MAX_SIZE = 1000  # 图片最大边长

MIN_SIZE = 600  # 图片最小边长

EPOCHES = 20000  # 迭代次数

BATCHES = 64

GAMMA = 1.2

KEEP_RATE = 0.8

WEIGHT_DECAY = 5e-3

THRESHOLD = 0.5  # 正负样本匹配的阙值

SCORE_THRESHOLD = 0.998  # 测试时正样本得分阙值

MIN_CROP_RATIO = 0.6  # 随即裁剪的最小比率

MAX_CROP_RATIO = 1.0

MODEL_PATH = './model/'  # 模型保存路径

LEARNING_RATE = 2e-4 # 学习率

CLASSES = ['', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',

           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',

           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',

           'train', 'tvmonitor']

# 图片三像素均值

PIXEL_MEANS = np.array([[[122.7717, 115.9465, 102.9801]]])

# 不同层预选框的长宽比

RATIOS = [[2, .5],

          [2, .5, 3, 1./3],

          [2, .5, 3, 1./3],

          [2, .5, 3, 1./3],

          [2, .5, 3, 1./3],

          [2, .5], [2, .5]]

# 每层的步长

STRIDES = [8, 16, 32, 64, 128, 256, 512]

# 论文中的s，认为是每层预选框的边长大小

S = [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06]

Sk = [(20.48, 51.2),

      (51.2, 133.12),

      (133.12, 215.04),

      (215.04, 296.96),

      (296.96, 378.88),

      (378.88, 460.8),

      (460.8, 542.72)]

# 用于调整边框回归值在loss中的比率

PRIOT_SCALING = (0.1, 0.1, 0.2, 0.2)
