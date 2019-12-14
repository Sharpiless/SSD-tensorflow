import numpy as np
import config as cfg
import math
import tensorflow as tf


def ssd_anchor_all_layers(image):

    image_shape = image.shape

    layers_shape, anchor_shapes = get_layers_shape(image_shape)

    layers_anchors = []

    for i, layer_size in enumerate(layers_shape):

        anchor_size = anchor_shapes[i]

        anchors = ssd_anchor_one_layer(
            image_shape, layer_size, anchor_size, ratio=cfg.RATIOS[i], stride=cfg.STRIDES[i]
        )

        layers_anchors.append(anchors)

    return layers_anchors


def ssd_anchor_one_layer(image_shape, layer_size, anchor_size, ratio, stride):

    y, x = np.mgrid[0:layer_size[0], 0:layer_size[1]]

    y = (y.astype(np.float)+0.5) * stride  # 原图上的锚定点
    x = (x.astype(np.float)+0.5) * stride  # 原图上的锚定点

    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    anchor_num = len(ratio) + len(anchor_size)

    h = np.zeros((anchor_num, ), dtype=np.float)
    w = np.zeros((anchor_num, ), dtype=np.float)

    di = 1

    h[0] = anchor_size[0]
    w[0] = anchor_size[0]

    if len(anchor_size) > 1:

        h[1] = math.sqrt(anchor_size[0]*anchor_size[1])
        w[1] = math.sqrt(anchor_size[0]*anchor_size[1])

        di += 1

    for i, r in enumerate(ratio):

        h[i+di] = anchor_size[0] / math.sqrt(r)
        w[i+di] = anchor_size[0] * math.sqrt(r)

    anchors = convert_format(y, x, w, h)

    return anchors.astype(np.float32)


def convert_format(y, x, w, h):

    bias = get_conners_coord(w, h)

    center_point = np.stack((y, x, y, x), axis=-1)

    anchors = center_point + bias 
    anchors = np.reshape(
        anchors, [y.shape[0], y.shape[1], bias.shape[0], 4])

    return anchors


def get_conners_coord(w, h):

    width = w
    height = h

    # 分别计算四点坐标
    x_min = np.round(0 - 0.5 * width)
    y_min = np.round(0 - 0.5 * height)
    x_max = np.round(0 + 0.5 * width)
    y_max = np.round(0 + 0.5 * height)

    bias = np.stack((y_min, x_min, y_max, x_max), axis=-1)

    return bias


def get_layers_shape(image_shape, Sk=cfg.Sk):

    height, width = image_shape[:2]

    H, W = height, width

    layers_shape = []

    for _ in range(3):

        height = math.ceil(height/2)
        width = math.ceil(width/2)

    for i in range(7):

        layers_shape.append((height, width))
        
        height = math.ceil(height/2)
        width = math.ceil(width/2)

    anchor_shapes = np.array(Sk)  # 查论文

    return layers_shape, anchor_shapes


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import cv2

    anchor_index = -4

    image = np.zeros((512, 512, 3))

    result = ssd_anchor_all_layers(image)

    anchors = result[anchor_index].reshape(-1, 4)
    nums = anchors.shape[0]

    anchors = anchors[int(nums/2):int(nums/2)+20]

    print(anchors.shape)

    for i in range(anchors.shape[0]):


        image.fill(255)

        color = (10 * (i+1), 0, 0)

        y_min, x_min, y_max, x_max = anchors[i].astype(np.int)

        cv2.rectangle(image, (x_min, y_min),
                      (x_max, y_max), color, thickness=2)

        plt.imshow(image)
        plt.show()