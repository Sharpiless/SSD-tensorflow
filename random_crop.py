import numpy as np
import config as cfg
import random


class Cropper(object):

    def __init__(self):

        self.min_ratio = cfg.MIN_CROP_RATIO

        self.max_ratio = cfg.MAX_CROP_RATIO

    def random_flip(self, image, boxes):

        flag = random.randint(0, 1)

        if flag:

            image = np.flip(image, axis=1)

            h, w = image.shape[:2]

            y_min = boxes[:, 0]
            x_min = boxes[:, 1]
            y_max = boxes[:, 2]
            x_max = boxes[:, 3]

            new_y_min = y_max
            new_y_max = y_min
            new_x_min = w - x_max
            new_x_max = w - x_min

            # print('flip')

            boxes = np.stack(
                [new_y_min, new_x_min, new_y_max, new_x_max], axis=-1)

            return image, boxes

        else:
            return image, boxes

    def random_crop(self, image, boxes, labels):

        h, w = image.shape[:2]

        ratio = random.random()

        scale = self.min_ratio + ratio * (self.max_ratio - self.min_ratio)

        new_h = int(h*scale)
        new_w = int(w*scale)

        y = np.random.randint(0, h - new_h)
        x = np.random.randint(0, w - new_w)

        image = image[y:y+new_h, x:x+new_w, :]

        y_min = boxes[:, 0]
        x_min = boxes[:, 1]
        y_max = boxes[:, 2]
        x_max = boxes[:, 3]

        raw_areas = (y_max - y_min) * (x_max - x_min)

        y_min = y_min - y
        y_max = y_max - y
        x_min = x_min - x
        x_max = x_max - x

        y_min = np.clip(y_min, 0, new_h)
        y_max = np.clip(y_max, 0, new_h)
        x_min = np.clip(x_min, 0, new_w)
        x_max = np.clip(x_max, 0, new_w)

        new_areas = (y_max - y_min) * (x_max - x_min)

        # keep_index = np.where(new_areas > raw_areas*0.7)[0]

        boxes = np.stack([y_min, x_min, y_max, x_max], axis=-1)
        # boxes = boxes[keep_index]
        # labels = labels[keep_index]

        return image, boxes, labels
