import config as cfg
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import os
import pickle
import random
from random_crop import Cropper


class Reader(object):
    def __init__(self, is_training):

        self.data_path = cfg.DATA_PATH

        self.cropper = Cropper()

        self.is_training = is_training

        self.max_size = cfg.MAX_SIZE

        self.min_size = cfg.MIN_SIZE

        self.CLASSES = cfg.CLASSES

        self.pixel_means = cfg.PIXEL_MEANS

        self.class_to_ind = dict(
            zip(self.CLASSES, range(len(self.CLASSES)))
        )

        self.cursor = 0

        self.epoch = 1

        self.true_labels = None

        self.pre_process()

    def read_image(self, path):

        image = cv2.imread(path)

        return image.astype(np.float)

    def load_one_info(self, name):

        filename = os.path.join(self.data_path, 'Annotations', name+'.xml')

        tree = ET.parse(filename)

        Objects = tree.findall('object')

        objs_num = len(Objects)

        Boxes = np.zeros((objs_num, 4), dtype=np.float32)

        True_classes = np.zeros((objs_num), dtype=np.float32)

        for i, obj in enumerate(Objects):

            bbox = obj.find('bndbox')

            x_min = float(bbox.find('xmin').text) - 1

            y_min = float(bbox.find('ymin').text) - 1

            x_max = float(bbox.find('xmax').text) - 1

            y_max = float(bbox.find('ymax').text) - 1

            obj_cls = obj.find('name').text.lower().strip()

            obj_cls = self.class_to_ind[obj_cls]

            Boxes[i, :] = [y_min, x_min, y_max, x_max]

            True_classes[i] = obj_cls

            image_path = os.path.join(
                self.data_path, 'JPEGImages', name + '.jpg')

        return {'boxes': Boxes, 'classes': True_classes, 'image_path': image_path}

    def load_labels(self):

        is_training = 'train' if self.is_training else 'test'

        if not os.path.exists('./dataset'):
            os.makedirs('./dataset')

        pkl_file = os.path.join('./dataset', is_training+'_labels.pkl')

        if os.path.isfile(pkl_file):

            print('Load Label From '+str(pkl_file))
            with open(pkl_file, 'rb') as f:
                labels = pickle.load(f)

            return labels

        # else

        print('Load labels from: '+str(cfg.ImageSets_PATH))

        if self.is_training:
            txt_path = os.path.join(cfg.ImageSets_PATH, 'Main', 'trainval.txt')
            # 这是用来存放训练集和测试集的列表的txt文件
        else:
            txt_path = os.path.join(cfg.ImageSets_PATH, 'Main', 'val.txt')

        with open(txt_path, 'r') as f:
            self.image_name = [x.strip() for x in f.readlines()]

        labels = []

        for name in self.image_name:
            # 包括objet box坐标信息 以及类别信息(转换成dict后的)
            true_label = self.load_one_info(name)
            labels.append(true_label)

        with open(pkl_file, 'wb') as f:
            pickle.dump(labels, f)

        print('Successfully saving '+is_training+'data to '+pkl_file)

        return labels

    def resize_image(self, image):

        image_shape = image.shape

        size_min = np.min(image_shape[:2])
        size_max = np.max(image_shape[:2])

        min_size = self.min_size

        scale = float(min_size) / float(size_min)

        image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)

        return image, scale

    def pre_process(self):

        true_labels = self.load_labels()

        if self.is_training:
            np.random.shuffle(true_labels)

        self.true_labels = true_labels

    def generate(self):

        image_path = self.true_labels[self.cursor]['image_path']
        image = self.read_image(image_path)
        true_boxes = self.true_labels[self.cursor]['boxes']
        true_labels = self.true_labels[self.cursor]['classes']

        image, true_boxes = self.cropper.random_flip(image, true_boxes)

        image, true_boxes, true_labels = self.cropper.random_crop(
            image, true_boxes, true_labels)

        image, scale = self.resize_image(image)

        true_boxes = true_boxes * scale

        self.cursor += 1

        if self.cursor >= len(self.true_labels):
            np.random.shuffle(self.true_labels)

            self.cursor = 0
            self.epoch += 1

        value = {'image': image, 'classes': true_labels,
                 'boxes': true_boxes, 'image_path': image_path, 'scale': scale}
        
        if true_boxes.shape[0] == 0:
            value = self.generate()

        return value


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    font = cv2.FONT_HERSHEY_SIMPLEX

    reader = Reader(is_training=True)

    for _ in range(10):

        value = reader.generate()

        image = value['image'].astype(np.int)
        classes = value['classes'].astype(np.int)
        true_boxes = value['boxes']

        height, width, _ = image.shape

        num = true_boxes.shape[0]

        for i in range(num):

            label = reader.CLASSES[classes[i]]
            box = true_boxes[i]

            y_min, x_min, y_max, x_max = box.astype(np.int)

            cv2.rectangle(image, (x_min, y_min),
                          (x_max, y_max), (255, 0, 0), thickness=2)

            cv2.putText(image, label, (x_min+20, y_min+20),
                        font, 1, (0, 0, 255), thickness=2)

        plt.imshow(image[:, :, [2, 1, 0]])
        plt.show()
