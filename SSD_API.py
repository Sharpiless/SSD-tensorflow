import tensorflow as tf
from network import Net
import config as cfg
import cv2
import numpy as np
from label_anchors import decode_targets
import matplotlib.pyplot as plt
from nms import py_cpu_nms


class SSD_detector(object):

    def __init__(self):

        self.net = Net(is_training=False)

        self.model_path = cfg.MODEL_PATH

        self.pixel_means = cfg.PIXEL_MEANS

        self.min_size = cfg.MIN_SIZE

        self.pred_loc, self.pred_cls = self.net.output

        self.score_threshold = cfg.SCORE_THRESHOLD

    def pre_process(self, image_path):

        image = cv2.imread(image_path)

        image = image.astype(np.float)

        image, scale = self.resize_image(image)

        value = {'image': image, 'scale': scale, 'image_path': image_path}

        return value

    def resize_image(self, image):

        image_shape = image.shape

        size_min = np.min(image_shape[:2])
        size_max = np.max(image_shape[:2])

        scale = float(self.min_size) / float(size_min)

        image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)

        return image, scale

    def test_ssd(self, image_paths):

        if isinstance(image_paths, str):
            image_paths = [image_paths]

        with tf.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(cfg.MODEL_PATH)

            if ckpt and ckpt.model_checkpoint_path:
                # 如果保存过模型，则在保存的模型的基础上继续训练
                self.net.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model Reload Successfully!')

            for path in image_paths:

                value = self.pre_process(path)

                image = value['image'] - self.pixel_means

                feed_dict = {self.net.x: image}

                pred_loc, pred_cls, layer_anchors = sess.run(
                    [self.pred_loc, self.pred_cls, self.net.anchors], feed_dict
                )

                pos_loc, pos_cls, pos_anchors, pos_scores = self.decode_output(
                    pred_loc, pred_cls, layer_anchors)

                pos_boxes = decode_targets(pos_anchors, pos_loc, image.shape)

                pos_scores = np.expand_dims(pos_scores, axis=-1)
                
                keep_index = py_cpu_nms(np.hstack([pos_boxes, pos_scores]))

                self.draw_result(
                    value['image'], pos_boxes[keep_index], pos_cls[keep_index], value['scale']
                )

    def draw_result(self, image, pos_boxes, pos_cls, scale, font=cv2.FONT_HERSHEY_SIMPLEX):

        image = cv2.resize(image, dsize=(0, 0), fx=1/scale, fy=1/scale)
        
        image = image.astype(np.int)
        
        pos_boxes = pos_boxes * (1/scale)

        for i in range(pos_boxes.shape[0]):

            bbox = pos_boxes[i]
            label = cfg.CLASSES[pos_cls[i]]

            y_min, x_min, y_max, x_max = bbox.astype(np.int)

            cv2.rectangle(image, (x_min, y_min),
                          (x_max, y_max), (0, 0, 255), thickness=2)

            cv2.putText(image, label, (x_min+20, y_min+20),
                        font, 1, (255, 0, 0), thickness=2)

        plt.imshow(image[:, :, [2, 1, 0]])
        plt.show()

    def decode_output(self, pred_loc, pred_cls, layer_anchors):

        pos_loc, pos_cls, pos_anchors, pos_scores = [], [], [], []

        for i in range(len(pred_cls)):

            loc_ = pred_loc[i]
            cls_ = pred_cls[i]  # cls_是每个分类的得分
            anchors = layer_anchors[i].reshape((-1, 4))

            max_scores = np.max(cls_[:, 1:], axis=-1)  # 非背景最大得分
            cls_ = np.argmax(cls_, axis=-1)  # 最大索引

            pos_index = np.where(max_scores > self.score_threshold)[0]  # 正样本

            pos_loc.append(loc_[pos_index])
            pos_cls.append(cls_[pos_index])
            pos_anchors.append(anchors[pos_index])
            pos_scores.append(max_scores[pos_index])

        pos_loc = np.vstack(pos_loc)
        pos_cls = np.hstack(pos_cls)
        pos_anchors = np.vstack(pos_anchors)
        pos_scores = np.hstack(pos_scores)

        return pos_loc, pos_cls, pos_anchors, pos_scores


if __name__ == "__main__":

    detector = SSD_detector()

    images_name = ['./{}.jpg'.format(i) for i in range(7)]

    detector.test_ssd(images_name)
