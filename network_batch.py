import config as cfg
import tensorflow as tf
from read_data import Reader
from anchors import ssd_anchor_all_layers
from label_anchors import ssd_bboxes_encode
from loss_function import loss_layer
import numpy as np
import os

slim = tf.contrib.slim


class Net(object):

    def __init__(self, is_training):

        self.reader = Reader(is_training)

        self.is_training = is_training

        self.learning_rate = cfg.LEARNING_RATE

        self.batch_size = cfg.BATCH_SIZE

        self.class_num = len(cfg.CLASSES)

        self.blocks = cfg.BLOCKS

        self.ratios = cfg.RATIOS

        self.Sk = cfg.Sk

        self.x = [tf.placeholder(tf.float32, [None, None, 3])]*self.batch_size

        self.true_labels = [tf.placeholder(tf.float32, [None])]*self.batch_size

        self.true_boxes = [tf.placeholder(
            tf.float32, [None, 4])]*self.batch_size

        self.pred_loc, self.pred_cls = self.ssd_net(self.x)

        self.saver = tf.train.Saver()

    def ssd_net(self, inputs):

        pred_loc_result = []
        pred_score_result = []

        for q in range(self.batch_size):

            layers = {}

            x = tf.expand_dims(inputs[q], axis=0)

            with tf.variable_scope('net', reuse=tf.AUTO_REUSE):

                # Block 1
                net = slim.repeat(x, 2, slim.conv2d,
                                64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1', padding='SAME')

                # Block 2
                net = slim.repeat(net, 2, slim.conv2d,
                                128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2', padding='SAME')

                # Block 3
                net = slim.repeat(net, 3, slim.conv2d,
                                256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3', padding='SAME')

                # Block 4
                net = slim.repeat(net, 3, slim.conv2d,
                                512, [3, 3], scope='conv4')

                layers['block4'] = net

                net = slim.max_pool2d(net, [2, 2], scope='pool4', padding='SAME')

                # Block 5
                net = slim.repeat(net, 3, slim.conv2d,
                                512, [3, 3], scope='conv5')

                # Block 6
                net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')

                # Block 7
                net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
                layers['block7'] = net

                # Block 8
                with tf.variable_scope('block8'):

                    net = slim.conv2d(net, 256, [1, 1], scope='conv8_1_1')
                    net = slim.conv2d(net, 512, [3, 3], 2,
                                    scope='conv8_3_3', padding='SAME')

                layers['block8'] = net

                # Block 9
                with tf.variable_scope('block9'):

                    net = slim.conv2d(net, 128, [1, 1], scope='conv9_1_1')
                    net = slim.conv2d(net, 256, [3, 3], 2,
                                    scope='conv9_3_3', padding='SAME')

                layers['block9'] = net

                # Block 10
                with tf.variable_scope('block10'):

                    net = slim.conv2d(net, 128, [1, 1], scope='conv10_1_1')
                    net = slim.conv2d(net, 256, [3, 3], 2,
                                    scope='conv10_3_3', padding='SAME')

                layers['block10'] = net

                # Block 11
                with tf.variable_scope('block11'):

                    net = slim.conv2d(net, 128, [1, 1], scope='conv11_1_1')
                    net = slim.conv2d(net, 256, [3, 3], 2,
                                    scope='conv11_3_3', padding='SAME')

                layers['block11'] = net

                # Block 12
                with tf.variable_scope('block12'):

                    net = slim.conv2d(net, 128, [1, 1], scope='conv12_1_1')
                    net = slim.conv2d(net, 256, [3, 3], 2,
                                    scope='conv12_3_3', padding='SAME')

                layers['block12'] = net
                self.layers = layers

                pred_loc = []
                pred_score = []

                for i, block in enumerate(self.blocks):

                    with tf.variable_scope(block+'_box'):

                        loc, score = self.ssd_multibox_layer(
                            layers[block], self.class_num, self.ratios[i], self.Sk[i]
                        )

                        pred_loc.append(loc)
                        pred_score.append(score)

                pred_loc_result.append(pred_loc)
                pred_score_result.append(pred_score)

        return pred_loc_result, pred_score_result

    def ssd_multibox_layer(self, inputs, class_num, ratio, size):

        num_anchors = len(size) + len(ratio)
        num_loc = num_anchors * 4
        num_cls = num_anchors * class_num

        # loc
        loc_pred = slim.conv2d(
            inputs, num_loc, [3, 3], activation_fn=None, scope='conv_loc')

        # cls
        cls_pred = slim.conv2d(
            inputs, num_cls, [3, 3], activation_fn=None, scope='conv_cls')

        loc_pred = tf.reshape(loc_pred, (-1, 4))
        cls_pred = tf.reshape(cls_pred, (-1, class_num))

        # softmax
        cls_pred = slim.softmax(cls_pred, scope='softmax')

        return loc_pred, cls_pred

    def train_net(self):

        self.optimizer = tf.compat.v1.train.MomentumOptimizer(
            learning_rate=self.learning_rate, momentum=0.9)

        self.loss = []

        for q in range(self.batch_size):

            anchors = tf.numpy_function(
                ssd_anchor_all_layers, [self.x],
                [tf.float32]*7
            )
            # self._anchors = ssd_anchor_all_layers(self.x)
            target_labels = []
            target_scores = []
            target_loc = []

            for i in range(7):

                t_labels, t_scores, t_loc = tf.numpy_function(
                    ssd_bboxes_encode, [anchors[i], self.true_boxes[q],
                                        self.true_labels[q], self.class_num],
                    [tf.float32, tf.float32, tf.float32]
                )

                target_labels.append(t_labels)
                target_scores.append(t_scores)
                target_loc.append(t_loc)

            total_cross_pos, total_cross_neg, total_loc = loss_layer(
                (self.pred_loc[q], self.pred_cls[q]), target_labels, target_scores, target_loc
            )

            loss = tf.add(
                tf.add(total_cross_pos,
                       total_cross_neg), total_loc
            )

            self.loss.append(loss)

        self.loss = tf.reduce_mean(self.loss, axis=0)

        self.train_step = self.optimizer.minimize(self.loss)

        with tf.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(cfg.MODEL_PATH)

            if ckpt and ckpt.model_checkpoint_path:
                # 如果保存过模型，则在保存的模型的基础上继续训练
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model Reload Successfully!')

            for i in range(cfg.EPOCHES):
                batch_images = []
                batch_labels = []
                batch_boxes = []

                for batch in range(self.batch_size):

                    value = self.reader.generate()

                    image = value['image']
                    true_labels = value['classes']
                    true_boxes = value['boxes']

                    batch_images.append(image)
                    batch_labels.append(true_labels)
                    batch_boxes.append(true_boxes)

                feed_dict = {self.x: batch_images,
                             self.true_labels: batch_labels,
                             self.true_boxes: batch_boxes}


                loss_value, _ = sess.run(
                    [self.loss, self.train_step], feed_dict)

                self.saver.save(sess, os.path.join(
                    cfg.MODEL_PATH, 'model.ckpt'))

                print('epoch:{}, loss:{}'.format(
                    self.reader.epoch, loss_value
                ))


if __name__ == '__main__':

    if not os.path.exists(cfg.MODEL_PATH):
        os.makedirs(cfg.MODEL_PATH)

    net = Net(is_training=True)

    net.train_net()
