import config as cfg
import tensorflow as tf
from read_data import Reader
from anchors import ssd_anchor_all_layers
from label_anchors import ssd_bboxes_encode
from loss_function import loss_layer
import numpy as np
import os
import logging as log

slim = tf.contrib.slim


class Net(object):

    def __init__(self, is_training):

        self.reader = Reader(is_training)

        self.is_training = is_training

        self.learning_rate = cfg.LEARNING_RATE

        self.class_num = len(cfg.CLASSES)

        self.weight_decay = cfg.WEIGHT_DECAY

        self.blocks = cfg.BLOCKS

        self.ratios = cfg.RATIOS

        self.keep_rate = cfg.KEEP_RATE

        self.model_path = cfg.MODEL_PATH

        self.momentum = cfg.MOMENTUM

        self.Sk = cfg.Sk

        self.x = tf.placeholder(tf.float32, [None, None, 3])

        self.true_labels = tf.placeholder(tf.float32, [None])

        self.true_boxes = tf.placeholder(tf.float32, [None, 4])

        self.output = self.ssd_net(
            tf.expand_dims(self.x, axis=0)
        )

        self.anchors = tf.numpy_function(
            ssd_anchor_all_layers, [self.x],
            [tf.float32]*7
        )

        with open('var.txt', 'a') as f:
            variables = tf.contrib.framework.get_variables_to_restore()
            variables_name = [
                v.name for v in variables if v.name.split('/')[0] != 'output']
            for v in variables_name:
                f.write(v+'\n')

        self.saver = tf.train.Saver()

    def ssd_net(self, inputs, scope='ssd_512_vgg'):

        layers = {}

        with tf.variable_scope(scope, 'ssd_512_vgg', [inputs], reuse=None):

            with slim.arg_scope([slim.conv2d],
                           activation_fn=tf.nn.relu,
                           weights_regularizer=slim.l2_regularizer(self.weight_decay)):

                # Block 1
                net = slim.repeat(inputs, 2, slim.conv2d,
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

                # net = tf.layers.batch_normalization(net, training=self.is_training)

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

                # Dropout
                if self.is_training:

                    net = tf.nn.dropout(net, keep_prob=self.keep_rate)

                # Block 7
                net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
                layers['block7'] = net

                # Dropout
                if self.is_training:

                    net = tf.nn.dropout(net, keep_prob=self.keep_rate)

                # Block 8
                with tf.variable_scope('block8'):

                    net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
                    net = slim.conv2d(net, 512, [3, 3], 2,
                                    scope='conv3x3', padding='SAME')

                layers['block8'] = net

                # Block 9
                with tf.variable_scope('block9'):

                    net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                    net = slim.conv2d(net, 256, [3, 3], 2,
                                    scope='conv3x3', padding='SAME')

                layers['block9'] = net

                # Block 10
                with tf.variable_scope('block10'):

                    net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                    net = slim.conv2d(net, 256, [3, 3], 2,
                                    scope='conv3x3', padding='SAME')

                layers['block10'] = net

                # Block 11
                with tf.variable_scope('block11'):

                    net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                    net = slim.conv2d(net, 256, [3, 3], 2,
                                    scope='conv3x3', padding='SAME')

                layers['block11'] = net

                # Block 12
                with tf.variable_scope('block12'):

                    net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                    net = slim.conv2d(net, 256, [4, 4], 2,
                                    scope='conv4x4', padding='SAME')

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

            return pred_loc, pred_score

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

        if not os.path.exists(cfg.MODEL_PATH):
            os.makedirs(cfg.MODEL_PATH)

        self.target_labels = []
        self.target_scores = []
        self.target_loc = []

        for i in range(7):

            target_labels, target_scores, target_loc = tf.numpy_function(
                ssd_bboxes_encode, [self.anchors[i], self.true_boxes,
                                    self.true_labels, self.class_num],
                [tf.float32, tf.float32, tf.float32]
            )

            self.target_labels.append(target_labels)
            self.target_scores.append(target_scores)
            self.target_loc.append(target_loc)

        self.total_cross_pos, self.total_cross_neg, self.total_loc = loss_layer(
            self.output, self.target_labels, self.target_scores, self.target_loc
        )

        self.loss = tf.add(
            tf.add(self.total_cross_pos, self.total_cross_neg), self.total_loc
        )

        # gradients = self.optimizer.compute_gradients(self.loss)

        self.optimizer = tf.compat.v1.train.MomentumOptimizer(
            learning_rate=self.learning_rate, momentum=self.momentum)
        # self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)

        self.train_step = self.optimizer.minimize(self.loss)

        with tf.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(self.model_path)

            if ckpt and ckpt.model_checkpoint_path:
                # 如果保存过模型，则在保存的模型的基础上继续训练
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                log.info('Model Reload Successfully!')

            for i in range(cfg.EPOCHES):
                loss_list = []
                for batch in range(cfg.BATCHES):

                    value = self.reader.generate()

                    image = value['image'] - cfg.PIXEL_MEANS
                    true_labels = value['classes']
                    true_boxes = value['boxes']

                    feed_dict = {self.x: image,
                                 self.true_labels: true_labels,
                                 self.true_boxes: true_boxes}

                    test = sess.run(self.target_scores, feed_dict)

                    total_pos = 0
                    
                    for v in test:
                        if np.max(v) > cfg.THRESHOLD:
                            total_pos += 1
                    if total_pos == 0:
                        continue

                    try:

                        sess.run(self.train_step, feed_dict)

                        loss_0, loss_1, loss_2 = sess.run(
                            [self.total_cross_pos, self.total_cross_neg, self.total_loc], feed_dict)

                    except EOFError as e:
                        print(e)

                    loss_list.append(
                        np.array([loss_0, loss_1, loss_2])
                    )

                    print('batch:{},pos_loss:{},neg_loss:{},loc_loss:{}'.format(
                        batch, loss_0, loss_1, loss_2
                    ), end='\r')

                loss_values = np.array(loss_list)  # (64, 3)

                loss_values = np.mean(loss_values, axis=0)

                with open('./result.txt', 'a') as f:
                    f.write(str(loss_values)+'\n')

                self.saver.save(sess, os.path.join(
                    self.model_path, 'model.ckpt'))

                print('epoch:{},pos_loss:{},neg_loss:{},loc_loss:{}'.format(
                    self.reader.epoch, loss_values[0], loss_values[1], loss_values[2]
                ))


if __name__ == '__main__':

    net = Net(is_training=True)

    net.train_net()
