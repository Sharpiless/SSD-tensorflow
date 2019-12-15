import numpy as np
import tensorflow as tf
import config as cfg
from read_data import Reader
from math import ceil
import os
from random_crop import random_crop
import matplotlib.pyplot as plt

slim = tf.contrib.slim


class Net(object):

    def __init__(self, is_training):

        self.is_training = is_training

        self.epoches = cfg.EPOCHES

        self.learning_rate = cfg.LEARNING_RATE

        self.num_classes = len(cfg.CLASSES)

        self.model_path = cfg.MODEL_PATH

        self.batch_size = cfg.BATCH_SIZE

        self.target_size = cfg.TARGET_SIZE

        self.keep_rate = cfg.KEEP_RATE

        self.reader = Reader(is_training=is_training)

        self.x = tf.placeholder(
            tf.float32, [None, self.target_size, self.target_size, 3])

        self.y = tf.placeholder(
            tf.int32, [None, self.target_size, self.target_size])

        self.loss_weght = tf.placeholder(
            tf.float32, [None, self.target_size, self.target_size])

        self.y_hat = self.network(self.x)

        self.loss = self.sample_loss(self.y, self.y_hat, self.loss_weght)

        self.saver = tf.train.Saver()

    def sample_loss(self, labels, logits, loss_weight):

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )

        loss = tf.losses.compute_weighted_loss(
            losses=loss, weights=loss_weight
        )

        return tf.reduce_mean(loss)

    def network(self, inputs):

        num_classes = self.num_classes
        train = self.is_training

        with tf.variable_scope('vgg'):

            conv1_1 = self._conv_layer(inputs, 64, 'conv1_1')
            conv1_2 = self._conv_layer(conv1_1, 64, 'conv1_2')
            pool1 = self._max_pool(conv1_2, 'pool1')

            conv2_1 = self._conv_layer(pool1, 128, 'conv2_1')
            conv2_2 = self._conv_layer(conv2_1, 128, 'conv2_2')
            pool2 = self._max_pool(conv2_2, 'pool2')

            conv3_1 = self._conv_layer(pool2, 256, 'conv3_1')
            conv3_2 = self._conv_layer(conv3_1, 256, 'conv3_2')
            conv3_3 = self._conv_layer(conv3_2, 256, 'conv3_3')
            pool3 = self._max_pool(conv3_3, 'pool3')

            conv4_1 = self._conv_layer(pool3, 512, 'conv4_1')
            conv4_2 = self._conv_layer(conv4_1, 512, 'conv4_2')
            conv4_3 = self._conv_layer(conv4_2, 512, 'conv4_3')
            pool4 = self._max_pool(conv4_3, 'pool4')

            conv5_1 = self._conv_layer(pool4, 512, 'conv5_1')
            conv5_2 = self._conv_layer(conv5_1, 512, 'conv5_2')
            conv5_3 = self._conv_layer(conv5_2, 512, 'conv5_3')
            pool5 = self._max_pool(conv5_3, 'pool5')

        fc6 = self._conv_layer(pool5, 4096, k_size=1, name='fc6')

        if train:
            fc6 = tf.nn.dropout(fc6, self.keep_rate)

        fc7 = self._conv_layer(fc6, 4096, k_size=1, name='fc7')

        if train:
            fc7 = tf.nn.dropout(fc7, self.keep_rate)

        fc8 = self._conv_layer(
            fc7, 1000, k_size=1, name='fc8')

        upscore1 = self._upscore_layer(fc8,
                                       shape=tf.shape(pool4),
                                       num_classes=num_classes,
                                       name='upscore1',
                                       ksize=4, stride=2)

        score_pool4 = self._conv_layer(
            pool4, num_classes, k_size=1, name='score_pool4')

        fuse_pool4 = tf.add(upscore1, score_pool4)
        # self.fuse_pool4 = self.score_pool4

        upscore2 = self._upscore_layer(fuse_pool4,
                                       shape=tf.shape(inputs),
                                       num_classes=num_classes,
                                       name='upscore2',
                                       ksize=32, stride=16)

        return tf.nn.softmax(upscore2, axis=-1)

    def _max_pool(self, bottom, name):

        pool = slim.max_pool2d(bottom, [2, 2], scope=name, padding='SAME')

        return pool

    def _conv_layer(self, bottom, filters, name, k_size=3):

        conv = slim.conv2d(bottom, filters, [
                           k_size, k_size], scope=name, padding='SAME')

        return conv

    def _upscore_layer(self, bottom, shape,
                       num_classes, name,
                       ksize=4, stride=2):

        strides = [1, stride, stride, 1]

        with tf.variable_scope(name):

            in_features = bottom.get_shape()[3].value

            if shape is None:
                # Compute shape out of Bottom
                in_shape = tf.shape(bottom)

                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, num_classes]

            else:
                new_shape = [shape[0], shape[1], shape[2], num_classes]

            output_shape = tf.stack(new_shape)

            f_shape = [ksize, ksize, num_classes, in_features]

            weights = self.get_deconv_filter(f_shape)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')

        return deconv

    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        height = f_shape[1]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name='up_filter', initializer=init,
                               shape=weights.shape)

    def train_net(self):

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.optimizer = tf.compat.v1.train.MomentumOptimizer(
            learning_rate=self.learning_rate, momentum=0.9)

        # self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)

        self.train_step = self.optimizer.minimize(self.loss)

        with tf.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(cfg.MODEL_PATH)

            if ckpt and ckpt.model_checkpoint_path:
                # 如果保存过模型，则在保存的模型的基础上继续训练
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model Reload Successfully!')

            for i in range(cfg.EPOCHES):
                loss_list = []
                for batch in range(cfg.BATCHES):

                    value = self.reader.generate(self.batch_size)

                    images = value['images']
                    labels = value['labels']
                    weights = value['weights']

                    feed_dict = {self.x: images,
                                 self.y: labels,
                                 self.loss_weght: weights}

                    _, loss = sess.run([self.train_step, self.loss], feed_dict)

                    loss_list.append(loss)

                    print('batch:{} loss:{}'.format(batch, loss), end='\r')

                loss_values = np.array(loss_list)  # (64, 3)

                loss_values = np.mean(loss_values)

                with open('./result.txt', 'a') as f:
                    f.write(str(loss_values)+'\n')

                self.saver.save(sess, os.path.join(
                    cfg.MODEL_PATH, 'model.ckpt'))

                print('epoch:{} loss:{}'.format(
                    self.reader.epoch, loss_values))

    def test(self, image_path):

        if not isinstance(image_path, list):
            image_path = [image_path]

        self.is_training = False

        with tf.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(cfg.MODEL_PATH)

            if ckpt and ckpt.model_checkpoint_path:
                # 如果保存过模型，则在保存的模型的基础上继续训练
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model Reload Successfully!')

            for path in image_path:

                image = self.reader.read_image(path)
                image, _ = self.reader.resize_image(
                    image, None, with_label=False)
                image, _ = random_crop(image, None, with_label=False)

                image = np.expand_dims(image, axis=0)

                label_image = sess.run(self.y_hat, feed_dict={
                                       self.x: image-self.reader.pixel_means})
                label_image = np.squeeze(image)

                label_image = np.argmax(label_image, axis=-1)
                label_image = self.reader.decode_label(label_image)

                image = np.squeeze(image).astype(np.int)

                result = np.vstack([image, label_image])

                plt.imshow(result)
                plt.show()


if __name__ == "__main__":

    net = Net(is_training=True)

    net.train_net()

    test_path = ['./0.jpg', './1.jpg']

    # net.test(test_path)
