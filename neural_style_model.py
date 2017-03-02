# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc

try:
    reduce
except NameError:
    from functools import reduce

class NS_model(object):
    """ Neural style model -- re-write by 蔡浩

    """
    def __init__(self, vgg_net, content, style, iterations, save_dir, output_dir,
                 content_weight, style_weight, tv_weight, learning_rate):

        self.vgg_net = vgg_net  # 应该直接传入， 不在这个类里面加载vgg模型
        self.vgg_mean_pixel = np.mean(self.vgg_net['normalization'][0][0][0],
                                      axis=(0, 1)).reshape((1, 1, 1, 3))
        # 改变content 和 style 的shape， 使得能够喂进网络中
        content = np.reshape(content, (1,) + content.shape)
        style = np.reshape(style, (1,) + style.shape)
        self.content = content
        self.style = style

        self.iteration_num = iterations
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.learning_rate = learning_rate

        self.content_layer = 'relu4_2'
        self.style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        self.save_dir = save_dir
        self.output_dir = output_dir


    def loss(self):
        # TODO:
        #g = tf.get_default_graph()


        with tf.Graph().as_default(),tf.device("/cpu:0"), tf.name_scope("content_feature"), tf.Session() as sess:
            #content = tf.constant(self.content - self.vgg_mean_pixel, name="constant_image", dtype=tf.float32)
            content_holder = tf.placeholder(tf.float32, shape=self.content.shape, name="content_holder")
            content_feeder = self.content - self.vgg_mean_pixel
            content_net = self.vgg_go(content_holder)
            content_feature = content_net[self.content_layer].eval(session=sess, feed_dict={content_holder: content_feeder})

        with tf.Graph().as_default(),tf.device("/cpu:0"), tf.name_scope("style_feature"), tf.Session() as sess:

            # style = tf.constant(self.style - self.vgg_mean_pixel, name="style_image", dtype=tf.float32)
            style_holder = tf.placeholder(tf.float32, shape=self.style.shape, name="style_holder")
            style_feeder = self.style - self.vgg_mean_pixel
            style_net = self.vgg_go(style_holder)

            style_features = [style_net[feature].eval(session=sess, feed_dict={style_holder: style_feeder}) for feature in self.style_layers]
            style_grams = []
            for feature in style_features:
                feature = np.reshape(feature, (-1, feature.shape[-1]))
                # 最后一维是输出信道数， 也是这一层的卷积核数
                gram = np.matmul(feature.T, feature) // feature.size
                style_grams.append(gram)


        with tf.Graph().as_default(), tf.variable_scope("train"):
            self.train_image = tf.get_variable(name='train_image', shape=(1, 400, 600, 3),
                                               initializer=tf.random_normal_initializer())

            train_net = self.vgg_go(self.train_image)
            train_content_feature = train_net[self.content_layer]
            train_style_features = [train_net[style] for style in self.style_layers]
            train_style_grams = []
            for tsf in train_style_features:
                _, height, width, N = tsf.get_shape().as_list()
                tsf = tf.reshape(tsf, (-1, N))
                gram = tf.matmul(tf.transpose(tsf), tsf) // (height * width * N)
                train_style_grams.append(gram)

            content_loss = tf.nn.l2_loss(train_content_feature - content_feature, name="content_loss")
            with tf.name_scope("style_loss"):
                style_loss = reduce(tf.add,
                                    [tf.nn.l2_loss(style_grams[i] - train_style_grams[i])/4 for i in range(len(self.style_layers))])
            with tf.name_scope("TV_loss"):  # TV_loss : total variation denoising (一种图像降噪的算法)
                tv_H = self.train_image[:, 1:, :, :]  # 变分法公式，沿Height方向的term
                tv_W = self.train_image[:, :, 1:, :]
                tv_H_size = reduce(np.multiply, tv_H.get_shape().as_list())
                tv_W_size = reduce(np.multiply, tv_W.get_shape().as_list())
                tv_loss = tf.add( tf.nn.l2_loss(tf.sub(self.train_image[:, :-1, :, :], tv_H)) / tv_H_size,
                                  tf.nn.l2_loss(tf.sub(self.train_image[:, :, :-1, :], tv_W)) / tv_W_size)

            self.total_loss = self.content_weight * content_loss + self.style_weight * style_loss + \
                         self.tv_weight * tv_loss
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)
            tf.scalar_summary("total_loss", self.total_loss)
            tf.scalar_summary("content_loss", content_loss)
            tf.scalar_summary("style_loss", style_loss)



    def train(self):
        self.loss()
        graph = self.total_loss.graph
        with graph.as_default():
            self.global_step = tf.Variable(1, name='global_step')
            update_global_step = tf.assign_add(self.global_step, 1)
            initializer = tf.global_variables_initializer()
            optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            sv = tf.train.Supervisor(logdir=self.save_dir, global_step=self.global_step)
            #with sv.managed_session(config=sess_config) as sess:
            with tf.Session() as sess:
                initializer.run()
                for i in xrange(self.iteration_num) :
                    sess.run(self.train_op)
                    if ( i + 1 ) % 50 == 0:
                        print "%d-th_total_loss: %d" %(i, self.total_loss.eval() )
                        sess.run(update_global_step)
                        image = self.train_image.eval(session=sess) + self.vgg_mean_pixel
                        scipy.misc.imsave("%d-iterations.jpg" % (i), image)


    def vgg_go(self, image):
        """ image will go through vgg19 except fully-connection layers."""
        layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )
        weights = self.vgg_net['layers'][0]
        current = image
        net = {}
        for i, name in enumerate(layers):
            kind = name[:4]
            if kind == 'conv':
                # just use it in this way
                kernels, bias = weights[i][0][0][0][0]

                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: [height, width, in_channels, out_channels]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                # bias.shape = (len(bias), [x, y, ... w, v]
                conv = tf.nn.conv2d(current, tf.constant(kernels), name=name,
                                       strides=(1, 1, 1, 1), padding='SAME')
                current = tf.nn.bias_add(conv, bias)
            elif kind == 'relu':
                current = tf.nn.relu(current, name=name)
            elif kind == 'pool':
                current = tf.nn.max_pool(current, ksize=(1, 2, 2, 1), name=name,
                                         strides=(1, 2, 2, 1), padding='SAME')
            net[name] = current
        return net




