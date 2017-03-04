import tensorflow as tf
import os
import scipy.misc
import scipy.io
from neural_style_model import NS_model

flags = tf.app.flags
flags.DEFINE_string("VGG_PATH", "imagenet-vgg-verydeep-19.mat",
                    "path of pre-trained vgg19")
flags.DEFINE_string("output_dir", "./output_imgs_gakki/",
                    "dir to store neural-style imgs")
flags.DEFINE_string("content_path", "./src_imgs/1-content.jpg",
                    "path of content image.")
flags.DEFINE_string("style_path", "./src_imgs/gakki.jpg",
                    "path of style image.")
flags.DEFINE_string("save_dir", "./log/",
                    "dir to store tensorboard log and checkpoints")
flags.DEFINE_float("learning_rate", 10,
                   "Learning rate of the model.")
flags.DEFINE_integer("STYLE_WEIGHT", 100,
                     "weight of style loss in total loss.")
flags.DEFINE_integer("TV_WEIGHT", 100,
                     "weight of TV loss in total loss.")
flags.DEFINE_integer("CONTENT_WEIGHT", 5,
                     "weight of content loss in total loss.")
flags.DEFINE_integer("iteration_num", 1000,
                     "number of iterations to run.")
FLAGS = flags.FLAGS


def main(_):
    for path in FLAGS.save_dir, FLAGS.output_dir:
        if not os.path.exists(path):
            os.makedirs(path)

    if FLAGS.style_path is None or FLAGS.content_path is None:
        raise ValueError("Please set both style image and content image.")
    else:
        content = scipy.misc.imread(FLAGS.content_path)
        style = scipy.misc.imread(FLAGS.style_path)

    vgg_net = scipy.io.loadmat(FLAGS.VGG_PATH)
    model = NS_model(vgg_net=vgg_net, learning_rate=FLAGS.learning_rate, save_dir=FLAGS.save_dir, content=content,
                     style=style, content_weight=FLAGS.CONTENT_WEIGHT, style_weight=FLAGS.STYLE_WEIGHT,
                     iterations=FLAGS.iteration_num, tv_weight=FLAGS.TV_WEIGHT, output_dir=FLAGS.output_dir)
    model.train()




if __name__ == '__main__':
    tf.app.run()