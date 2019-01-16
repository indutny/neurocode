import tensorflow as tf
import math

class Model:
  def __init__(self):
    pass

  def encode(self, data, training):
    with tf.variable_scope('neurocode', reuse=tf.AUTO_REUSE, values=[data]):
      x = data

      x = self.deconv2d(x, 32, 3, 1, name='encode_1', training=training)
      x = self.deconv2d(x, 16, 3, 2, name='encode_2', training=training)
      x = self.deconv2d(x, 8, 3, 1, name='encode_3', training=training)
      x = self.deconv2d(x, 2, 1, 1, name='encode_4', training=training,
          activation=tf.nn.l2_normalize, bn=False)

      return x

  def to_image(self, image):
    sin, cos = tf.split(image, [ 1, 1 ], axis=-1)
    angle = tf.math.atan2(sin, cos)
    hue = (angle / math.pi + 1.0) / 2.0
    pad = tf.ones_like(hue)
    hsv = tf.concat([ hue, pad / 1.5, pad ], axis=-1)
    return tf.image.hsv_to_rgb(hsv)

  def decode(self, image, training):
    with tf.variable_scope('neurocode', reuse=tf.AUTO_REUSE, values=[image]):
      x = image

      f_training = tf.cast(training, dtype=tf.float32)


      x = self.conv2d(x, 8, 3, 1, name='decode_4', training=training)
      x = self.conv2d(x, 16, 3, 2, name='decode_3', training=training)
      x = self.conv2d(x, 32, 3, 1, name='decode_2', training=training)
      x = self.conv2d(x, 16, 1, 1, name='decode_1', training=training,
          activation=None, bn=False)

      return x

  def loss_and_metrics(self, predictions, labels, tag='train'):
    int_labels = tf.cast(labels, dtype=tf.int32)
    binary_labels = tf.one_hot(int_labels, 2, axis=-1)
    binary_predictions = tf.reshape(predictions, tf.shape(binary_labels))

    loss = tf.losses.softmax_cross_entropy(onehot_labels=binary_labels,
        logits=binary_predictions)

    int_predictions = tf.argmax(binary_predictions, axis=-1)
    int_predictions = tf.cast(int_predictions, dtype=tf.int32)

    accuracy = tf.cast(
        tf.equal(int_predictions, int_labels), dtype=tf.float32)
    accuracy = tf.reduce_mean(accuracy)

    metrics = [
      tf.summary.scalar('{}/loss'.format(tag), loss),
      tf.summary.scalar('{}/accuracy'.format(tag), accuracy),
    ]
    return loss, tf.summary.merge(metrics)

  def conv2d(self, x, filters, size, strides, name, training,
             activation=tf.nn.relu, bn=True):
    x = tf.layers.conv2d(x, filters=filters, kernel_size=size,
        padding='SAME',
        strides=(strides, strides), name='{}_conv2d'.format(name))
    if not activation is None:
      x = activation(x)
    return x

  def deconv2d(self, x, filters, size, strides, name, training,
      activation=tf.nn.relu, bn=True):
    x = tf.layers.conv2d_transpose(x, filters=filters, kernel_size=size,
        padding='SAME',
        strides=(strides, strides), name='{}_deconv2d'.format(name))
    if not activation is None:
      x = activation(x)
    return x
