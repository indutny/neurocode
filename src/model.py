import tensorflow as tf

class Model:
  def __init__(self):
    pass

  def encode(self, data, training):
    with tf.variable_scope('neurocode', reuse=tf.AUTO_REUSE, values=[data]):
      x = data

      x = self.deconv2d(x, 8, 4, 2, name='encode_1', training=training)
      x = self.deconv2d(x, 8, 4, 2, name='encode_2', training=training)
      x = self.deconv2d(x, 16, 4, 2, name='encode_3', training=training)
      x = self.deconv2d(x, 1, 1, 1, name='encode_4', training=training,
          activation=tf.nn.sigmoid, bn=False)

      return x

  def decode(self, image, training):
    with tf.variable_scope('neurocode', reuse=tf.AUTO_REUSE, values=[image]):
      x = image

      f_training = tf.cast(training, dtype=tf.float32)
      contrast = tf.exp(tf.random.normal(tf.shape(x), \
          mean=0.0, stddev=0.223143))
      x -= 0.5
      x *= contrast * f_training + (1.0 - f_training)
      x += 0.5

      x = self.conv2d(x, 16, 4, 2, name='decode_4', training=training)
      x = self.conv2d(x, 8, 4, 2, name='decode_3', training=training)
      x = self.conv2d(x, 8, 4, 2, name='decode_2', training=training)
      x = self.conv2d(x, 8, 1, 1, name='decode_1', training=training,
          activation=tf.nn.sigmoid, bn=False)

      return x

  def loss_and_metrics(self, predictions, labels, tag='train'):
    loss = tf.losses.mean_squared_error(predictions=predictions, labels=labels)

    accuracy = tf.cast(
        tf.equal(predictions > 0.5, labels > 0.5), dtype=tf.float32)
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
