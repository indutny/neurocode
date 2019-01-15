import tensorflow as tf

class Model:
  def __init__(self):
    pass

  def encode(self, data, training):
    with tf.variable_scope('neurocode', reuse=tf.AUTO_REUSE, values=[data]):
      x = data

      x = self.deconv2d(x, 2, 3, 2, name='encode_1', training=training)
      x = self.deconv2d(x, 4, 3, 2, name='encode_2', training=training)
      x = self.deconv2d(x, 1, 1, 1, name='encode_3', training=training,
          activation=tf.nn.sigmoid, bn=False)

      return x

  def decode(self, image, training):
    with tf.variable_scope('neurocode', reuse=tf.AUTO_REUSE, values=[image]):
      x = image

      if False:
        f_training = tf.cast(training, dtype=tf.float32)
        contrast = 1.9 * tf.random.uniform([ tf.shape(image)[0], 1, 1, 1 ]) + 0.1
        x -= 0.5
        x *= contrast * f_training + (1.0 - f_training)
        x += 0.5

      # x = tf.layers.dropout(x, training=training, rate=0.8)
      x = self.conv2d(x, 2, 1, 1, name='decode_3', training=training)
      x = self.conv2d(x, 4, 3, 2, name='decode_2', training=training)
      x = self.conv2d(x, 1, 3, 2, name='decode_1', training=training,
          activation=tf.nn.sigmoid, bn=False)

      return x

  def loss_and_metrics(self, predictions, labels, tag='train'):
    loss = tf.losses.mean_squared_error(predictions=predictions, labels=labels)

    metrics = [
      tf.summary.scalar('{}/loss'.format(tag), loss),
    ]
    return loss, tf.summary.merge(metrics)

  def conv2d(self, x, filters, size, strides, name, training,
             activation=tf.nn.relu, bn=True):
    x = tf.layers.conv2d(x, filters=filters, kernel_size=size,
        padding='SAME',
        strides=(strides, strides), name='{}_conv2d'.format(name))
    if bn:
      x = tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5,
          training=training, name='{}_bn'.format(name))
    if not activation is None:
      x = activation(x)
    return x

  def deconv2d(self, x, filters, size, strides, name, training,
      activation=tf.nn.relu, bn=True):
    x = tf.layers.conv2d_transpose(x, filters=filters, kernel_size=size,
        padding='SAME',
        strides=(strides, strides), name='{}_deconv2d'.format(name))
    if bn:
      x = tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5,
          training=training, name='{}_bn'.format(name))
    if not activation is None:
      x = activation(x)
    return x
