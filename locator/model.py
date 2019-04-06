import tensorflow as tf

class Model:
  def __init__(self):
    pass

  def forward(self, data, training):
    with tf.variable_scope('neurocode', reuse=tf.AUTO_REUSE, values=[data]):
      x = data

      def img(name, t):
        data = tf.concat([
          tf.reduce_min(t, axis=-1, keepdims=True),
          tf.reduce_mean(t, axis=-1, keepdims=True),
          tf.reduce_max(t, axis=-1, keepdims=True)
        ], axis=-1)
        return tf.summary.image(name, data)

      x = self.conv2d(x, 16, 3, 1, name='locate_1', training=training)
      x = self.pool(x, 2, 2, name='locate_1')
      s1 = img('s1', x)
      x = self.conv2d(x, 32, 3, 1, name='locate_2', training=training)
      x = self.pool(x, 2, 2, name='locate_2')
      s2 = img('s2', x)
      x = self.conv2d(x, 64, 3, 1, name='locate_3', training=training)
      x = self.pool(x, 2, 2, name='locate_3')
      s3 = img('s3', x)
      x = self.conv2d(x, 128, 3, 1, name='locate_4', training=training)
      x = self.pool(x, 2, 2, name='locate_4')
      s4 = img('s4', x)
      x = self.conv2d(x, 256, 3, 1, name='locate_5', training=training)
      x = self.pool(x, 2, 2, name='locate_5')
      s5 = img('s5', x)
      x = self.conv2d(x, 512, 3, 1, name='locate_6', training=training)
      x = self.pool(x, 2, 2, name='locate_6')
      s6 = img('s6', x)

      # confidence, x, y, size
      x = self.conv2d(x, 5, 1, 1, name='locate_final', training=training,
          activation=None, bn=False)
      confidence, pos, size = tf.split(x, [ 2, 2, 1 ], axis=-1)

      pos = tf.tanh(pos, name='pos')
      size = tf.exp(size, name='size')

      return tf.concat([ confidence, pos, size ], axis=-1), \
          tf.summary.merge([ s1, s2, s3, s4, s5, s6 ])

  def loss_and_metrics(self, predictions, labels, tag='train'):
    labels = tf.expand_dims(labels, axis=1)

    confidence, pos, size = tf.split(predictions, [ 2, 2, 1 ], axis=-1)
    l_present, l_pos, l_size = tf.split(labels, [ 1, 2, 1 ], axis=-1)
    l_int_present = tf.cast(l_present, dtype=tf.int32)

    confidence_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=tf.one_hot(l_int_present, 2, axis=-1),
        logits=confidence)

    pos_loss = tf.reduce_mean(((pos - l_pos) ** 2) * l_present)
    size_loss = tf.reduce_mean(
        ((size - l_size) ** 2) * l_present)

    loss = confidence_loss + 3.0 * (pos_loss + size_loss)

    metrics = [
      tf.summary.scalar('{}/loss'.format(tag), loss),
      tf.summary.scalar('{}/confidence_loss'.format(tag), confidence_loss),
      tf.summary.scalar('{}/pos_loss'.format(tag), pos_loss),
      tf.summary.scalar('{}/size_loss'.format(tag), size_loss),
    ]
    return loss, tf.summary.merge(metrics)

  def conv2d(self, x, filters, size, strides, name, training,
             activation=tf.nn.relu, bn=True):
    x = tf.layers.conv2d(x, filters=filters, kernel_size=size,
        padding='SAME',
        strides=(strides, strides), name='{}_conv2d'.format(name))
    if not activation is None:
      x = tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5,
          training=training,
          name='{}_bn'.format(name))
      x = activation(x)
    return x

  def pool(self, x, size, strides, name):
    return tf.layers.max_pooling2d(x, pool_size=size, strides=strides,
        name='{}_mp'.format(name))
