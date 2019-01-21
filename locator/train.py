import os
import math
import tensorflow as tf

from args import parse_args
from model import Model

MARKER_FILE = os.path.join(os.path.dirname(__file__), 'marker.png')

RUN_NAME, CONFIG, args = parse_args(kind='train')
print('Booting up {}'.format(RUN_NAME))
print('config', CONFIG)

LOG_DIR = os.path.join('.', 'logs', RUN_NAME)
SAVE_DIR = os.path.join('.', 'saves', RUN_NAME)

BATCH_SIZE = 1024
SAVE_EVERY = 100
VALIDATE_EVERY = 10

writer = tf.summary.FileWriter(LOG_DIR)

def gen_data(marker, batch_size = BATCH_SIZE):
  height = tf.shape(marker)[0]
  width = tf.shape(marker)[1]
  f_height = tf.cast(height, dtype=tf.float32)
  f_width = tf.cast(width, dtype=tf.float32)

  marker = tf.tile(tf.expand_dims(marker, axis=0), [ batch_size, 1, 1, 1 ])

  size = tf.random.uniform([ batch_size, 1 ], minval=0.4, maxval=1.0,
      name='l_size')
  pos = tf.random.uniform([ batch_size, 2 ], name='l_pos') * \
      (1.0 - size)
  angle = tf.random.uniform([ batch_size, 1 ], minval=0, maxval=math.pi / 2.0,
      name='l_angle')
  present = tf.cast(
      tf.random.uniform([ batch_size, 1 ], name='l_pos') > 0.5,
      dtype=tf.float32)

  # Helpers
  one = tf.ones_like(size)
  zero = tf.zeros_like(size)

  size_transform = tf.concat(
      [ 1.0 / size, zero, zero, zero, 1.0 / size, zero, zero, zero ], axis=-1)

  # NOTE: Centered at full size, between 0.125 and 0.875
  pos_x, pos_y = tf.split(pos, [ 1, 1 ], axis=-1)

  pos_transform = tf.concat(
      [ one, zero, -pos_x * f_width, zero, one, -pos_y * f_height, zero, zero ],
      axis=-1)

  transform = tf.contrib.image.compose_transforms(
      size_transform, pos_transform)

  # Note: -1.0 pushes black to -1.0 and white to 0.0, making
  # `tf.contrib.image.transform` pad with white
  images = tf.contrib.image.transform(marker - 1.0, transform,
      interpolation='BILINEAR') + 0.5

  # Add some contrast noise
  contrast = tf.exp(tf.random.normal(tf.shape(images), \
      mean=0.0, stddev=0.18232155))
  images *= contrast

  # Add some noise
  noise = tf.random_uniform(tf.shape(images), minval=-0.5, maxval=0.5)
  images += noise

  e_present = tf.expand_dims(present, axis=-1)
  e_present = tf.expand_dims(e_present, axis=-1)
  images = e_present * images + \
      (1.0 - e_present) * (noise + tf.random_uniform([ batch_size, 1, 1, 1 ]))

  images = tf.clip_by_value(images, -0.5, 0.5)

  norm_pos = pos - (1.0 - size) / 2.0

  labels = tf.concat([ present, norm_pos, size ], axis=-1)
  return images, labels

with tf.Session() as sess:
  marker = sess.run(tf.image.decode_image(tf.read_file(MARKER_FILE),
    channels=1)) / 255.0
  marker = tf.constant(marker, dtype=tf.float32, name='marker')

  optimizer = tf.train.AdamOptimizer(CONFIG['lr'])

  # Steps
  epoch = tf.Variable(name='epoch', initial_value=0, dtype=tf.int32)
  epoch_inc = epoch.assign_add(1)

  global_step = tf.Variable(name='global_step', initial_value=0, dtype=tf.int32)

  print('Initializing model')
  model = Model()

  data, labels = gen_data(marker)

  prediction = model.forward(data, training=True)

  loss, metrics = \
      model.loss_and_metrics(predictions=prediction, labels=labels)
  train = optimizer.minimize(loss, global_step)

  # Sample image
  image = tf.summary.image('input',
      tf.cast((data[:1] + 0.5) * 255.0, dtype=tf.uint8))
  validation_metrics = tf.summary.merge([ image ])

  writer.add_graph(tf.get_default_graph())
  saver = tf.train.Saver(max_to_keep=100, name=RUN_NAME)

  sess.run(tf.global_variables_initializer())
  sess.graph.finalize()

  total_size = 0
  for var in tf.trainable_variables():
    size = 1
    for layer_size in var.shape:
      size *= layer_size
    total_size += size
    print('name: {} shape: {} size: {}'.format(var.name, var.shape, size))
  print('total size: {}'.format(total_size))

  epoch_value = sess.run(epoch)
  while True:
    epoch_value = sess.run(epoch_inc)
    print('Epoch {}'.format(epoch_value))

    _, metrics_v = sess.run([ train, metrics ])
    writer.add_summary(metrics_v, epoch_value)
    writer.flush()

    if epoch_value % VALIDATE_EVERY == 0:
      metrics_v = sess.run(validation_metrics)
      writer.add_summary(metrics_v, epoch_value)
      writer.flush()

    if epoch_value % SAVE_EVERY == 0:
      saver.save(sess, os.path.join(SAVE_DIR, '{:08d}'.format(epoch_value)))
