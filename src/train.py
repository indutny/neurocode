import os
import tensorflow as tf

from args import parse_args
from model import Model

RUN_NAME, CONFIG, args = parse_args(kind='train')
print('Booting up {}'.format(RUN_NAME))
print('config', CONFIG)

LOG_DIR = os.path.join('.', 'logs', RUN_NAME)
SAVE_DIR = os.path.join('.', 'saves', RUN_NAME)

BATCH_SIZE = 32
SAVE_EVERY = 100
VALIDATE_EVERY = 10

writer = tf.summary.FileWriter(LOG_DIR)

def gen_data():
  batch_noise = tf.random.uniform([ BATCH_SIZE, 32, 32, 8 ])
  return tf.cast(batch_noise > 0.5, dtype=tf.float32)

with tf.Session() as sess:
  optimizer = tf.train.AdamOptimizer(CONFIG['lr'])

  # Steps
  epoch = tf.Variable(name='epoch', initial_value=0, dtype=tf.int32)
  epoch_inc = epoch.assign_add(1)

  global_step = tf.Variable(name='global_step', initial_value=0, dtype=tf.int32)

  print('Initializing model')
  model = Model()

  training_data = gen_data()
  validation_data = gen_data()

  training_encoding = model.encode(training_data, training=True)
  training_prediction = model.decode(training_encoding, training=True)

  validation_encoding = model.encode(validation_data, training=False)
  validation_prediction = model.decode(validation_encoding, training=False)

  loss, training_metrics = \
      model.loss_and_metrics(predictions=training_prediction, \
          labels=training_data)
  _, validation_metrics = \
      model.loss_and_metrics(predictions=validation_prediction, \
          labels=validation_data, tag='val')
  train = optimizer.minimize(loss, global_step)

  # Sample image
  image = tf.summary.image('sample',
      model.encode(validation_data[:1], training=False))
  validation_metrics = tf.summary.merge([ validation_metrics, image ])

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

    _, metrics_v = sess.run([ train, training_metrics ])
    writer.add_summary(metrics_v, epoch_value)
    writer.flush()

    if epoch_value % VALIDATE_EVERY == 0:
      metrics_v = sess.run(validation_metrics)
      writer.add_summary(metrics_v, epoch_value)
      writer.flush()

    if epoch_value % SAVE_EVERY == 0:
      saver.save(sess, os.path.join(SAVE_DIR, '{:08d}'.format(epoch_value)))
