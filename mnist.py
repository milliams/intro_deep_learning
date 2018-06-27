#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order
import numpy as np
import imageio
import os

import mnist_dataset as dataset
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers


LEARNING_RATE = 1e-4
INVERT = bool(int(os.environ.get('MNIST_INVERT', 0)))
print("INVERT:", INVERT)
BRIGHTNESS = False


def create_model(data_format):
  """Model to recognize digits in the MNIST dataset.

  Network structure is equivalent to:
  https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/mnist/mnist_deep.py
  and
  https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py

  But uses the tf.keras API.

  Args:
    data_format: Either 'channels_first' or 'channels_last'. 'channels_first' is
      typically faster on GPUs while 'channels_last' is typically faster on
      CPUs. See
      https://www.tensorflow.org/performance/performance_guide#data_formats

  Returns:
    A tf.keras.Model.
  """
  if data_format == 'channels_first':
    input_shape = [1, 28, 28]
  else:
    assert data_format == 'channels_last'
    input_shape = [28, 28, 1]

  l = tf.keras.layers
  max_pool = l.MaxPooling2D(
      (2, 2), (2, 2), padding='same', data_format=data_format)
  # The model consists of a sequential chain of layers, so tf.keras.Sequential
  # (a subclass of tf.keras.Model) makes for a compact description.
  return tf.keras.Sequential(
      [
          l.Reshape(
              target_shape=input_shape,
              input_shape=(28 * 28,)),
          l.Conv2D(
              32,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Conv2D(
              64,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Flatten(),
          l.Dense(1024, activation=tf.nn.relu),
          l.Dropout(0.4),
          l.Dense(10)
      ])


def define_mnist_flags():
  flags_core.define_base()
  flags_core.define_image()
  flags.adopt_module_key_flags(flags_core)
  flags_core.set_defaults(data_dir='/tmp/mnist_data',
                          batch_size=100,
                          train_epochs=40
                          )


def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  model = create_model(params['data_format'])
  image = features
  if isinstance(image, dict):
    image = features['image']

  if mode == tf.estimator.ModeKeys.PREDICT:
    logits = model(image, training=False)
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits),
    }
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

    # If we are running multi-GPU, we need to wrap the optimizer.
    if params.get('multi_gpu'):
      optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

    logits = model(image, training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=tf.argmax(logits, axis=1))

    # Name tensors to be logged with LoggingTensorHook.
    tf.identity(LEARNING_RATE, 'learning_rate')
    tf.identity(loss, 'cross_entropy')
    tf.identity(accuracy[1], name='train_accuracy')

    # Save accuracy scalar to Tensorboard output.
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))
  if mode == tf.estimator.ModeKeys.EVAL:
    logits = model(image, training=False)
    #print('######################################', labels.shape)
    #print('######################################', logits.shape)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=loss,
        eval_metric_ops={
            'accuracy':
                tf.metrics.accuracy(
                    labels=labels, predictions=tf.argmax(logits, axis=1)),
        })


def run_mnist(flags_obj):
  """Run MNIST training and eval loop.

  Args:
    flags_obj: An object containing parsed flag values.
  """
  model_helpers.apply_clean(flags_obj)
  model_function = model_fn

  # Get number of GPUs as defined by the --num_gpus flags and the number of
  # GPUs available on the machine.
  num_gpus = flags_core.get_num_gpus(flags_obj)
  multi_gpu = num_gpus > 1

  if multi_gpu:
    # Validate that the batch size can be split into devices.
    distribution_utils.per_device_batch_size(flags_obj.batch_size, num_gpus)

    # There are two steps required if using multi-GPU: (1) wrap the model_fn,
    # and (2) wrap the optimizer. The first happens here, and (2) happens
    # in the model_fn itself when the optimizer is defined.
    model_function = tf.contrib.estimator.replicate_model_fn(
        model_fn, loss_reduction=tf.losses.Reduction.MEAN,
        devices=["/device:GPU:%d" % d for d in range(num_gpus)])

  data_format = flags_obj.data_format
  if data_format is None:
    data_format = ('channels_first'
                   if tf.test.is_built_with_cuda() else 'channels_last')
  mnist_classifier = tf.estimator.Estimator(
      model_fn=model_function,
      params={
          'data_format': data_format,
          'multi_gpu': multi_gpu
      })

  # Set up training and evaluation input functions.
  def train_input_fn():
    """Prepare data for training."""

    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes use less memory. MNIST is a small
    # enough dataset that we can easily shuffle the full epoch.
    ds = dataset.train(flags_obj.data_dir)

    def invert(image, label):
      return (image * -1.0) + 1.0, label

    def brightness(image, label):
      return tf.image.random_brightness(image, max_delta=0.2), label 

    if INVERT:
      inverted = ds.map(invert)
      ds = ds.concatenate(inverted)

    if BRIGHTNESS:
      ds = ds.concatenate(ds.map(brightness)) 

    ds = ds.cache().shuffle(buffer_size=50000).batch(flags_obj.batch_size)

    # Iterate through the dataset a set number (`epochs_between_evals`) of times
    # during each training session.
    ds = ds.repeat(flags_obj.epochs_between_evals)
    return ds

  def eval_input_fn():
    return dataset.test(flags_obj.data_dir).batch(
        flags_obj.batch_size).make_one_shot_iterator().get_next()

  # Set up hook that outputs training logs every 100 steps.
  train_hooks = hooks_helper.get_train_hooks(
      flags_obj.hooks,
      batch_size=flags_obj.batch_size)

  # Train and evaluate model.
  for _ in range(flags_obj.train_epochs // flags_obj.epochs_between_evals):
    mnist_classifier.train(input_fn=train_input_fn, hooks=train_hooks)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print('\nEvaluation results:\n\t%s\n' % eval_results)

    if model_helpers.past_stop_threshold(flags_obj.stop_threshold,
                                         eval_results['accuracy']):
      break

  # Export the model
  if flags_obj.export_dir is not None:
    image = tf.placeholder(tf.float32, [None, 28, 28])
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'image': image,
    })
    mnist_classifier.export_savedmodel(flags_obj.export_dir, input_fn)

  def our_test_fn():
    images = []
    for i in list(range(1,10)) + ['dog']:
      images.append(np.array(imageio.imread('{}.png'.format(i)).ravel()/255.0, dtype='float32'))
    images = np.array(images)
    return tf.convert_to_tensor(images)

  # Check our own examples
  predictions = mnist_classifier.predict(input_fn=our_test_fn)
  table = []
  for i in list(range(1, 10)) + ['dog']:
    prediction = next(predictions)
    if i == 'dog':
      print("{}. CNN thinks it's a {} ({:.1f}%)".format(i, prediction['classes'], prediction['probabilities'][prediction['classes']]*100))
    else:
      print("{} at {:.1f}. CNN thinks it's a {} ({:.1f}%)".format(i, prediction['probabilities'][i]*100, prediction['classes'], prediction['probabilities'][prediction['classes']]*100))
    table.append((i, prediction['probabilities']))
  #for i, l in table:
  #  print("<tr>")
  #  print('<td><img src="{}.png" style="margin: 1px 0px"></td>'.format(i))
  #  for m in l:
  #      print("<td>{}%</td>".format(int(round(m*100))))
  #  print("</tr>")

def main(_):
  run_mnist(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_mnist_flags()
  absl_app.run(main)
