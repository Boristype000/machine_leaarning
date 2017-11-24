# -*- coding: utf-8 -*-
# Project: maxent-ml
# Author: chaoxu create this file
# Time: 2017/11/4
# Company : Maxent
# Email: chao.xu@maxent-inc.com
"""
this file is used to run cifar10 on meituan yuan
in distribution but only use cpus
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cifar10


def tower_loss(scope, images, labels):
    """Calculate the total loss on a single tower running the CIFAR model.

    Args:
      scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
      images: Images. 4D tensor of shape [batch_size, height, width, 3].
      labels: Labels. 1D tensor of shape [batch_size].

    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """

    # Build inference Graph.
    logits = cifar10.inference(images)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = cifar10.loss(logits, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    """Train CIFAR-10 for a number of steps."""
    print("will use {0} gpus".format(FLAGS.num_gpus))
    ps_hosts = FLAGS.ps_hosts.split(",")

    worker_hosts = FLAGS.worker_hosts.split(",")

    # 创建TensorFlow集群描述对象

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    # 如果是参数服务，直接启动即可

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        # with tf.Graph().as_default(), tf.device('/cpu:0'):
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,
                                                      cluster=cluster)):
            # Create a variable to count the number of train() calls. This equals the
            # number of batches processed * FLAGS.num_gpus.
            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)

            # Calculate the learning rate schedule.
            num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                     FLAGS.batch_size)
            decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)

            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                            global_step,
                                            decay_steps,
                                            cifar10.LEARNING_RATE_DECAY_FACTOR,
                                            staircase=True)

            # Create an optimizer that performs gradient descent.
            opt = tf.train.GradientDescentOptimizer(lr)

            with tf.name_scope('input'):
                image_batch = tf.placeholder(dtype=tf.float32,
                                             shape=[FLAGS.batch_size, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, 3],
                                             name="image-input")
                label_batch = tf.placeholder(dtype=tf.float32,
                                             shape=(FLAGS.batch_size,),
                                             name='label-input')

            # Calculate the gradients for each model tower.
            tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in xrange(FLAGS.num_gpus):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
                            # Dequeues one batch for the GPU
                            # image_batch, label_batch = batch_queue.dequeue()
                            # Calculate the loss for one tower of the CIFAR model. This function
                            # constructs the entire CIFAR model but shares the variables across
                            # all towers.
                            loss = tower_loss(scope, image_batch, label_batch)

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

                            # Retain the summaries from the final tower.
                            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                            # Calculate the gradients for the batch of data on this CIFAR tower.
                            grads = opt.compute_gradients(loss)

                            # Keep track of the gradients across all towers.
                            tower_grads.append(grads)

            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            grads = average_gradients(tower_grads)

            # Add a summary to track the learning rate.
            summaries.append(tf.summary.scalar('learning_rate', lr))

            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

            # Apply the gradients to adjust the shared variables.
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))

            # Track the moving averages of all trainable variables.
            variable_averages = tf.train.ExponentialMovingAverage(
                cifar10.MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            # Group all updates to into a single train op.
            train_op = tf.group(apply_gradient_op, variables_averages_op)

            # Create a saver.
            # saver = tf.train.Saver(tf.global_variables())

            # Build the summary operation from the last tower summaries.
            summary_op = tf.summary.merge(summaries)

            # Build an initialization operation to run below.
            init_op = tf.global_variables_initializer()

        # 读入MNIST训练数据集
        images, labels = cifar10.distorted_inputs()
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images, labels], capacity=2 * FLAGS.num_gpus)

        # sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
        #                          logdir=FLAGS.tb_dir,
        #                          init_op=init_op,
        #                          summary_op=summary_op,
        #                          global_step=global_step
        #                          # save_model_secs=600
        #                      )
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 global_step=global_step,
                                 init_op=init_op)

        # 创建TensorFlow session对象，用于执行TensorFlow图计算

        with sv.prepare_or_wait_for_session(master=server.target,
                                            config=tf.ConfigProto(allow_soft_placement=True,
                                                              log_device_placement=FLAGS.log_device_placement)) as sess:
            step = 0

            while not sv.should_stop() and step < FLAGS.max_steps:

                images_batch_, labels_batch_ = batch_queue.dequeue()

                train_feed = {image_batch: images_batch_, label_batch: labels_batch_}

                # 执行分布式TensorFlow模型训练
                summary_writer = tf.summary.FileWriter(FLAGS.tb_dir, graph=tf.get_default_graph())

                for step in xrange(FLAGS.max_steps):
                    start_time = time.time()
                    _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict=train_feed)
                    duration = time.time() - start_time

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    if step % 10 == 0:
                        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration / FLAGS.num_gpus
                    format_str = "%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ''sec/batch)"
                    print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

                    if step % 100 == 0:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)
        sv.stop()
        print("done")


def main(_):
    train()

if __name__ == '__main__':
    parser = cifar10.parser

    parser.add_argument('--train_dir', type=str, default='/tmp/cifar10_train',
                        help='Directory where to write event logs and checkpoint.')

    parser.add_argument('--model_dir', type=str, default='/tmp/cifar10_model',
                        help='Directory where to save model.')

    parser.add_argument('--tb_dir', type=str, default='/tmp/cifar10_tab',
                        help='Directory where to write event logs and checkpoint.')

    parser.add_argument('--max_steps', type=int, default=1000000,
                        help='Number of batches to run.')

    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Whether to log device placement.')

    parser.add_argument('--log_frequency', type=int, default=10,
                        help='How often to log results to the console.')

    parser.add_argument('--num_gpus', type=int, default=1,
                        help='define how many gpus can be used')

    # TensorFlow集群描述信息，ps_hosts表示参数服务节点信息，worker_hosts表示worker节点信息

    parser.add_argument("--ps_hosts", type=str, default="", help="Comma-separated list of hostname:port pairs")
    parser.add_argument("--worker_hosts", type=str, default="", help="Comma-separated list of hostname:port pairs")

    # TensorFlow Server模型描述信息，包括作业名称，任务编号，隐含层神经元数量，MNIST数据目录以及每次训练数据大小（默认一个批次为100个图片）

    parser.add_argument("--job_name", type=str, default="", help="One of 'ps', 'worker'")
    parser.add_argument("--task_index", type=int, default=0, help="Index of task within the job")

    FLAGS, _ = parser.parse_known_args()

    tf.app.run()

