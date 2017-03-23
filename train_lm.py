#!/usr/bin/python -u
import os
import time
import math

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.framework import ops, dtypes, sparse_tensor
from tensorflow.python.ops import init_ops, array_ops, math_ops, nn, variable_scope, sparse_ops


from reader import *

flags = tf.flags
flags.DEFINE_string("data_path", "ptb_data",
                    "Where the training/test data is stored.")
flags.DEFINE_bool("use_adaptive_softmax", True, "Train using adaptive softmax")

FLAGS = flags.FLAGS


def adaptive_softmax_loss(inputs,
                          labels,
                          cutoff,
                          project_factor=4,
                          initializer=None,
                          name=None):
    """Computes and returns the adaptive softmax loss (a improvement of
    hierarchical softmax).

    See [Efficient softmax approximation for GPUs](https://arxiv.org/pdf/1609.04309v2.pdf).

    This is a faster way to train a softmax classifier over a huge number of
    classes, and can be used for **both training and prediction**. For example, it
    can be used for training a Language Model with a very huge vocabulary, and
    the trained languaed model can be used in speech recognition, text generation,
    and machine translation very efficiently.

    Args:
      inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
      labels: `Tensor` of shape `[d_0, d_1, ..., d_{r-2}]` and dtype `int32` or
        `int64`. Each entry in `labels` must be an index in `[0, num_classes)`.
      cutoff: A list indicating the limits of the different clusters.
      project_factor: A floating point value greater or equal to 1.0. The projection
        factor between two neighboring clusters.
      initializer: Initializer for adaptive softmax variables (optional).
      name: A name for the operation (optional).
    Returns:
      loss: A `batch_size` 1-D tensor of the adaptive softmax cross entropy loss.
      training_losses: A list of 1-D tensors of adaptive softmax loss for each
        cluster, which can be used for calculating the gradients and back
        propagation when training.
    """
    input_dim = int(inputs.get_shape()[1])
    sample_num = int(inputs.get_shape()[0])
    cluster_num = len(cutoff) - 1
    with ops.name_scope(name, "AdaptiveSoftmax"):
        if initializer is None:
            stdv = math.sqrt(1. / input_dim)
            initializer = init_ops.random_uniform_initializer(
                                      -stdv * 0.8, stdv * 0.8)

        head_dim = cutoff[0] + cluster_num
        head_w = variable_scope.get_variable("adaptive_softmax_head_w",
                                             [input_dim, head_dim], initializer=initializer)

        tail_project_factor = project_factor
        tail_w = []
        for i in range(cluster_num):
            project_dim = max(1, input_dim // tail_project_factor)
            tail_dim = cutoff[i + 1] - cutoff[i]
            tail_w.append([
                variable_scope.get_variable("adaptive_softmax_tail{}_proj_w".format(i + 1),
                                            [input_dim, project_dim], initializer=initializer),
                variable_scope.get_variable("adaptive_softmax_tail{}_w".format(i + 1),
                                            [project_dim, tail_dim], initializer=initializer)
            ])
            tail_project_factor *= project_factor

        # Get tail masks and update head labels
        training_losses = []
        loss = array_ops.zeros([sample_num], dtype=dtypes.float32)
        head_labels = labels
        for i in range(cluster_num):
            mask = math_ops.logical_and(math_ops.greater_equal(labels, cutoff[i]),
                                        math_ops.less(labels, cutoff[i + 1]))

            # Update head labels
            head_labels = tf.where(mask, array_ops.constant([cutoff[0] + i] *
                                                            sample_num), head_labels)

            # Compute tail loss
            tail_inputs = array_ops.boolean_mask(inputs, mask)
            tail_labels = array_ops.boolean_mask(labels - cutoff[i], mask)

            tail_logits = math_ops.matmul(math_ops.matmul(tail_inputs, tail_w[i][0]),
                                          tail_w[i][1])
            tail_loss = nn.sparse_softmax_cross_entropy_with_logits(logits=tail_logits,
                                                                    labels=tail_labels)
            training_losses.append(tail_loss)
            aligned_tail_loss = sparse_tensor.SparseTensor(
                array_ops.squeeze(array_ops.where(mask)), tail_loss, [sample_num])
            loss += sparse_ops.sparse_tensor_to_dense(aligned_tail_loss)

        # Compute head loss
        head_logits = math_ops.matmul(inputs, head_w)
        head_loss = nn.sparse_softmax_cross_entropy_with_logits(logits=head_logits,
                                                                labels=head_labels)
        loss += head_loss
        training_losses.append(head_loss)

        return loss, training_losses


class LSTMLM(object):
    def __init__(self, config, mode, device, reuse=None):
        self.config = config
        self.mode = mode
        if mode == "Train":
            self.is_training = True
            self.batch_size = self.config.train_batch_size
            self.step_size = self.config.train_step_size
        elif mode == "Valid":
            self.is_training = False
            self.batch_size = self.config.valid_batch_size
            self.step_size = self.config.valid_step_size
        else:
            self.is_training = False
            self.batch_size = self.config.test_batch_size
            self.step_size = self.config.test_step_size

        vocab_size = config.vocab_size
        embed_dim = config.word_embedding_dim
        lstm_size = config.lstm_size
        lstm_layers = config.lstm_layers
        lstm_forget_bias = config.lstm_forget_bias
        batch_size = self.batch_size
        step_size = self.step_size

        with tf.device(device), tf.name_scope(mode), tf.variable_scope("LSTMLM", reuse=reuse):
            # INPUTS and TARGETS
            self.inputs = tf.placeholder(tf.int32, [batch_size, step_size])
            self.targets = tf.placeholder(tf.int32, [batch_size, step_size])

            # Inititial state
            self.initial_state = tuple(rnn.LSTMStateTuple(tf.placeholder(tf.float32,
                                                                         [batch_size, lstm_size]),
                                                          tf.placeholder(tf.float32,
                                                                         [batch_size, lstm_size]))
                                       for _ in range(lstm_layers))

            # WORD EMBEDDING
            stdv = np.sqrt(1. / vocab_size)
            self.word_embedding = tf.get_variable("word_embedding", [
                vocab_size, embed_dim], initializer=tf.random_uniform_initializer(-stdv, stdv))
            inputs = tf.nn.embedding_lookup(self.word_embedding, self.inputs)

            # INPUT DROPOUT
            if self.is_training and self.config.dropout_prob > 0:
                inputs = tf.nn.dropout(
                    inputs, keep_prob=1 - config.dropout_prob)

            # LSTM
            lstm_cell = rnn.BasicLSTMCell(
                lstm_size, forget_bias=lstm_forget_bias, state_is_tuple=True)
            if self.is_training and config.dropout_prob > 0:
                lstm_cell = rnn.DropoutWrapper(lstm_cell,
                                               output_keep_prob=1. - config.dropout_prob)
            cell = rnn.MultiRNNCell(
                [lstm_cell] * lstm_layers, state_is_tuple=True)

            inputs = tf.unstack(inputs, axis=1)
            outputs, self.final_state = rnn.static_rnn(
                cell, inputs, initial_state=self.initial_state)

            output = tf.reshape(tf.concat(outputs, 1), [-1, lstm_size])

            # Softmax & loss
            labels = tf.reshape(self.targets, [-1])
            if config.softmax_type == 'AdaptiveSoftmax':
                cutoff = config.adaptive_softmax_cutoff
                self.loss, training_losses = adaptive_softmax_loss(output,
                                                                   labels, cutoff)
            else:
                stdv = np.sqrt(1. / vocab_size)
                initializer = tf.random_uniform_initializer(
                    -stdv * 0.8, stdv * 0.8)
                softmax_w = tf.get_variable(
                    "softmax_w", [lstm_size, vocab_size], initializer=initializer)
                softmax_b = tf.get_variable("softmax_b", [vocab_size],
                                            initializer=tf.constant_initializer(0.0, dtype=tf.float32))
                logits = tf.matmul(output, softmax_w) + softmax_b
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels)
                training_losses = [self.loss]

            self.cost = tf.reduce_sum(self.loss)

            if self.is_training:
                self.lr = tf.Variable(0.0, trainable=False)
                optimizer = tf.train.AdagradOptimizer(
                    self.lr, config.adagrad_eps)
                tvars = tf.trainable_variables()
                grads = tf.gradients([tf.reduce_sum(loss) / batch_size for loss in training_losses],
                                     tvars)
                grads = [tf.clip_by_norm(
                    grad, config.max_grad_norm) if grad is not None else grad for grad in grads]
                self.eval_op = optimizer.apply_gradients(zip(grads, tvars))
            else:
                self.eval_op = tf.no_op()

    def update_lr(self, session, learning_rate):
        if self.is_training:
            session.run(tf.assign(self.lr, learning_rate))

    def get_initial_state(self):
        return tuple(np.zeros([self.batch_size, self.config.lstm_size], dtype=np.float32)
                     for _ in range(2 * self.config.lstm_layers))


class Config(object):
    epoch_num = 20
    train_batch_size = 128
    train_step_size = 20
    valid_batch_size = 128
    valid_step_size = 20
    test_batch_size = 20
    test_step_size = 1
    word_embedding_dim = 512
    lstm_layers = 1
    lstm_size = 512
    lstm_forget_bias = 0.0
    max_grad_norm = 0.25
    init_scale = 0.05
    learning_rate = 0.2
    decay = 0.5
    decay_when = 1.0
    dropout_prob = 0.5
    adagrad_eps = 1e-5
    vocab_size = 10001
    softmax_type = "AdaptiveSoftmax"
    adaptive_softmax_cutoff = [2000, vocab_size]


class LearningRateUpdater(object):
    def __init__(self, init_lr, decay_rate, decay_when):
        self._init_lr = init_lr
        self._decay_rate = decay_rate
        self._decay_when = decay_when
        self._current_lr = init_lr
        self._last_ppl = -1

    def get_lr(self):
        return self._current_lr

    def update(self, cur_ppl):
        if self._last_ppl > 0 and self._last_ppl - cur_ppl < self._decay_when:
            current_lr = self._current_lr * self._decay_rate
            INFO_LOG("learning rate: {} ==> {}".format(
                self._current_lr, current_lr))
            self._current_lr = current_lr
        self._last_ppl = cur_ppl


def run(session, model, reader, verbose=True):
    state = model.get_initial_state()
    total_cost = 0
    total_word_cnt = 0
    start_time = time.time()

    for batch in reader.yieldSpliceBatch(model.mode, model.batch_size, model.step_size):
        batch_id, batch_num, x, y, word_cnt = batch
        feed = {model.inputs: x, model.targets: y, model.initial_state: state}
        cost, state, _ = session.run(
            [model.cost, model.final_state, model.eval_op], feed)
        total_cost += cost
        total_word_cnt += word_cnt
        if verbose and (batch_id % max(10, batch_num // 10)) == 0:
            ppl = np.exp(total_cost / total_word_cnt)
            wps = total_word_cnt / (time.time() - start_time)
            print("  [%5d/%d]ppl: %.3f speed: %.0f wps costs %.3f words %d" % (
                batch_id, batch_num, ppl, wps, total_cost, total_word_cnt))
    return total_cost, total_word_cnt, np.exp(total_cost / total_word_cnt)


def main(_):
    reader = Reader(FLAGS.data_path)
    config = Config()

    if FLAGS.use_adaptive_softmax:
        config.softmax_type = 'AdaptiveSoftmax'
    else:
        config.softmax_type = 'FullSoftmax'

    gpuid = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(gpuid)
    device = '/gpu:0'

    lr_updater = LearningRateUpdater(
        config.learning_rate, config.decay, config.decay_when)

    graph = tf.Graph()
    with graph.as_default():
        trainm = LSTMLM(config, device=device, mode="Train", reuse=False)
        validm = LSTMLM(config, device=device, mode="Valid", reuse=True)
        testm = LSTMLM(config, device=device, mode="Test", reuse=True)

    session_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=session_config) as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(config.epoch_num):
            trainm.update_lr(session, lr_updater.get_lr())
            INFO_LOG("Epoch {}, learning rate: {}".format(
                epoch + 1, lr_updater.get_lr()))
            cost, word_cnt, ppl = run(session, trainm, reader)
            INFO_LOG("Epoch %d Train perplexity %.3f words %d" %
                     (epoch + 1, ppl, word_cnt))

            cost, word_cnt, ppl = run(session, validm, reader)
            INFO_LOG("Epoch %d Valid perplexity %.3f words %d" %
                     (epoch + 1, ppl, word_cnt))

            lr_updater.update(ppl)
            cost, word_cnt, ppl = run(session, testm, reader)
            INFO_LOG("Epoch %d Test perplexity %.3f words %d" %
                     (epoch + 1, ppl, word_cnt))


if __name__ == '__main__':
    tf.app.run()
