#!/usr/bin/python -u
import os
import time

import numpy as np
import tensorflow as tf

from reader import *

flags = tf.flags
flags.DEFINE_string("data_path", "ptb_data", "Where the training/test data is stored.")
flags.DEFINE_bool("use_adaptive_softmax", True, "Train using adaptive softmax")

FLAGS = flags.FLAGS

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
            self.inputs  = tf.placeholder(tf.int32, [batch_size, step_size]) 
            self.targets = tf.placeholder(tf.int32, [batch_size, step_size])

            # Inititial state
            self.initial_state = tf.placeholder(tf.float32, 
                [batch_size, lstm_size * 2 * lstm_layers])

            # WORD EMBEDDING
            stdv = np.sqrt(1. / vocab_size)
            self.word_embedding = tf.get_variable("word_embedding", [
                vocab_size, embed_dim], initializer=tf.random_uniform_initializer(-stdv, stdv))
            inputs = tf.nn.embedding_lookup(self.word_embedding, self.inputs)

            # INPUT DROPOUT 
            if self.is_training and self.config.dropout_prob > 0:
                inputs = tf.nn.dropout(inputs, keep_prob=1 - config.dropout_prob)

            # LSTM
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=lstm_forget_bias, state_is_tuple=False)
            if self.is_training and config.dropout_prob > 0:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, 
                    output_keep_prob=1. - config.dropout_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * lstm_layers, state_is_tuple=False)

            inputs = tf.unstack(inputs, axis=1)
            outputs, self.final_state = tf.nn.rnn(cell, inputs, initial_state=self.initial_state)

            output = tf.reshape(tf.concat(1, outputs), [-1, lstm_size])

            # Softmax & loss
            labels = tf.reshape(self.targets, [-1])
            if config.softmax_type == 'AdaptiveSoftmax':
                cutoff = config.adaptive_softmax_cutoff
                self.loss, training_losses = tf.contrib.layers.adaptive_softmax_loss(output, 
                    labels, cutoff)
            else:
                stdv = np.sqrt(1. / vocab_size)
                initializer = tf.random_uniform_initializer(-stdv * 0.8, stdv * 0.8)
                softmax_w = tf.get_variable("softmax_w", [lstm_size, vocab_size], initializer=initializer)
                softmax_b = tf.get_variable("softmax_b", [vocab_size], 
                    initializer=tf.constant_initializer(0.0, dtype=tf.float32))
                logits = tf.matmul(output, softmax_w) + softmax_b
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
                training_losses = [self.loss]

            self.cost = tf.reduce_sum(self.loss)

            if self.is_training:
                self.lr = tf.Variable(0.0, trainable=False)
                optimizer = tf.train.AdagradOptimizer(self.lr, config.adagrad_eps)
                tvars = tf.trainable_variables()
                grads = tf.gradients([tf.reduce_sum(loss) / batch_size for loss in training_losses],
                    tvars)
                grads = [tf.clip_by_norm(grad, config.max_grad_norm) if grad is not None else grad for grad in grads]
                self.eval_op = optimizer.apply_gradients(zip(grads, tvars))
            else:
                self.eval_op = tf.no_op()

    def update_lr(self, session, learning_rate):
        if self.is_training:
            session.run(tf.assign(self.lr, learning_rate))

    def get_initial_state(self):
        return np.zeros([self.batch_size, self.config.lstm_size * 2 * self.config.lstm_layers], dtype=np.float32)


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
            INFO_LOG("learning rate: {} ==> {}".format(self._current_lr, current_lr))
            self._current_lr = current_lr
        self._last_ppl = cur_ppl

def run(session, model, reader, verbose=True):
    state = model.get_initial_state()
    total_cost = 0
    total_word_cnt = 0
    start_time = time.time()

    for batch in reader.yieldSpliceBatch(model.mode, model.batch_size, model.step_size):
        batch_id, batch_num, x, y, word_cnt = batch
        feed = {model.inputs: x, model.targets:y, model.initial_state: state}
        cost, state, _ = session.run([model.cost, model.final_state, model.eval_op], feed)
        total_cost += cost
        total_word_cnt += word_cnt
        if verbose and (batch_id % max(10, batch_num//10)) == 0:
            ppl = np.exp(total_cost / total_word_cnt)
            wps = total_word_cnt / (time.time() - start_time)
            print "  [%5d/%d]ppl: %.3f speed: %.0f wps costs %.3f words %d" % (
                batch_id, batch_num, ppl, wps, total_cost, total_word_cnt)
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
    
    lr_updater = LearningRateUpdater(config.learning_rate, config.decay, config.decay_when)

    graph = tf.Graph()
    with graph.as_default():
        trainm = LSTMLM(config, device=device, mode="Train", reuse=False)
        validm = LSTMLM(config, device=device, mode="Valid", reuse=True)
        testm  = LSTMLM(config, device=device, mode="Test", reuse=True)
    
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=session_config) as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(config.epoch_num):
            trainm.update_lr(session, lr_updater.get_lr())
            INFO_LOG("Epoch {}, learning rate: {}".format(epoch + 1, lr_updater.get_lr()))
            cost, word_cnt, ppl = run(session, trainm, reader)
            INFO_LOG("Epoch %d Train perplexity %.3f words %d" % (epoch + 1, ppl, word_cnt))

            cost, word_cnt, ppl = run(session, validm, reader)
            INFO_LOG("Epoch %d Valid perplexity %.3f words %d" % (epoch + 1, ppl, word_cnt))

            lr_updater.update(ppl)
            cost, word_cnt, ppl = run(session, testm, reader)
            INFO_LOG("Epoch %d Test perplexity %.3f words %d" % (epoch + 1, ppl, word_cnt))

if __name__ == '__main__':
    tf.app.run()

