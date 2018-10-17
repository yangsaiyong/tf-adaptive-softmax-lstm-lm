#!/usr/bin/python -u
#encoding=utf8
#Author: yangsaiyong@gmail.com
#Update: 2018.10.17

import tensorflow as tf
import numpy as np

class FullSoftmax(object):
    def __init__(self, input_dim, vocab_size, initializer=None, name=None):
        with tf.variable_scope(name or type(self).__name__, initializer=initializer):
            self.softmax_w = tf.get_variable("softmax_w", [input_dim, vocab_size])
            self.softmax_b = tf.get_variable("softmax_b", [vocab_size], \
                initializer=tf.constant_initializer(0.0, dtype=tf.float32))

    def loss(self, inputs, labels, name='loss'):
        logits = tf.matmul(inputs, self.softmax_w) + self.softmax_b
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name=name)
        return loss, [loss]

    def softmax(self, inputs, name='softmax'):
        logits = tf.matmul(inputs, self.softmax_w) + self.softmax_b
        return tf.nn.softmax(logits, name=name)

    def log_softmax(self, inputs, name='log_softmax'):
        logits = tf.matmul(inputs, self.softmax_w) + self.softmax_b
        return tf.nn.log_softmax(logits, name=name)

class AdaptiveSoftmax(object):
    def __init__(self, input_dim, cutoff, project_factor=4, project_dims=None, initializer=None, name=None):
        self.cluster_num = len(cutoff) - 1
        if project_dims:
            assert(len(project_dims) == self.cluster_num)
        else:
            project_dims = []
            tail_project_factor = project_factor
            for i in range(self.cluster_num):
                dim = max(1, input_dim / tail_project_factor)
                project_dims.append(dim)
                tail_project_factor *= project_factor

        self.cutoff = cutoff
        with tf.variable_scope(name or type(self).__name__, initializer=initializer):
            head_dim = cutoff[0] + self.cluster_num
            self.head_w = tf.get_variable("adaptive_softmax_head_w", [input_dim, head_dim])
            
            self.tail_w = []
            for i in range(self.cluster_num):
                project_dim = project_dims[i]
                tail_dim = cutoff[i + 1] - cutoff[i]
                self.tail_w.append([
                    tf.get_variable("adaptive_softmax_tail{}_proj_w".format(i+1), [input_dim, project_dim]),
                    tf.get_variable("adaptive_softmax_tail{}_w".format(i+1), [project_dim, tail_dim])
                ])

    def loss(self, inputs, labels, name='loss'): 
        # Get tail masks and update head labels
        training_losses = []
        head_labels = labels
        ones = tf.ones([tf.size(labels)], dtype=tf.int32)
        for i in range(self.cluster_num):
            mask = tf.logical_and(tf.greater_equal(labels, self.cutoff[i]), tf.less(labels, self.cutoff[i + 1]))
            
            # Update head labels
            head_labels = tf.where(mask, ones * (self.cutoff[0] + i), head_labels)

            # Compute tail loss
            tail_inputs = tf.boolean_mask(inputs, mask)
            tail_logits = tf.matmul(tf.matmul(tail_inputs, self.tail_w[i][0]), self.tail_w[i][1])
            tail_labels = tf.boolean_mask(labels - self.cutoff[i], mask)
            tail_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tail_logits, labels=tail_labels)
            training_losses.append(tail_loss)
            aligned_tail_loss = tf.SparseTensor(tf.squeeze(tf.where(mask)), tail_loss, [tf.size(labels, out_type=tf.int64)])
            loss = tf.sparse_tensor_to_dense(aligned_tail_loss) if i == 0 else \
                loss + tf.sparse_tensor_to_dense(aligned_tail_loss)

        # Compute head loss
        head_logits = tf.matmul(inputs, self.head_w) # (sample_num, head_size)
        head_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=head_logits, labels=head_labels) #(sample_num)
        training_losses.append(head_loss)
        loss = tf.add(loss, head_loss, name=name)

        return loss, training_losses

    def softmax(self, inputs, name='softmax'):
        head_logits = tf.matmul(inputs, self.head_w)
        head_softmax = tf.nn.softmax(head_logits)
        softmax_list = [head_softmax[:, :self.cutoff[0]]]
        for i in range(self.cluster_num):
            tail_logits = tf.matmul(tf.matmul(inputs, self.tail_w[i][0]), self.tail_w[i][1])
            tail_softmax = tf.nn.softmax(tail_logits)
            index = self.cutoff[0] + i
            softmax_list.append(tail_softmax * head_softmax[:, index:index+1])
        return tf.concat(softmax_list, axis=1, name=name)

    def log_softmax(self, inputs, name='log_softmax'):
        head_logits = tf.matmul(inputs, self.head_w)
        head_logsoftmax = tf.nn.log_softmax(head_logits)
        logsoftmax_list = [head_logsoftmax[:, :self.cutoff[0]]]
        for i in range(self.cluster_num):
            tail_logits = tf.matmul(tf.matmul(inputs, self.tail_w[i][0]), self.tail_w[i][1])
            tail_logsoftmax = tf.nn.log_softmax(tail_logits)
            index = self.cutoff[0] + i
            logsoftmax_list.append(tail_logsoftmax + head_logsoftmax[:, index:index+1])
        return tf.concat(logsoftmax_list, axis=1, name=name)

