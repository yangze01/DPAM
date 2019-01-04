#coding=utf8
from __future__ import division
import sys
import tensorflow as tf
from tensorflow.contrib import learn


class DPAM(object):
    def __init__(
        self, sequence_length, num_classes, batch_size, vocab_size,
              embedding_size, filter_sizes, num_filters, l2_reg_lambda):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        self.l2_loss = tf.constant(0.0)
        self.dynamic_l2_loss = tf.constant(0.0)

    def add_placeholders(self):
        """
            Add placeholders to the graph
        """
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.input_rule = tf.placeholder(tf.int32, [None, self.sequence_length], name = "input_rule")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def add_embedding(self):
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                name = "W")

        self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
        self.embedded_rule = tf.nn.embedding_lookup(W, self.input_rule)

        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        self.embedded_rule_expanded = tf.expand_dims(self.embedded_rule, -1)

    def add_conv_pool(self):
        pooled_outputs = []
        pooled_outputs_rule = []

        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]

                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")

                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides = [1, 1, 1, 1],
                    padding = "VALID",
                    name = "conv")

                conv_rule = tf.nn.conv2d(
                    self.embedded_rule_expanded,
                    W,
                    strides = [1, 1, 1, 1],
                    padding = "VALID",
                    name = "conv")

                # Apply non_linearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                h_rule = tf.nn.relu(tf.nn.bias_add(conv_rule, b), name = "relu")

                # Max_pooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides = [1, 1, 1, 1],
                    padding="VALID",
                    name="pool")
                pooled_rule = tf.nn.max_pool(
                    h_rule,
                    ksize = [1, self.sequence_length - filter_size + 1, 1, 1],
                    strides = [1, 1, 1, 1],
                    padding = "VALID",
                    name = "pool")
                pooled_outputs.append(pooled)
                pooled_outputs_rule.append(pooled_rule)

        # Combine all the pooled features
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, axis = 3)
        self.h_pool_relu = tf.concat(pooled_outputs_rule, axis = 3)

        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])
        self.h_pool_flat_relu = tf.reshape(self.h_pool_relu, [-1, self.num_filters_total])


    def add_dropout(self):
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            self.h_drop_rule = tf.nn.dropout(self.h_pool_flat_relu, self.dropout_keep_prob)

    def add_ratio_cal(self):
        multi_matrix = tf.matmul(self.h_drop_rule, tf.transpose(self.h_drop_rule))
        diag_matrix = tf.diag_part(multi_matrix)
        expand_diag_matrix = tf.expand_dims(diag_matrix, 0)
        norm_matrix = tf.sqrt(tf.matmul(tf.transpose(expand_diag_matrix), expand_diag_matrix))
        self.sim_matrix = tf.div(multi_matrix, norm_matrix)
        self.initial_matrix = tf.zeros([1, self.num_classes])

        for i in range(0, 64):
            y_label = self.input_y[i]
            self.where = tf.not_equal(y_label, 0)
            self.indices = tf.reshape(tf.where(self.where), [1, -1])
            self.gather_data = tf.gather(self.sim_matrix, self.indices)
            self.prod_data = tf.reduce_prod(self.gather_data, axis = 1)
            self.initial_matrix = tf.concat([self.initial_matrix, self.prod_data], axis = 0)
        self.initial_matrix = tf.gather(self.initial_matrix, range(1, self.initial_matrix.shape[0]))
        self.input_ratio = tf.gather(self.input_y, range(0, 64)) * self.initial_matrix


    def add_output(self):
        with tf.name_scope("cross_output"):
            self.W = tf.get_variable(
                "W",
                shape = [self.num_filters_total, self.num_classes],
                initializer = tf.contrib.layers.xavier_initializer())
            self.b = tf.Variable(tf.constant(0.1, shape = [self.num_classes]), "b")

            self.scores = tf.nn.xw_plus_b(self.h_drop, self.W, self.b, name = "scores")

        self.l2_loss += self.l2_reg_lambda * tf.nn.l2_loss(self.W)

        with tf.name_scope("dynamic_threshold"):
            self.dynamic_W = tf.get_variable(
                "dynamic_W",
                shape = [self.num_filters_total, self.num_classes],
                initializer = tf.contrib.layers.xavier_initializer())
            self.dynamic_b = tf.Variable(tf.constant(0.01, shape = [self.num_classes]), name = "dynamic_b")
            self.dynamic_scores = tf.nn.xw_plus_b(self.h_drop, self.dynamic_W, self.dynamic_b, name = "dynamic_scores")
            self.threshold = tf.clip_by_value(self.dynamic_scores, 0.1, 0.9)

        self.dynamic_l2_loss += self.l2_reg_lambda * tf.nn.l2_loss(self.dynamic_W)

        with tf.name_scope("output"):
            self.sigmoid_value = tf.nn.sigmoid(self.scores)
            self.sigmoid_final = self.sigmoid_value - self.threshold
            self.predictions_tmp = tf.greater_equal(self.sigmoid_final, 0)
            self.predictions = tf.cast(self.predictions_tmp, "float", name = "predictions")


    def cal_dynamic_loss(self, logits, labels):
        margin = -labels*(logits - self.dynamic_scores)
        return tf.log(1 + tf.exp(margin))

    def cal_neg_loss(self, logits, labels):
        y_hat = tf.sigmoid(self.scores * labels)
        ns_hat = tf.sigmoid(-self.scores * (1 - labels))
        cost_s = -tf.log(y_hat)
        cost_n = -tf.log(ns_hat)
        cost_loss = cost_s + cost_n

        return cost_loss

    def add_loss(self):
        with tf.name_scope("loss"):
            self.index = 1 - tf.gather(self.input_y, range(0, 64))
            self.mul_value = tf.gather(self.sigmoid_value, range(0, 64)) * self.input_ratio
            self.add_value = tf.log(self.mul_value + self.index, name = "add_value")


            dynamic_losses = self.cal_dynamic_loss(logits = self.sigmoid_value, labels = (self.input_y - 0.5) * 2)
            self.dynamic_loss = tf.reduce_mean(tf.gather(dynamic_losses, range(0, 64)))

            neg_losses = self.cal_neg_loss(logits = self.scores, labels = self.input_y)
            self.neg_loss = tf.reduce_mean(neg_losses)

            self.start_loss = 0.5 * self.dynamic_loss + 0.5 * self.neg_loss + self.l2_loss + self.dynamic_l2_loss

            self.loss = 0.5 * self.dynamic_loss + \
                        0.5 * tf.reduce_mean(self.neg_loss - self.add_value) + self.l2_loss + self.dynamic_l2_loss

            self.dev_loss = 0.5 * self.dynamic_loss + 0.5 * self.neg_loss + self.l2_loss + self.dynamic_l2_loss








