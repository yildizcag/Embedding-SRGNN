#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/10/16 4:36
# @Author : {ZM7}
# @File : model.py
# @Software: PyCharm
import tensorflow as tf
import math



class Model(object):
    def __init__(self, hidden_size=100, out_size=100, batch_size=100, nonhybrid=True):
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.mask = tf.keras.Input(shape=(None,), dtype=tf.dtypes.float32)
        self.alias = tf.keras.Input(shape=(None,), dtype=tf.dtypes.int32)
        self.item = tf.compat.v1.placeholder(dtype=tf.int32)
        self.tar = tf.keras.Input(shape=(), dtype=tf.dtypes.int32)
        self.nonhybrid = nonhybrid
        self.stdv = 1.0 / math.sqrt(self.hidden_size)

        self.nasr_w1 = tf.Variable(tf.random_uniform_initializer(-self.stdv, self.stdv)([self.out_size, self.out_size]), 
                                    name='nasr_w1', shape=[self.out_size, self.out_size], dtype=tf.float32)
        self.nasr_w2 = tf.Variable(tf.random_uniform_initializer(-self.stdv, self.stdv)([self.out_size, self.out_size]),
                                    name='nasr_w2', shape=[self.out_size, self.out_size], dtype=tf.float32)
        self.nasr_v = tf.Variable(tf.random_uniform_initializer(-self.stdv, self.stdv)([1, self.out_size]),
                                    name='nasr_v', shape=[1, self.out_size], dtype=tf.float32)
        self.nasr_b = tf.Variable(tf.zeros_initializer()([self.out_size]), name='nasr_b', shape=[self.out_size], dtype=tf.float32)

    def forward(self, re_embedding, train=True):
        rm = tf.math.reduce_sum(self.mask, 1)
        last_id = tf.gather_nd(self.alias, tf.stack([tf.range(self.batch_size), tf.dtypes.cast(rm, tf.int32)-1], axis=1))
        last_h = tf.gather_nd(re_embedding, tf.stack([tf.range(self.batch_size), last_id], axis=1))
        seq_h = tf.stack([tf.nn.embedding_lookup(re_embedding[i], self.alias[i]) for i in range(self.batch_size)],
                         axis=0)                                                           #batch_size*T*d
        last = tf.linalg.matmul(last_h, self.nasr_w1)
        seq = tf.linalg.matmul(tf.reshape(seq_h, [-1, self.out_size]), self.nasr_w2)
        last = tf.reshape(last, [self.batch_size, 1, -1])
        m = tf.math.sigmoid(last + tf.reshape(seq, [self.batch_size, -1, self.out_size]) + self.nasr_b)
        coef = tf.linalg.matmul(tf.reshape(m, [-1, self.out_size]), self.nasr_v, transpose_b=True) * tf.reshape(
            self.mask, [-1, 1])
        b = self.embedding[1:]
        if not self.nonhybrid:
            ma = tf.concat([tf.math.reduce_sum(tf.reshape(coef, [self.batch_size, -1, 1]) * seq_h, 1),
                            tf.reshape(last, [-1, self.out_size])], -1)
            try:
                self.B
            except AttributeError:
                self.B = tf.Variable(tf.random_uniform_initializer(-self.stdv, self.stdv)([2 * self.out_size, self.out_size]),
                                            name='B', dtype=tf.float32, shape=[2 * self.out_size, self.out_size])
            y1 = tf.linalg.matmul(ma, self.B)
            logits = tf.linalg.matmul(y1, b, transpose_b=True)
        else:
            ma = tf.math.reduce_sum(tf.reshape(coef, [self.batch_size, -1, 1]) * seq_h, 1)
            logits = tf.linalg.matmul(ma, b, transpose_b=True)
        loss = tf.math.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tar - 1, logits=logits))
        self.vars = tf.compat.v1.trainable_variables()
        if train:
            lossL2 = tf.math.add_n([tf.nn.l2_loss(v) for v in self.vars if v.name not
                               in ['ggnn_model/gru/rnn/gru_cell/gates/bias:0', 'gamma', 'b', 'g', 'ggnn_model/gru/rnn/gru_cell/candidate/bias:0']]) * self.L2
            loss = loss + lossL2
        return loss, logits

    def run(self, fetches, tar, item, adj_in, adj_out, alias, mask):
        #return fetches(tar,item,adj_in,adj_out,alias,mask)
        return self.sess.run(fetches, feed_dict={self.tar: tar, self.item: item, self.adj_in: adj_in,
                                                 self.adj_out: adj_out, self.alias: alias, self.mask: mask})


class GGNN(Model):
    def __init__(self,hidden_size=100, out_size=100, batch_size=300, n_node=None,
                 lr=None, l2=None, step=1, decay=None, lr_dc=0.1, nonhybrid=False):
        super(GGNN,self).__init__(hidden_size, out_size, batch_size, nonhybrid)
        self.embedding = tf.Variable(tf.random_uniform_initializer(-self.stdv, self.stdv)([n_node, hidden_size]),
                                        name='embedding', shape=[n_node, hidden_size], dtype=tf.float32)
        self.adj_in = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.adj_out = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.n_node = n_node
        self.L2 = l2
        self.step = step
        self.nonhybrid = nonhybrid
        self.W_in = tf.Variable(tf.random_uniform_initializer(-self.stdv, self.stdv)([self.out_size, self.out_size]),
                                    name='W_in', shape=[self.out_size, self.out_size], dtype=tf.float32)
        self.b_in = tf.Variable(tf.random_uniform_initializer(-self.stdv, self.stdv)([self.out_size]),
                                    name='b_in', shape=[self.out_size], dtype=tf.float32)
        self.W_out = tf.Variable(tf.random_uniform_initializer(-self.stdv, self.stdv)([self.out_size, self.out_size]),
                                    name='W_out', shape=[self.out_size, self.out_size], dtype=tf.float32)
        self.b_out = tf.Variable(tf.random_uniform_initializer(-self.stdv, self.stdv)([self.out_size]),
                                    name='b_out', shape=[self.out_size], dtype=tf.float32)
        with tf.compat.v1.variable_scope('ggnn_model', reuse=None):
            self.loss_train, _ = self.forward(self.ggnn())
        with tf.compat.v1.variable_scope('ggnn_model', reuse=True):
            self.loss_test, self.score_test = self.forward(self.ggnn(), train=False)
        self.global_step = tf.Variable(0)
        self.learning_rate = tf.compat.v1.train.exponential_decay(lr, global_step=self.global_step, decay_steps=decay,
                                                        decay_rate=lr_dc, staircase=True)
        self.opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss_train, global_step=self.global_step)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
            except RuntimeError as e:
                print(e)
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def ggnn(self):
        fin_state = tf.nn.embedding_lookup(self.embedding, self.item)
        cell = tf.keras.layers.GRUCell(self.out_size)
        with tf.compat.v1.variable_scope('gru'):
            for i in range(self.step):
                fin_state = tf.reshape(fin_state, [self.batch_size, -1, self.out_size])
                fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                    self.W_in) + self.b_in, [self.batch_size, -1, self.out_size])
                fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                     self.W_out) + self.b_out, [self.batch_size, -1, self.out_size])
                av = tf.concat([tf.linalg.matmul(self.adj_in, fin_state_in),
                                tf.linalg.matmul(self.adj_out, fin_state_out)], axis=-1)
                state_output, fin_state = \
                    tf.compat.v1.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(av, [-1, 2*self.out_size]), axis=1),
                                      initial_state=tf.reshape(fin_state, [-1, self.out_size]))
        return tf.reshape(fin_state, [self.batch_size, -1, self.out_size])


