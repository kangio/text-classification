#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.contrib import learn
from tensorflow.contrib import rnn

hidden_size = 100
keep_prob = 1.0
embeding_path='/Users/yangkang/PycharmProjects/TextClassification/utils/embed_w2v.pkl'
# embeding_path='/Users/yangkang/PycharmProjects/TextClassification/utils/embed_w2v.pkl'
num_layers=2
class LSTMModel():
    def __init__(self, is_training, vocab_size,label_size, seq_length):
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        with tf.name_scope('input'):
            self.input_data=tf.placeholder(tf.int64,[None,seq_length],name='input_x')
            self.targets = tf.placeholder(tf.int64, [None,],name='y_label')  # target is class label
        self.lr = tf.placeholder(dtype=tf.float32,shape=[],name='learning_rate')

        with tf.variable_scope('embeddingLayer'):
            # W = tf.get_variable('W', [vocab_size, hidden_size])
            # W=tf.Variable(np.load('/Users/yangkang/PycharmProjects/TextClassification/utils/vocab.npy'))
            with open(embeding_path,'rb') as cc:
                W=tf.Variable(pickle.load(cc))

            embedded = tf.nn.embedding_lookup(W, self.input_data)
            # shape: (batch_size, seq_length, cell.input_size) => (seq_length, batch_size, cell.input_size)
            inputs = tf.split(embedded, seq_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

                # if is_training:
                #     inputs=tf.nn.dropout(inputs,keep_prob,name='input_dropout')
                #     print(inputs)

        with tf.variable_scope('rnnLayer'):
            self.cell = rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
            outputs, last_state = rnn.static_rnn(self.cell, inputs, dtype=tf.float32)

        with tf.variable_scope('softmaxLayer'):
            softmax_w = tf.get_variable('w', [hidden_size, label_size])
            tf.summary.histogram('weight',softmax_w)
            softmax_b = tf.get_variable('b', [label_size])
            tf.summary.histogram('bias',softmax_b)
            self.logits = tf.matmul(outputs[-1], softmax_w) + softmax_b
        with tf.name_scope('cost'):
            self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets))  # Softmax loss
            tf.summary.scalar('cost',self.cost)
        self.final_state = last_state
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)  # Adam Optimizer
        with tf.name_scope("accuracy"):
            self.probs = tf.nn.softmax(self.logits)
            self.correct_pred = tf.equal(tf.argmax(self.probs, 1), self.targets)
            self.correct_num = tf.reduce_sum(tf.cast(self.correct_pred, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


