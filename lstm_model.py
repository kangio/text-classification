#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import pickle
import os
from tensorflow.contrib import rnn
import time
from util import segment


class LSTMModel():
    def __init__(self,
                 data_loader,
                 hidden_size=128,
                 num_layers=1,
                 embedding_mode='w2v',
                 keep_prob=0.5
                 ):
        self.is_trainning = True
        self.data = data_loader
        vocab_size = data_loader.vocab_size
        label_size = data_loader.label_size
        seq_length = data_loader.seq_length
        self.save_dir = './save/lstm/'

        def lstm_cell():
            lstm = rnn.BasicLSTMCell(
                hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
            if self.is_trainning: lstm = rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return lstm

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(tf.int64, [None, seq_length], name='input_x')
            self.targets = tf.placeholder(tf.int64, [None, ], name='y_label')  # target is class label
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')

        with tf.variable_scope('embeddingLayer'):
            if embedding_mode == 'random':
                W = tf.get_variable('W', [vocab_size, hidden_size])
            else:
                embedding_path = './embedding/single_w2v_vec.pkl' if self.data.is_char else './embedding/word_w2v_vec.pkl'
                with open(embedding_path, 'rb') as cc:
                    W = tf.Variable(pickle.load(cc))

            embedded = tf.nn.embedding_lookup(W, self.input_data)
            # shape: (batch_size, seq_length, cell.input_size) => (seq_length, batch_size, cell.input_size)
            inputs = tf.split(embedded, seq_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        with tf.variable_scope('rnnLayer'):
            self.cell = rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
            outputs, last_state = rnn.static_rnn(self.cell, inputs, dtype=tf.float32)

        with tf.variable_scope('softmaxLayer'):
            softmax_w = tf.get_variable('w', [hidden_size, label_size])
            tf.summary.histogram('weight', softmax_w)
            softmax_b = tf.get_variable('b', [label_size])
            tf.summary.histogram('bias', softmax_b)
            self.logits = tf.matmul(outputs[-1], softmax_w) + softmax_b
        with tf.name_scope('cost'):
            self.cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets))  # Softmax loss
            tf.summary.scalar('cost', self.cost)
        self.final_state = last_state
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)  # Adam Optimizer
        with tf.name_scope("accuracy"):
            self.probs = tf.nn.softmax(self.logits)
            self.correct_pred = tf.equal(tf.argmax(self.probs, 1), self.targets)
            self.correct_num = tf.reduce_sum(tf.cast(self.correct_pred, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        if not os.path.exists('./save/lstm'):
            os.mkdir('./save/lstm')
        self.saver = tf.train.Saver(tf.global_variables())

    def fit(self,
            learning_rate=0.001,
            decay_rate=0.9,
            num_epochs=20,
            save_every=500,
            ):
        print("开始fit")
        self.is_trainning = True
        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('logs', sess.graph)
            init = tf.global_variables_initializer()
            sess.run(init)

            for e in range(num_epochs):

                self.data.reset_batch_pointer()
                total_correct_nums = 0
                for b in range(self.data.num_batches):

                    start = time.time()
                    x, y = self.data.next_train_batch()
                    feed = {self.lr: learning_rate * (decay_rate ** e), self.input_data: x,
                            self.targets: y}
                    train_loss, _, correct_num = sess.run([self.cost, self.optimizer, self.correct_num],
                                                          feed_dict=feed)
                    end = time.time()
                    total_correct_nums += correct_num
                    if (e * self.data.num_batches + b + 1) % save_every == 0 \
                            or (e == num_epochs - 1 and b == self.data.num_batches - 1):
                        self.saver.save(sess, self.save_dir + 'model.ckpt',
                                        global_step=e * self.data.num_batches + b + 1)

                    if b == self.data.num_batches - 1:
                        print('{}/{} (epoch {}), train_loss = {:.3f}, accuracy = {:.3f}, time/batch = {:.3f}' \
                              .format(e * self.data.num_batches + b + 1,
                                      num_epochs * self.data.num_batches,
                                      e + 1,
                                      train_loss,
                                      total_correct_nums / self.data.text_length,
                                      end - start))

                        res = sess.run(merged, feed_dict=feed)
                        writer.add_summary(res, e)

    def test_accuracy(self):
        self.is_trainning = False
        with tf.Session() as sess:
            # saver.restore(sess, '/Users/yangkang/PycharmProjects/TextClassification/save/lstm/model.ckpt')
            ckpt = tf.train.get_checkpoint_state(self.save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print('ok')
                self.saver.restore(sess, ckpt.model_checkpoint_path)

            correct_total = 0.0
            num_total = 0.0
            start = time.time()
            x, y = self.data.next_test_batch()

            feed = {self.input_data: x}
            probs = sess.run(self.probs, feed_dict=feed)

            results = np.argmax(probs, 1)

            correct_num = np.sum(results == y)
            end = time.time()
            print('cost time {:.3f}, sub_accuracy = {:.6f}'.format(end - start, correct_num * 1.0 / len(x)))

            correct_total += correct_num
            num_total += len(x)

            accuracy_total = correct_total / num_total
            print('total_num = {}, total_accuracy = {:.6f}'.format(int(num_total), accuracy_total))

    def predict(self, text):
        text = ' '.join(segment(text))
        x = self.data.transform(text)
        with tf.Session() as sess:
            # saver.restore(sess,'/Users/yangkang/PycharmProjects/TextClassification/save/cnn/model.ckpt')
            ckpt = tf.train.get_checkpoint_state(self.save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            # list(predict_label(sess, model, data_loader.labels, x))
            x = np.array([x])
            feed = {self.input_data: x}
            probs = sess.run(self.probs, feed_dict=feed)
        print(probs)
        results = []
        for index, i in enumerate(probs[0]):
            if i > 0.06:
                results.append(index)
        #
        # results = np.argmax(probs, 1)
        id2labels = dict(zip(self.data.labels.values(), self.data.labels.keys()))
        labels = list(map(id2labels.get, results))
        print(labels)
