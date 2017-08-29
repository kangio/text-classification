#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0
@author: yangkang
@license: Apache Licence 
@contact: ghooo55@gmail.com
@site: http://abc.com/
@software: PyCharm
@file: cnn_model.py
@time: 2017/7/27 下午5:33
"""

import numpy as np
import tensorflow as tf
import pickle
from util import segment
import time

class CNNModel():
    def __init__(self,
                 data_loader,
                 hidden_size=128,
                 num_filters=32,
                 filter_sizes=(2, 3, 4),
                 embedding_mode='w2v',
                 ):
        self.is_trainning = True
        self.data = data_loader
        self.lr = tf.placeholder(tf.float32)
        vocab_size = data_loader.vocab_size
        label_size = data_loader.label_size
        seq_length = data_loader.seq_length
        self.save_dir = './save/cnn/'

        with tf.name_scope('data_input'):
            self.input_data = tf.placeholder(tf.int64, [None, self.data.seq_length])
            self.targets = tf.placeholder(tf.int64, [None])
            self.target_onehot = tf.one_hot(self.targets, depth=label_size)

        with tf.variable_scope("embeddingLayer"):
            if embedding_mode == 'random':
                embedding = tf.get_variable('embedding', [vocab_size, hidden_size])
            # embedding=tf.Variable(np.load(embedding_path))
            elif embedding_mode == 'w2v':
                embedding_path = './embedding/single_w2v_vec.pkl' if self.data.is_char else './embedding/word_w2v_vec.pkl'

                with open(embedding_path, 'rb') as cc:
                    embedding = tf.Variable(pickle.load(cc))
            else:
                return
            embeded = tf.nn.embedding_lookup(embedding, self.input_data)
            embeded = tf.expand_dims(embeded, axis=-1)
        total_result = []

        for index, filter_size in enumerate(filter_sizes):
            with tf.variable_scope('conv-pooling%d' % index):
                filter_shape = [filter_size, hidden_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                tf.summary.histogram(name='conW', values=W)
                b = tf.Variable(tf.constant(0.1, shape=[num_filters], dtype=tf.float32), name='b')
                tf.summary.histogram(name='conb', values=b)

                conv = tf.nn.conv2d(embeded, W, strides=[1, 1, 1, 1], padding="VALID", name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, bias=b), name='relu')
                pooled = tf.nn.max_pool(h, ksize=[1, self.data.seq_length - filter_size + 1, 1, 1]
                                        , strides=[1, 1, 1, 1], padding='VALID', name='pooling')
                total_result.append(pooled)
        self.h_pool = tf.concat(values=total_result, axis=3)

        with tf.variable_scope('outputlayer'):
            num_filters_total = num_filters * len(filter_sizes)
            pooled_flat = tf.reshape(self.h_pool, shape=[-1, num_filters_total])
            if self.is_trainning:
                pooled_flat = tf.nn.dropout(pooled_flat, keep_prob=0.6)
            W = tf.Variable(tf.truncated_normal([num_filters_total, label_size]), name='W')
            b = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[label_size]), name='b')
            self.logits = tf.nn.xw_plus_b(pooled_flat, W, b, name='score')
            self.probs = tf.nn.softmax(self.logits)
            self.prediction = tf.argmax(self.logits, 1, name='predictions')

        with tf.variable_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target_onehot)
            self.cost = tf.reduce_mean(loss)
            tf.summary.scalar(name="cnn_loss", tensor=self.cost)
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        with tf.variable_scope('accuracy'):
            self.correct_pred = tf.cast(tf.equal(self.prediction, tf.argmax(self.target_onehot, 1)), dtype=tf.float32)
            self.correct_num = tf.reduce_sum(tf.cast(self.correct_pred, tf.float32))
            self.accuracy = tf.reduce_mean(self.correct_num, name='accuracy')
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
        id2labels = dict(zip(self.data.labels.values(), self.data.labels.keys()))
        labels = list(map(id2labels.get, results))
        print(labels)
