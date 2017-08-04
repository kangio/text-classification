#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0
@author: yangkang
@license: Apache Licence 
@contact: ghooo55@gmail.com
@site: http://abc.com/
@software: PyCharm
@file: train.py
@time: 2017/7/26 上午9:49
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import time
import argparse
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from cnn_model import CNNModel
from util import DataLoader
# from newutil import DataLoader
from lstm_model import LSTMModel
from config import Config

data_loader = DataLoader(is_training=True)
vocab_size = data_loader.vocab_size
label_size = data_loader.label_size

config = Config
lstm_checkpoint_path = 'save/lstm/model.ckpt'
cnn_checkpoint_path = 'save/cnn/model.ckpt'

import os

for i in ['save/cnn', 'save/lstm']:
    if not os.path.exists(i):
        os.mkdir(i)


def train(model_name):
    # data_loader数据获取模块
    # 数据大小
    # label个数(类数)
    # vocab:字到编号的字典
    # labels:类标签到编号的字典
    if model_name == 'lstm':
        model = LSTMModel(is_training=True, vocab_size=vocab_size, label_size=label_size)
        checkpoint_path = lstm_checkpoint_path

    elif model_name == 'cnn':
        model = CNNModel(is_training=True, vocab_size=vocab_size, label_size=label_size)
        checkpoint_path = cnn_checkpoint_path
    else:
        return

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs', sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables())

        for e in range(config.num_epochs):
            # if e > 0: tf.get_variable_scope().reuse_variables()
            # sess.run(tf.assign(model.lr, config.learning_rate * (config.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            total_correct_nums = 0

            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.lr: config.learning_rate * (config.decay_rate ** e), model.input_data: x,
                        model.targets: y}
                train_loss, _, correct_num = sess.run([model.cost, model.optimizer, model.correct_num], feed_dict=feed)
                end = time.time()
                total_correct_nums += correct_num
                if (e * data_loader.num_batches + b + 1) % config.save_every == 0 \
                        or (e == config.num_epochs - 1 and b == data_loader.num_batches - 1):
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b + 1)

                if b == data_loader.num_batches - 1:
                    print('{}/{} (epoch {}), train_loss = {:.3f}, accuracy = {:.3f}, time/batch = {:.3f}' \
                          .format(e * data_loader.num_batches + b + 1,
                                  config.num_epochs * data_loader.num_batches,
                                  e + 1,
                                  train_loss,
                                  total_correct_nums / data_loader.text_length,
                                  end - start))
                    res = sess.run(merged, feed_dict=feed)
                    writer.add_summary(res, e)


if __name__ == "__main__":
    train(config.model)
    # lstm_train()
    # print(len(data_loader.next_batch()[0]))
