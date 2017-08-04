#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0
@author: yangkang
@license: Apache Licence 
@contact: ghooo55@gmail.com
@site: http://abc.com/
@software: PyCharm
@file: config.py
@time: 2017/8/4 上午9:38
"""

class Config:

    model='cnn'
    word_mode='single'         #otherwise 'word'



    seq_length = 20
    train_batch_size = 128
    learning_rate = 0.001
    max_grad_norm = 5
    decay_rate = 0.9
    num_epochs = 10
    save_every = 200
    init_scale = 0.1
    num_layers = 1
    num_steps = 20
    hidden_size = 128
    keep_prob = 1.0
    lr_decay = 0.5
    test_batch_size = 30

    #cnn参数
    filter_sizes=[2]
    num_filters=32
    sequence_length=20
    embeding_path='/Users/yangkang/PycharmProjects/TextClassification/utils/embed_w2v.pkl'

    lstm_save_dir = 'save/lstm'
    cnn_save_dir = 'save/cnn'


