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
from config import Config



config=Config

class CNNModel():
    def __init__(self, is_training,vocab_size,label_size):
        self.vocab_size=vocab_size
        self.label_size=label_size
        self.lr=tf.placeholder(tf.float32)
        with tf.name_scope('data_input'):
            self.input_data=tf.placeholder(tf.int64, [None, config.num_steps])
            self.targets=tf.placeholder(tf.int64,[None])
            self.target_onehot=tf.one_hot(self.targets,depth=label_size)

        with tf.variable_scope("embedingLayer"):
            # embeding=tf.get_variable('embeding',[vocab_size,config.embeding_size])
            # embeding=tf.Variable(np.load(config.embeding_path))
            with open(config.embeding_path,'rb') as cc:
                embeding=tf.Variable(pickle.load(cc))
            embeded=tf.nn.embedding_lookup(embeding, self.input_data)
            embeded=tf.expand_dims(embeded,axis=-1)
        total_result = []

        for index,filter_size in enumerate(config.filter_sizes):
            with tf.variable_scope('conv-pooling%d'%index):
                filter_shape=[filter_size,config.hidden_size,1,config.num_filters]
                W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='W')
                tf.summary.histogram(name='conW',values=W)
                b=tf.Variable(tf.constant(0.1,shape=[config.num_filters],dtype=tf.float32),name='b')
                tf.summary.histogram(name='conb',values=b)

                conv=tf.nn.conv2d(embeded,W,strides=[1,1,1,1],padding="VALID",name='conv')
                h=tf.nn.relu(tf.nn.bias_add(conv,bias=b),name='relu')
                pooled=tf.nn.max_pool(h,ksize=[1,config.sequence_length-filter_size+1,1,1]
                                      ,strides=[1,1,1,1],padding='VALID',name='pooling')
                total_result.append(pooled)
        self.h_pool = tf.concat(values=total_result,axis=3)

        with tf.variable_scope('outputlayer'):
            num_filters_total = config.num_filters * len(config.filter_sizes)
            pooled_flat=tf.reshape(self.h_pool,shape=[-1,num_filters_total])
            if is_training:
                pooled_flat=tf.nn.dropout(pooled_flat,keep_prob=0.6)
            W=tf.Variable(tf.truncated_normal([num_filters_total,label_size]),name='W')
            b=tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[label_size]),name='b')
            self.logits=tf.nn.xw_plus_b(pooled_flat, W, b, name='score')
            self.probs=tf.nn.softmax(self.logits)
            self.predict=tf.argmax(self.logits, 1, name='predictions')

        with tf.variable_scope('loss'):
            loss=tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target_onehot)
            self.cost=tf.reduce_mean(loss)
            tf.summary.scalar(name="cnn_loss",tensor=self.cost)
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        with tf.variable_scope('accuracy'):
            self.correct_pred=tf.cast(tf.equal(self.predict,tf.argmax(self.target_onehot,1)),dtype=tf.float32)
            self.correct_num = tf.reduce_sum(tf.cast(self.correct_pred, tf.float32))
            self.accuracy=tf.reduce_mean(self.correct_num,name='accuracy')

if __name__ == "__main__":
    pass