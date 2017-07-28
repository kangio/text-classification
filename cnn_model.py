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


class Config():
    embeding_size=50
    filter_size=2
    learning_rate=0.01
    num_filters=64
    sequence_length=20

config=Config

class CNNModel():
    def __init__(self, is_training,vocab_size,label_size,seq_length):
        self.vocab_size=vocab_size
        self.label_size=label_size
        self.input_data=tf.placeholder(tf.int64, [None, seq_length])
        self.target=tf.placeholder(tf.int64,[None])
        self.targets=tf.one_hot(self.target,depth=label_size)

        with tf.variable_scope("embedingLayer") as em:
            embeding=tf.get_variable('embeding',[vocab_size,config.embeding_size])
            embeded=tf.nn.embedding_lookup(embeding, self.input_data)
            embeded=tf.expand_dims(embeded,axis=-1)

        with tf.variable_scope('convolution'):
            filter_shape=[config.filter_size,config.embeding_size,1,config.num_filters]
            W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='W')
            b=tf.Variable(tf.constant(0.1,shape=[config.num_filters],dtype=tf.float32),name='b')
            conv=tf.nn.conv2d(embeded,W,strides=[1,1,1,1],padding="VALID",name='conv')
            h=tf.nn.relu(tf.nn.bias_add(conv,bias=b),name='relu')
        with tf.variable_scope('pooling'):
            pooled=tf.nn.max_pool(h,ksize=[1,config.sequence_length-config.filter_size+1,1,1]
                                  ,strides=[1,1,1,1],padding='VALID',name='pooling')


        with tf.variable_scope('outputlayer'):
            pooled_flat=tf.reshape(pooled,shape=[-1,config.num_filters])
            W=tf.Variable(tf.truncated_normal([config.num_filters,label_size]),name='W')
            b=tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[label_size]),name='b')
            self.logits=tf.nn.xw_plus_b(pooled_flat, W, b, name='score')
            self.probs=tf.nn.softmax(self.logits)
            self.predict=tf.argmax(self.logits, 1, name='predictions')

        with tf.variable_scope('loss'):
            loss=tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets)
            self.loss=tf.reduce_mean(loss)
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)

        with tf.variable_scope('accuracy'):
            correct_nums=tf.equal(self.predict,tf.argmax(self.targets,1))
            self.accuracy=tf.reduce_mean(tf.cast(correct_nums,dtype=tf.float32),name='accuracy')





if __name__ == "__main__":
    pass