#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0
@author: yangkang
@license: Apache Licence 
@contact: ghooo55@gmail.com
@site: http://abc.com/
@software: PyCharm
@file: utils.py.py
@time: 2017/7/26 上午9:50
"""
import codecs
import pickle
import os
import numpy as np
import pandas as pd
class DataLoader:
    def __init__(self,is_training,batch_size,seq_length):
        self.batch_size=batch_size
        self.seq_length=seq_length

        label_file=os.path.join('utils','labels.pkl')
        vocab_file=os.path.join('utils','vocab.pkl')
        self.labels=_get_label_file(label_file)
        self.label_size=len(self.labels)
        if vocab_file is not None:
            with open(vocab_file, 'rb') as f:
                self.chars = pickle.load(f)
            self.vocab_size = len(self.chars) + 1

            self.vocab = dict(zip(self.chars, range(1, len(self.chars) + 1)))

        if is_training:
            train_file = os.path.join('data', 'train.csv')

            data = pd.read_csv(train_file, encoding='utf8')
            tensor_x = np.array(list(map(self.transform, data['text'])))
            tensor_y = np.array(list(map(self.labels.get, data['label'])))
            self.tensor = np.c_[tensor_x, tensor_y].astype(int)
            self.reset_batch_pointer()
        else:
            test_file = os.path.join('data', 'test.csv')
            data = pd.read_csv(test_file, encoding='utf8')
            tensor_x = np.array(list(map(self.transform, data['text'])))
            tensor_y = np.array(list(map(self.labels.get, data['label'])))
            self.tensor = np.c_[tensor_x, tensor_y].astype(int)
            self.reset_batch_pointer()


    def transform(self, d):
        new_d = list(map(self.vocab.get, d[:self.seq_length]))

        new_d = list(map(lambda i: i if i else 0, new_d))
        if len(new_d) >= self.seq_length:
            new_d = new_d[:self.seq_length]
        else:
            new_d = new_d + [0] * (self.seq_length - len(new_d))
        return new_d
    def create_batches(self):
        self.num_batches = int(self.tensor.shape[0] / self.batch_size)
        if self.num_batches == 0:
            assert False, 'Not enough data, make batch_size small.'

        np.random.shuffle(self.tensor)
        tensor = self.tensor[:self.num_batches * self.batch_size]
        self.x_batches = np.split(tensor[:, :-1], self.num_batches, 0)
        self.y_batches = np.split(tensor[:, -1], self.num_batches, 0)

    def reset_batch_pointer(self):
        self.create_batches()
        self.pointer = 0

    def next_batch(self):
        x = self.x_batches[self.pointer]
        y = self.y_batches[self.pointer]
        self.pointer += 1
        return x, y


def _get_label_file(label_file):
    if not os.path.exists(label_file):

        #TODO 写入序列化label文件
        pass
    else:
        with open(label_file, 'rb') as f:
            return pickle.load(f)
if __name__ == "__main__":
    a=DataLoader(is_training=True,batch_size=10,seq_length=5)
    print(a.next_batch())
