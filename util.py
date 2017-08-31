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
import random
import jieba
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

# import glove
# class Config:



# 分解为测试集和训练集
def write_csv(per,is_char=True):
    if not os.path.exists('./data/wdata.csv'):
        data = pd.read_csv('./data/data.csv')
        index = list(data.index)
        train_index = random.sample(index, k=int(per * len(data)))
        test_index = [i for i in index if i not in train_index]
        train = data.loc[train_index]
        test = data.loc[test_index]
        train.to_csv('./data/train.csv', index=False)
        test.to_csv('./data/test.csv', index=False)
        if not is_char:
            for i in ['data.csv', 'train.csv', 'test.csv']:
                df = pd.read_csv('./data/' + i,encoding='utf-8')
                old_text = df['text']
                new_text = [' '.join(i) for i in segment(old_text)]
                label = df['label']
                pd.DataFrame({'text': new_text, 'label': label}).to_csv('./data/w' + i, index=False,encoding='utf-8')


# 分词模块
def segment(corpus, HMM=False):

    jb = jieba.Tokenizer('./utils/dict.txt')
    jb.load_userdict('./utils/user_dict.txt')
    if type(corpus)==str:
        return list(jb.cut(corpus))
    result_list = [list(jb.cut(line, HMM=HMM)) for line in corpus]
    return result_list


def w2v(size=128, is_char=True):
    if is_char:
        if not os.path.exists('./embedding/single_w2v_vec.pkl'):
            corpus = codecs.open('./utils/corpus.txt','r',encoding='utf-8').readlines()
            input_list = [list(i.strip()) for i in corpus]
            model = gensim.models.Word2Vec(sentences=input_list, size=size, min_count=3)
            model.save('./embedding/single_w2v_model')
            with open('./embedding/single_w2v_vocab.pkl', 'wb') as cc:
                pickle.dump(model.wv.index2word, cc)
            a=np.zeros((1,size),dtype=np.float32)
            a=np.concatenate((a,model.wv.syn0),axis=0)
            with open('./embedding/single_w2v_vec.pkl', 'wb') as cc:
                pickle.dump(a, cc)



    else:
        if not os.path.exists('./embedding/word_w2v_vec.pkl'):
            seg = './utils/corpus_seged.pkl'
            if os.path.exists(seg):
                with open(seg, 'rb') as pk:
                    input_list = pickle.load(pk)
            else:
                cor = codecs.open('./utils/corpus.txt','r',encoding='utf-8').readlines()
                cor = [i.strip() for i in cor]
                input_list = segment(cor)
                with open(seg, 'wb') as f:
                    pickle.dump(input_list, f)
            model = gensim.models.Word2Vec(sentences=input_list, size=size, min_count=3)
            model.save('./embedding/word_w2v_model')
            with open('./embedding/word_w2v_vocab.pkl', 'wb') as cc:
                pickle.dump(model.wv.index2word, cc)
            a = np.zeros((1,size), dtype=np.float32)
            a = np.concatenate((a, model.wv.syn0), axis=0)
            with open('./embedding/word_w2v_vec.pkl', 'wb') as cc:
                pickle.dump(a, cc)


def glv(path, size, mode):
    # TODO
    pass


# 数据处理主体部分
def data_prepare(is_char, reset=False, percent=0.9):
    if reset or not os.path.exists('./data/train.csv'):
        print('开始拆分数据集...')
        write_csv(percent,is_char)
    # 生成分词后的数据集


    # 生成标签pkl
    if not os.path.exists('./utils/labels.pkl'):
        data = pd.read_csv('./data/data.csv')
        a = list(set(list(data['label'])))
        mdict = dict(zip(a, range(len(a))))
        with open('./utils/labels.pkl', 'wb') as f:
            pickle.dump(mdict, f)
    w2v(is_char=is_char)


# 独立的特征选择

class CorpusLoader():
    def __init__(self,chi=None):
        write_csv(0.9,is_char=False)
        train_df=pd.read_csv('./data/wtrain.csv',encoding='utf-8')

        train_text=list(train_df['text'])
        train_label=list(train_df['label'])

        test_df=pd.read_csv('./data/wtest.csv',encoding='utf-8')
        test_text=list(test_df['text'])
        test_label=list(test_df['label'])
        self.vec=TfidfVectorizer()
        self.train_X=self.vec.fit_transform(train_text).toarray()
        self.test_X=self.vec.transform(test_text).toarray()
        with open('./utils/labels.pkl', 'rb') as f:
            self.labels = pickle.load(f)
        self.label_size = len(self.labels)
        self.to_label=dict(zip(self.labels.values(),self.labels.keys()))
        self.train_y=np.array(list(map(self.labels.get,train_label)))
        self.test_y=np.array(list(map(self.labels.get,test_label)))

        self.vocab=self.vec.get_feature_names()
        if chi:
            self.ch2 = SelectKBest(chi2, k=chi)
            self.train_X = self.ch2.fit_transform(self.train_X, self.train_y)
            self.test_X = self.ch2.transform(self.test_X)
    def get_train_data(self):
        return self.train_X,self.train_y
    def get_test_data(self):
        return self.test_X,self.test_y
    def to_tfidf(self,text):
        text=' '.join(segment(text))
        return self.ch2.transform(self.vec.transform([text]))




class DataLoader:
    def __init__(self,
                 is_char,
                 batch_size=128,
                 seq_length=20,
                 ):
        print('正在准备数据...')
        data_prepare(is_char=is_char)
        print('数据已就绪')
        self.is_char = is_char
        self.batch_size = batch_size
        self.seq_length = seq_length

        with open('./utils/labels.pkl', 'rb') as f:
            self.labels = pickle.load(f)
        self.label_size = len(self.labels)
        if self.is_char:
            self.embedding_path='./embedding/single_w2v_vec.pkl'
            with open('./embedding/single_w2v_vocab.pkl', 'rb') as f:
                self.chars = pickle.load(f)
        else:
            self.embedding_path='./embedding/word_w2v_vec.pkl'
            with open('./embedding/word_w2v_vocab.pkl', 'rb') as f:
                self.chars = pickle.load(f)
        print('总词个数:%d,前50个词:'%len(self.chars),self.chars[:50])
        self.vocab_size = len(self.chars) + 1

        self.vocab = dict(zip(self.chars, range(1, len(self.chars) + 1)))

        if is_char:
            train_file = os.path.join('data', 'train.csv')
            test_file = os.path.join('data', 'test.csv')

        else:
            train_file = os.path.join('data', 'wtrain.csv')
            test_file = os.path.join('data', 'wtest.csv')

        train_data = pd.read_csv(train_file, encoding='utf8')
        test_data = pd.read_csv(test_file, encoding='utf8')
        self.text_length = len(train_data['text'])
        train_tensor_x = np.array(list(map(self.transform, train_data['text'])))
        train_tensor_y = np.array(list(map(self.labels.get, train_data['label'])))
        test_tensor_x = np.array(list(map(self.transform, test_data['text'])))
        test_tensor_y = np.array(list(map(self.labels.get, test_data['label'])))
        self.train_tensor = np.c_[train_tensor_x, train_tensor_y].astype(int)
        np.random.shuffle(self.train_tensor)
        self.test_tensor = np.c_[test_tensor_x, test_tensor_y].astype(int)
        self.reset_batch_pointer()

    def transform(self, d):
        if self.is_char:
            word_list = list(d)
        else:
            word_list = d.split(' ')

        new_d = list(map(self.vocab.get, word_list[:self.seq_length]))
        # 不在词表中的词设为0,长度不足的补0
        new_d = list(map(lambda i: i if i else 0, new_d))
        if len(new_d) < self.seq_length:
            new_d = new_d + [0] * (self.seq_length - len(new_d))
        return new_d

    def reset_batch_pointer(self):
        self.num_batches = int(self.train_tensor.shape[0] / self.batch_size)
        if self.num_batches == 0:
            assert False, 'Not enough data, make batch_size small.'
        tensor = self.train_tensor[:self.num_batches * self.batch_size]
        self.x_batches = np.split(tensor[:, :-1], self.num_batches, 0)
        self.y_batches = np.split(tensor[:, -1], self.num_batches, 0)
        self.train_pointer = 0

    def next_train_batch(self):
        x = self.x_batches[self.train_pointer]
        y = self.y_batches[self.train_pointer]
        self.train_pointer += 1
        return x, y
    def next_test_batch(self):
        x = self.test_tensor[:,:-1]
        y = self.test_tensor[:,-1]
        return x, y


if __name__ == "__main__":
    # a = DataLoader(is_char=True)
    # x, y = a.next_test_batch()
    # print(x)
    print(CorpusLoader(chi=50).get_test_data()[0].shape)
