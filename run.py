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

import time
from cnn_model import CNNModel
from lstm_model import LSTMModel
from util import DataLoader
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from time import time
from util import CorpusLoader

# 根据一个分类模型，训练模型后，进行测试

if __name__ == "__main__":
    print('模型初始化')
    data_loader = DataLoader(is_char=True)
    # data_loader = CorpusLoader(chi=100)

    model = LSTMModel(data_loader)
    # model = CNNModel(data_loader)
    # model.fit(num_epochs=1)
    # model.test_accuracy()
    # model.predict('还剩多少话费')

    from traditional_models import RandomForest, SVM, SGD

    # model = SGD(data_loader)
    model.fit()
    model.predict('还剩多少话费')
    # model.accurancy()
