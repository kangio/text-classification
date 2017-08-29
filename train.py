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

# 根据一个分类模型，训练模型后，进行测试
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time.time()
    clf.fit(train_X, train_y)
    train_time = time.time() - t0
    print("train time: %0.3fs" % train_time)
    t0 = time.time()
    pred = clf.predict(test_X)
    test_time = time.time() - t0
    print("test time:  %0.3fs" % test_time)
    score = metrics.accuracy_score(test_y, pred)
    print("accuracy:   %0.3f" % score)
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time
if __name__ == "__main__":
    # data_loader = DataLoader(is_char=True)
    print('模型初始化')

    from util import CorpusLoader

    data = CorpusLoader(chi=1000)
    # train_X, train_y = data.get_train_data()
    # test_X, test_y = data.get_test_data()

    # clf = RandomForestClassifier(n_estimators=100)
    # clf = RidgeClassifier(tol=1e-2, solver="lsqr")
    # clf = Perceptron(n_iter=50)
    # clf = LinearSVC()
    # clf = GradientBoostingClassifier()

    # clf = SGDClassifier(alpha=.0001, n_iter=50,penalty="l1")
    # clf = SGDClassifier(alpha=.0001, n_iter=50,penalty="elasticnet")

    # clf = NearestCentroid()
    # clf = MultinomialNB(alpha=.01)
    # clf = BernoulliNB(alpha=.01)

    # pipeline模型特征选择和分类模型结合在一起
    # clf = Pipeline([ ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)), ('classification', LinearSVC())])

    # benchmark(clf)
    # model = LSTMModel(data_loader)
    # model = CNNModel(data_loader)
    # model.fit(num_epochs=1)
    # model.test_accuracy()
    # model.predict('还剩多少话费')
    from traditional_models import RandomForest
    model=RandomForest(data)
    model.fit()
    # model.accurancy()
