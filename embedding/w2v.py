#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0
@author: yangkang
@license: Apache Licence 
@contact: ghooo55@gmail.com
@site: http://abc.com/
@software: PyCharm
@file: w2v.py
@time: 2017/7/28 下午5:05
"""
#coding=utf-8
import gensim
import os
import codecs
from embedding.corpus_segment import CorpusSegement

import pickle
def save_pickle(model,arg):
    if arg=='w':
        with open('word_vocab_w2v.pkl','wb') as cc:
            pickle.dump(model.wv.index2word,cc)
        with open('word_embed_w2v.pkl','wb') as cc:
            pickle.dump(model.wv.syn0,cc)

    elif arg=='g':
        with open('word_vocab_glv.pkl','wb') as cc:
            pickle.dump(model.wv.index2word,cc)
        with open('word_embed_glv.pkl','wb') as cc:
            pickle.dump(model.wv.syn0,cc)
def single_w2v_model():
    cor = codecs.open('/Users/yangkang/PycharmProjects/TextClassification/embedding/corpus.txt').readlines()
    input_list = [list(i.strip()) for i in cor]
    if not os.path.exists('single_vocab_model'):
        model = gensim.models.Word2Vec(sentences=input_list,size=128,min_count=3)
        model.save('single_vocab_model')
    else:model=gensim.models.Word2Vec.load('single_vocab_model')
    return model

def word_w2v_model():
    if os.path.exists('corpus_seged.pkl'):
        with open('corpus_seged.pkl','rb') as pk:
            input_list=pickle.load(pk)
    else:
        cor=codecs.open('/Users/yangkang/PycharmProjects/TextClassification/embedding/corpus.txt').readlines()
        cor=[i.strip() for i in cor]
        input_list=CorpusSegement() .segmenter(corpus=cor)
    if not os.path.exists('word_vocab_model'):
        model = gensim.models.Word2Vec(sentences=input_list,size=128,min_count=3)
        model.save('word_vocab_model')
    else:model=gensim.models.Word2Vec.load('word_vocab_model')
    return model

def word_glv_model():
    model2 = gensim.models.KeyedVectors.load_word2vec_format('word_vectors.txt', binary=False)
    return model2


def single_glv_model():
    model2 = gensim.models.KeyedVectors.load_word2vec_format('single_vectors.txt', binary=False)
    return model2


# if not os.path.exists('vectors.txt'):
save_pickle('w')
save_pickle('g')






