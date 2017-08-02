#coding=utf-8
import re
import jieba
import jieba.posseg as posseg
import codecs
import pickle


class CorpusSegement:
    def __init__(self):
        self.jb=jieba.Tokenizer('/Users/yangkang/PycharmProjects/TextClassification/utils/dict.txt')
        self.jb.load_userdict('/Users/yangkang/PycharmProjects/TextClassification/utils/newdict.txt')
        # for i in codecs.open('/Users/yangkang/PycharmProjects/TextClassification/utils/newdict.txt','r','utf-8'):
        #     elem=i.strip().split(' ')
        #     if len(elem)==1:
        #
        #         self.jb.add_word(i)
        #     elif len(elem)==2:
        #         print(elem[0])
        #         self.jb.add_word(elem[0], tag=elem[1])
        #     else:self.jb.add_word(elem[0], tag=elem[1],freq=int(elem[2]))




    def single_segment(self,line):

        return list(self.jb.cut(line,HMM=False))
    def segmenter(self,corpus):

        result_list=[self.single_segment(line) for line in corpus]
        cc=codecs.open('seged.txt','w','utf-8')
        for i in result_list:
            cc.write(' '.join(i)+'\n')
        # with open('corpus_seged.pkl','wb') as pk:
        #     pickle.dump(result_list,pk)
        return result_list
import os
def new_data(dir='/Users/yangkang/PycharmProjects/TextClassification/data'):
    b=CorpusSegement()
    for i in os.listdir(dir):
        if 'csv' in i:
            path=os.path.join(dir,i)
            newpath=os.path.join(dir,'w'+i)

            wr=codecs.open(newpath,'w','utf-8')
            rr=codecs.open(path,'r','utf-8')
            wr.write(rr.readline())
            a=rr.readline()
            while len(a)!=0:
                wr.write(' '.join(b.single_segment(a.split(',')[0]))+','+a.split(',')[1])
                a=rr.readline()
            wr.close()
            rr.close()


if __name__=='__main__':
    # b=CorpusSegement()
    #
    # # print(b.single_segment('今天天气不错啊抓不到这个月真的假的流量安心包好像'))
    # cor=codecs.open('/Users/yangkang/PycharmProjects/TextClassification/embedding/corpus.txt').readlines()
    # cor=[i.strip() for i in cor]
    # b.segmenter(cor)
    # with open('/Users/yangkang/PycharmProjects/TextClassification/embedding/corpus_seged.pkl','rb') as pk:
    #     print(pickle.load(pk)[0])
    # for i in csv.reader(codecs.open('/Users/yangkang/PycharmProjects/TextClassification/data/data.csv','r','utf-8'),dialect='text'):
    #     print(i)
    # data=pd.read_csv('/Users/yangkang/PycharmProjects/TextClassification/data/data.csv')
    # print(data['text'][0])
    new_data()




