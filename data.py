import numpy as np
import pandas as pd
import gensim
from gensim.models import Word2Vec
import jieba
import re
import torch
dummy = np.random.rand(1, 128)
data1 = pd.read_csv('data/data.csv').astype(str)
data2 = pd.read_csv('data/data_non.csv').astype(str)
data = pd.concat([data1, data2])
#model = gensim.models.KeyedVectors.load_word2vec_format('data/embedding/190721_AAAA_jieba_vec_128.vec')


# 创建停用词列表
def get_stopwords_list():
    stopwords = [line.strip() for line in open('data/cn_stopwords.txt', encoding='UTF-8').readlines()]
    return stopwords


# 对句子进行中文分词
def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    sentence_depart = jieba.lcut(sentence.strip(), HMM=True)
    return sentence_depart


def remove_digits(input_str):
    punc = u'0123456789.'
    output_str = re.sub(r'[{}]+'.format(punc), '', input_str)
    return output_str


# 去除停用词
def move_stopwords(sentence_list, stopwords_list):
    # 去停用词
    out_list = []
    for word in sentence_list:
        if word not in stopwords_list:
            if not remove_digits(word):
                continue
            if word != '\t':
                out_list.append(word)
    return out_list


# sentence to vector
def sen2vec(sen):
    if sen not in model:
        stopwords = get_stopwords_list()
        sentence_depart = seg_depart(sen)
        # print(sentence_depart)
        sentence_depart = move_stopwords(sentence_depart, stopwords)
        seg_vec = []
        for j in range(len(sentence_depart)):
            if sentence_depart[j] in model:
                seg_vec.append(model[sentence_depart[j]].reshape((1, -1)))
        # 分词后仍不在vector里
        if len(seg_vec) == 0:
            vec = dummy
        else:
            vec = np.average(np.array(seg_vec), axis=0)
    else:
        vec = model[sen].reshape((1, -1))
    return vec


def getFmat():
    fmatrix = 0
    for i in range(len(data)):
        vec_head = sen2vec(data['Entity1'][i])
        vec_tail = sen2vec(data['Entity2'][i])
        vec_sen = sen2vec(data['Sentence'][i])
        vec = np.hstack((vec_head, vec_tail, vec_sen))
        if i == 0:
            fmatrix = vec
        else:
            fmatrix = np.vstack((fmatrix, vec))
    print(fmatrix.shape)
    return fmatrix
