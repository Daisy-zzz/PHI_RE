import pandas as pd
import jieba
import re
import gensim


def get_data(raw):
    data = pd.read_json(raw)
    return data


def get_entity_pair(data):
    for line in range(len(data)):



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
