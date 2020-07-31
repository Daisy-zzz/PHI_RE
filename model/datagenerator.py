import pandas as pd
import jieba
import re
import gensim
import numpy as np
from sklearn.utils import shuffle
from ast import literal_eval
import math

#jieba.load_userdict('../data/user_dict.txt')
# jieba.add_word('<e>')
# jieba.add_word('</e>')
# jieba.add_word('<unit>')

def str_insert(str, pos, insert):
    str_list = list(str)
    str_list.insert(pos, insert)
    return ''.join(str_list)


def list_find_str(l, str):
    for i in range(len(l)):
        if l[i].find(str) > 0:
            return i
    return -1


def get_data(raw):
    data = pd.read_json(raw)
    return data


def parse_data(data):
    # annotation each record:T49	E95f2a617 5 9	入院情况
    # e: {'T49': ['E95f2a617', '入院情况'], ···}

    for index, line in data.iterrows():
        # process annotation
        e = {}
        add_pos = 0
        #print(line['annotation'].split('\r'))
        line['annotation'] = line['annotation'].replace('\r', '')
        line['content'] = line['content'].replace('\r', '')
        e_record = line['annotation'].split('\n')
        e_record = list(filter(None, e_record))
        for record in e_record:
            if record[0] == 'T':
                record = record.split()
                if len(record) < 5:
                    continue
                e_name = record[0]
                e[e_name] = [record[1], record[4]]
                # add tag to each entity in content like <T1/>···<T1>
                b_pos = int(record[2])
                e_pos = int(record[3])
                line['content'] = str_insert(line['content'], b_pos + add_pos, '<{}/>'.format(e_name))
                add_pos = add_pos + 3 + len(e_name)
                line['content'] = str_insert(line['content'], e_pos + add_pos, '<{}>'.format(e_name))
                add_pos = add_pos + 2 + len(e_name)
            elif record[0] == 'R':
                record = record.split()
                r_type = record[1]
                e1 = record[2].split(':')[1]
                e2 = record[3].split(':')[1]
                # e1_type = e.get(e1)[0]
                # e2_type = e.get(e2)[0]
                # e1_text = e.get(e1)[1]
                # e2_text = e.get(e2)[1]
                e[record[0]] = [r_type, e1, e2]
        line['annotation'] = e
        #print(e)
        # parse content
        s_list = list(filter(None, re.split("。[\n\s\u3000）】)\]]*|(?<!。)\n+", line['content'])))
        #print(s_list)
        for sentence in s_list:
            #r = "/(?!>)+|[!_,$&%^*()+\"'?@#|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
            #sentence_new = re.sub(r, '', sentence_unit)
            sentence_new = stringQ2B(sentence)
            s_list[s_list.index(sentence)] = sentence_new
        line['content'] = s_list
    return data


def get_entity_pair(parsed_data):
    train_data = pd.DataFrame(columns=['e1type', 'e2type', 'sentence', 'relation'])
    pre_data = []  #pd.DataFrame(columns=['e1', 'e2', 's'])
    for i in range(len(parsed_data)):
        e_dict = parsed_data['annotation'][i]
        s_list = parsed_data['content'][i]
        key_list = []
        for key in e_dict.keys():
            if key[0] == 'R':
                key_list.append(key)
        pre_data_line = []
        for j in range(len(key_list)):
            e1 = e_dict.get(key_list[j])[1]
            e2 = e_dict.get(key_list[j])[2]
            idx1 = list_find_str(s_list, '<' + e1 + '>')
            idx2 = list_find_str(s_list, '<' + e2 + '>')
            # ? how to concat two different sentences
            sentence = s_list[idx1] if idx1 == idx2 else s_list[idx1] + s_list[idx2]
            sentence = re.sub('<' + e1 + '>', '</e>', sentence)
            sentence = re.sub('<' + e2 + '>', '</e>', sentence)
            sentence = re.sub('<' + e1 + '/>', '<e>', sentence)
            sentence = re.sub('<' + e2 + '/>', '<e>', sentence)
            sentence = re.sub('<T[0-9]+/*>', '', sentence)

            # r_unit = "[0-9]+[a-zA-Z]+/*[a-zA-Z]*"
            r_unit = "[0-9][0-9a-zA-Z^/%*&#$.]*[a-zA-Z%℃]"
            sentence = re.sub(r_unit, '<unit>', sentence)
            sentence = seg_depart(sentence)
            r_type = e_dict.get(key_list[j])[0]
            if e_dict.get(e1) and e_dict.get(e2):
                e1_type = e_dict.get(e1)[0]
                e1_text = e_dict.get(e1)[1]
                e2_type = e_dict.get(e2)[0]
                e2_text = e_dict.get(e2)[1]
                train_data.loc[len(train_data)] = [e1_type, e2_type, sentence, r_type]
    return train_data


def get_non_data(train_data):
    non_data = pd.read_csv('../data/data_non.csv').dropna(axis=0)
    for index, line in non_data.iterrows():
        e1text = line['Entity1']
        e2text = line['Entity2']
        e1type = line['E1type']
        e2type = line['E2type']
        rtype = line['Relation']
        sentence = line['Sentence'].replace('\r', '').replace('\n', '')

        idx1 = sentence.find(e1text)
        s_list = list(sentence)
        s_list.insert(idx1, '<e>')
        s_list.insert(idx1 + len(e1text) + 1, '</e>')
        sentence_1 = ''.join(s_list)

        idx2 = sentence_1.find(e2text)
        s_list = list(sentence_1)
        s_list.insert(idx2, '<e>')
        s_list.insert(idx2 + len(e2text) + 1, '</e>')
        new_sen = ''.join(s_list)

        r_unit = "[0-9][0-9a-zA-Z^/%*&#$.]*[a-zA-Z%℃]"
        new_sen = re.sub(r_unit, '<unit>', new_sen)
        new_sen = stringQ2B(new_sen)
        new_sen = seg_depart(new_sen)
        train_data.loc[len(train_data)] = [e1type, e2type, new_sen, rtype]
    return train_data


def Q2B(uchar):
    """单个字符 全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e: #转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)


def stringQ2B(ustring):
    """把字符串全角转半角"""
    return "".join([Q2B(uchar) for uchar in ustring])


# 创建停用词列表
def get_stopwords_list():
    stopwords = [line.strip() for line in open('data/cn_stopwords.txt', encoding='UTF-8').readlines()]
    return stopwords


# 对句子进行中文分词
def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    jieba.add_word('<e>', freq=3, tag='n')
    jieba.add_word('</e>', freq=3, tag='n')
    jieba.add_word('<unit>', freq=3, tag='n')
    jieba.re_han_default = re.compile('(.+)', re.U)
    sentence_depart = jieba.lcut(sentence, HMM=False)
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


if __name__ == '__main__':
    #raw = pd.read_csv('../data/train_data.csv', index_col=0)
    #raw = raw.dropna(axis=0)
    #parsed_data = parse_data(raw)
    #print(parsed_data)
    #train_data = get_entity_pair(parsed_data)
    #print(train_data)

    list_train = []
    list_test = []
    #get_data = get_non_data(train_data)
    d1 = pd.read_pickle('../data/dev.pkl')
    d2 = pd.read_pickle('../data/test.pkl')
    print(len(d1), len(d2))
    # get_data = pd.concat([d1, d2])
    # print('data__ok')
    # while len(list_test) is not 21 or len(list_test) is not 21:
    #     list_test.clear()
    #     list_train.clear()
    #     data = shuffle(d2).reset_index(drop=True)
    #     train = data.iloc[0: int(len(data) * 0.5)]
    #     test = data.iloc[int(len(data) * 0.5) + 1:]
    #     for index, line in train.iterrows():
    #         list_train.append(line['relation'])
    #     for index, line in test.iterrows():
    #         list_test.append(line['relation'])
    #     list_train = list(set(list_train))
    #     list_test = list(set(list_test))
    #     print(len(list_train), len(list_test))
    # # pre_data是三重list，(文本行数，每行的实体对数，2)
    # print(list_test, list_train)
    # train.to_pickle('../data/test.pkl')
    # test.to_pickle('../data/dev.pkl')
    #data = pd.read_pickle('../data/test.pkl')
    # for index, line in data.iterrows():
    #     print(len(line['sentence']))
    #删除sentence过长的部分
    # for index, line in d1.iterrows():
    #     s_list = literal_eval(line['sentence'])
    #     idx1 = max(0, s_list.index('<e>') - 5)
    #     idx2 = min([i for i in range(len(s_list)) if s_list[i] == '</e>'][-1] + 5, len(s_list))
    #     #print(len(d1['sentence'][index]))
    #     d1['sentence'][index] = s_list[idx1: idx2]
    #     #print(len(d1['sentence'][index]))
    # d1.to_pickle('../data/train.pkl')
    # for index, line in d2.iterrows():
    #     s_list = literal_eval(line['sentence'])
    #     idx1 = max(0, s_list.index('<e>') - 5)
    #     idx2 = min([i for i in range(len(s_list)) if s_list[i] == '</e>'][-1] + 5, len(s_list))
    #     d2['sentence'][index] = s_list[idx1: idx2]
    # d2.to_pickle('../data/test.pkl')

