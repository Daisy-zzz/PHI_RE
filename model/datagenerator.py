import pandas as pd
import jieba
import re
import gensim
import numpy as np


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
        print(e)
        # parse content
        s_list = list(filter(None, re.split("[。！!？?\n]", line['content'])))
        for sentence in s_list:
            r = "/(?!>)+|[!_,$&%^*()+\"'?@#|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
            r_unit = "[0-9]+[a-zA-Z]+/*[a-zA-Z]*"
            sentence_unit = re.sub(r_unit, '<unit>', sentence)
            sentence_new = re.sub(r, '', sentence_unit)
            s_list[s_list.index(sentence)] = sentence_new
        line['content'] = s_list
    return data


def get_entity_pair(parsed_data):
    train_data = pd.DataFrame(columns=['e1type', 'e1text', 'e2type', 'e2text', 'sentence', 'relation'])
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

            r_type = e_dict.get(key_list[j])[0]
            e1_type = e_dict.get(e1)[0]
            e1_text = e_dict.get(e1)[1]
            e2_type = e_dict.get(e2)[0]
            e2_text = e_dict.get(e2)[1]
            train_data.loc[len(train_data)] = [e1_type, e1_text, e2_type, e2_text, sentence, r_type]
    return train_data


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


if __name__ == '__main__':
    raw = pd.read_csv('../data/train_data.csv', index_col=0)[: 3]
    parsed_data = parse_data(raw)
    #print(parsed_data)
    train_data = get_entity_pair(parsed_data)
    print(train_data)
    # pre_data是三重list，(文本行数，每行的实体对数，2)
    train_data.to_csv('train.csv')


