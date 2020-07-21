import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import gensim
from gensim.models import Word2Vec
import jieba
import random

data = pd.read_csv('data/train_data.csv')
data = data.dropna(axis=0).reset_index(drop=True)
#print(len(data['annotation']))

cols = ['Entity1', 'E1type', 'Entity2', 'E2type', 'Relation', 'Sentence']
new_data = pd.DataFrame(columns=cols)
data_non = pd.DataFrame(columns=cols)
# 处理annotation
eList = []
eDict = {}
rDict = {}

def re2id():
    relation = []
    entity = []
    for j in range(len(data)):
        strList = data['annotation'][j].split('\n')
        # e: {T79:[E320ca3f6,7,9,乳腺],...}
        for i in range(len(strList) - 1):
            str_splited = strList[i].split()
            if len(str_splited) and str_splited[0][0] == 'T':
                entity.append(str_splited[1])
            elif len(str_splited) and str_splited[0][0] == 'R':
                relation.append(str_splited[1])
    entity = list(set(entity))
    relation = list(set(relation))
    e2id = ''
    r2id = ''
    with open("data/entity2id.txt", "w") as f:
        for i in range(len(entity)):
            f.write(entity[i] + '\t' + str(i) + '\n')

    with open("data/relation2id.txt", "w") as f:
        f.write('None' + '\t' + '0' + '\n')
        for i in range(len(relation)):
            f.write(relation[i] + '\t' + str(i + 1) + '\n')
re2id()



def gen_nonType(n):
    j = 0
    while j < n:
        row = np.random.randint(0, 1888)
        strList = data['annotation'][row].split('\n')
        e = {}
        e2r_dict = {}
        entity = []
        for i in range(len(strList) - 1):
            str_splited = strList[i].split()
            if len(str_splited) and str_splited[0][0] == 'T':
                # eList.append(str_splited[-1])
                e[str_splited[0]] = str_splited[1:]
                entity.append(str_splited[0])
            elif len(str_splited) and str_splited[0][0] == 'R':
                relation = str_splited[1]
                e1 = str_splited[2].split(':')[1]
                e2 = str_splited[3].split(':')[1]
                v1 = e.get(e1)
                v2 = e.get(e2)
                e1type = v1[0]
                e2type = v2[0]
                content = data['content'][row]
                entity1 = content[int(v1[1]): int(v1[2])]
                entity2 = content[int(v2[1]): int(v2[2])]
                if e2r_dict.get(relation):
                    temp = e2r_dict.get(relation)
                    temp = temp.append([e1, e2])
                    e2r_dict[relation] = temp
                else:
                    e2r_dict[relation] = [[e1, e2]]
        print(e2r_dict)
        flag = True
        random.shuffle(entity)
        for n1 in range(len(entity)):
            if not flag:
                break
            for n2 in range(n1 + 1, len(entity)):
                if [entity[n1], entity[n2]] not in list(e2r_dict.values()):
                    flag = False
                    j = j + 1
                    relation = 'None'
                    e1 = entity[n1]
                    e2 = entity[n2]
                    v1 = e.get(e1)
                    v2 = e.get(e2)
                    e1type = v1[0]
                    e2type = v2[0]
                    content = data['content'][row]
                    entity1 = content[int(v1[1]): int(v1[2])]
                    entity2 = content[int(v2[1]): int(v2[2])]
                    k = int(v1[1])
                    l = int(v1[2])
                    while 0 < k < len(content) and content[k] != '\n':
                        k = k - 1
                    while 0 < l < len(content) and content[l] != '\n':
                        l = l + 1
                    sentence = content[k: l].strip()
                    data_non.loc[len(data_non)] = [entity1, e1type, entity2, e2type, relation, sentence]
                    break
    data_non.to_csv('data_non.csv', encoding='utf-8')



def gen_new_table():
    max_length = 0
    for j in range(len(data)):
        strList = data['annotation'][j].split('\n')
        # e: {T79:[E320ca3f6,7,9,乳腺],...}
        e = {}
        for i in range(len(strList) - 1):
            str_splited = strList[i].split()
            if len(str_splited) and str_splited[0][0] == 'T':
                # eList.append(str_splited[-1])
                # 一个entity type对应的所有entity
                if eDict.get(str_splited[1]):
                    temp = eDict.get(str_splited[1])
                    temp.append(str_splited[-1])
                    eDict[str_splited[1]] = temp
                else:
                    eDict[str_splited[1]] = [str_splited[-1]]
                #####
                e[str_splited[0]] = str_splited[1: ]
            elif len(str_splited) and str_splited[0][0] == 'R':
                #####
                if rDict.get(str_splited[1]):
                    temp = rDict.get(str_splited[1])
                    temp.append(str_splited[-1])
                    rDict[str_splited[1]] = temp
                else:
                    rDict[str_splited[1]] = [str_splited[-1]]
                #####
                relation = str_splited[1]
                e1 = str_splited[2].split(':')[1]
                e2 = str_splited[3].split(':')[1]
                v1 = e.get(e1)
                v2 = e.get(e2)
                e1type = v1[0]
                e2type = v2[0]
                content = data['content'][j]
                entity1 = content[int(v1[1]): int(v1[2])]
                entity2 = content[int(v2[1]): int(v2[2])]
                k = int(v1[1])
                l = int(v1[2])
                while k > 0 and k < len(content) and content[k] != '\n':
                    k = k - 1
                while l > 0 and l < len(content) and content[l] != '\n':
                    l = l + 1
                contentList = content.split('\n')
                for m in range(len(contentList)):
                    max_length = max(max_length, len(contentList[m]))
                sentence = content[k: l].strip()
                new_data.loc[len(new_data)] = [entity1, e1type, entity2, e2type, relation, sentence]
    #new_data.to_csv('data_1.csv', encoding='utf-8')
    for key, value in eDict.items():
        print(key, len(value))
# gen_new_table()




# one-hot
# values = np.array(eList)
# # label encode
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(values)
# print(integer_encoded)
# # binary encode
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# print(onehot_encoded.shape)

# 分词
# nonword = []
# model = gensim.models.KeyedVectors.load_word2vec_format('data/wordembedding/190721_AAAA_jieba_vec_128.vec')
# for i in range(len(eList)):
#     if eList[i] not in model:
#         seg = jieba.lcut(eList[i], cut_all=True, HMM=True)
#         for j in range(len(seg)):
#             if seg[j] not in model:
#                 nonword.append(seg[j])
#         #nonword.append(eList[i])
# nonword = list(set(nonword))

# print(len(nonword), nonword)