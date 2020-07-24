import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model.encoding as encoding
import pandas as pd


def id2relation():
    with open('../data/relation2id.txt', 'r') as f:
        str_list = f.read().split('\n')
        str_list = list(filter(None, str_list))
        r_dict = {}
        for s in str_list:
            r_dict[int(s.split()[1])] = s.split()[0]
    return r_dict


class LSTMTagger(torch.nn.Module):
    def __init__(self,embedding_dim,hidden_dim,voacb_size,target_size):
        super(LSTMTagger,self).__init__()
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.voacb_size=voacb_size
        self.target_size=target_size
        #  LSTM 以 word_embeddings 作为输入, 输出维度为 hidden_dim 的隐状态值
        self.lstm=nn.LSTM(self.embedding_dim,self.hidden_dim)
        ## 线性层将隐状态空间映射到标注空间
        self.out2tag=nn.Linear(self.hidden_dim,self.target_size)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 开始时刻, 没有隐状态
        # 各个维度的含义是 (Seguence, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self,inputs):

        # 预处理文本转成稠密向量
        embeds=inputs
        #根据文本的稠密向量训练网络
        out,self.hidden=self.lstm(embeds.view(len(inputs),1,-1),self.hidden)
        #做出预测
        tag_space=self.out2tag(out.view(len(inputs),-1))
        tags=F.log_softmax(tag_space,dim=1)
        return tags


pre_data, predicet_data = encoding.embedding()
raw = pd.read_csv('../data/try.csv', index_col=0)[:3]
for data in predicet_data:
    vocab_size = data.shape[0]
    embedding_size = data.shape[1]
    model=LSTMTagger(embedding_size,100,vocab_size,23)
    loss_function=nn.NLLLoss()
    optimizer=optim.SGD(model.parameters(),lr=0.1)
    # output
    index = predicet_data.index(data)
    r_list = id2relation()
    r_info = ''
    e_info = pre_data[index]
    annotation = raw['annotation'][index]
    with torch.no_grad():
        input_s=torch.tensor(data, dtype=torch.float32)
        tag_s=model(input_s)
        for i in range(tag_s.shape[0]):
            class_index = torch.argmax(tag_s[i])
            relation = r_list.get(int(class_index))
            r_info = r_info + ''.join(['R', str(i), '\t', str(relation)]) + '\t'
            arg1 = list(e_info[i][0].keys())[0]
            arg2 = list(e_info[i][1].keys())[0]
            r_info = r_info + 'Arg1:' + arg1 + '\t' + 'Arg2:' + arg2 + '\n'
    new_annotation = ''.join([annotation, r_info])
    raw['annotation'][index] = new_annotation

raw.to_csv('../data/try_result.csv')
