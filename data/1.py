import re
import jieba
import gensim
from gensim.models import Word2Vec

model = gensim.models.KeyedVectors.load_word2vec_format('wordembedding/190721_AAAA_jieba_vec_128.vec')
word = model.wv.index2word
l = list(word)
print(l)