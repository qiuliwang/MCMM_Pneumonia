# -*- coding:utf-8 -*-


from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import csvTools

path = get_tmpfile("word2vec.model") #创建临时文件

contexts = csvTools.readCSV('./transformedlistd.csv')

sentences = []



for onecontext in contexts:
    words = onecontext[3].split(' ')
    sentences.append(words)

#加载模型
# model = Word2Vec.load("word2vec.model")
model = Word2Vec(sentences, size=50, window=3, min_count=1, workers=2)
model.train(words, total_examples=1, epochs=1)

# model.save("word2vec.model")
vector = model['咳嗽', '发热', '畏寒', '活动', '后', '气促', '伴', '心悸', '余天']
print(vector.shape)