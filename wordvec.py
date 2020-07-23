from gensim.test.utils import common_texts,get_tmpfile
from gensim.models import Word2Vec
#getting the word2vec model to train the other vectors
#as given in the problem that either polyglot or word2vec embeddings can be chosen
path=get_tmpfile("word2vec.model")

model=Word2Vec(common_texts,size=100,window=5,min_count=1,workers=4)
myvector = model.wv
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

filename = get_tmpfile("vectors.kv")
#myvector.save(filename)
myvector = KeyedVectors.load(filename, mmap='r')
from gensim.test.utils import datapath
wv_from_text = KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'), binary=False)

import gensim.downloader as api
myvector = api.load("glove-twitter-50") 

print("similar word for Dog :")
print(myvector.similar_by_word('dog',1))
print("-------------------------")
print("similar word for Whale :")
print(myvector.similar_by_word('whale',1))
print("-------------------------")
print("similar word for before :")
print(myvector.similar_by_word('before',1))
print("-------------------------")
print("similar word for however :")
print(myvector.similar_by_word('however',1))
print("-------------------------")
print("similar word for fabricate :")
print(myvector.similar_by_word('fabricate',1))
print("-------------------------")

#one vector is created which is computing the addition and subtraction of the other vectors
diff_vectors = myvector.wv['cat']  + myvector.wv['puppy'] - myvector.wv['dog']

#the result stored in diff_vectors can be used to get the insight
print(myvector.most_similar(positive=[diff_vectors])[1])


#the result came by adding and subtracting the vector is almost same with the result came by the predefined function
print(myvector.most_similar(positive=['cat', 'puppy'],negative=['dog'],topn=1))

#by using the predefined function of addition and subtraction in genesis
#getting the top 3 similar words
print(myvector.most_similar(positive=['cat', 'puppy'],negative=['dog'],topn=3))

