# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.manifold import TSNE
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.style.use('ggplot')

# read the word vectors
allVec = pd.read_csv('wordVectors.txt', sep=' ', header=None)
allVec = allVec.iloc[:, :-1]

# read vocab
vocab = pd.read_table('vocab.txt', header=None, quoting=3)
vocab.columns = ['word']
vocab = vocab.reset_index()

# read training samples
train = pd.read_csv('train', sep='\t', header=None)
train = train.drop_duplicates()
train.columns = ['word','label']
train.word = train.word.apply(lambda x: str(x).lower())

# pick first 6000 samples
t1 = train[:6000]
vocab_index = pd.merge(t1, vocab, on='word', how='left')
#vocab_index = vocab_index.fillna(0)
vocab_index = vocab_index.dropna()
# make the index from vocab as index to this df
vocab_index.set_index('index')
vocab_index = vocab_index.set_index('index')

# join allVec based on index (to align with the right word in vocab)
vec_label = vocab_index.join(allVec)
vec_label['colors'] = vec_label.label.map(\
    {'O':'w','PER':'b','MISC':'g','ORG':'r','LOC':'c'})

xcols = allVec.columns.tolist()

# TSNE
tsne = TSNE(random_state=37153)
tsne_old = tsne.fit_transform(vec_label[xcols])
plt.scatter(tsne_old[:,0],tsne_old[:,1], c=vec_label.colors)

# read in new vectors after training
newVec = pd.read_csv('newRndVectors.txt', sep=' ', header=None)
newVec = newVec.iloc[:, :-1]

new_vec_label = vocab_index.join(newVec)
new_vec_label['colors'] = new_vec_label.label.map(\
    {'O':'w','PER':'b','MISC':'g','ORG':'r','LOC':'c'})

tsne = TSNE(random_state=37153)
tsne_new = tsne.fit_transform(new_vec_label[xcols])
plt.scatter(tsne_new[:,0],tsne_new[:,1], c=new_vec_label.colors)

flag_no = (tsne_new[:,0] < 0) & (tsne_new[:,1] > 0) & (new_vec_label.colors != 'w')
vec_label.ix[flag_no,['word','colors']].values
'''
Vectors
array([['and', 'r'],
       ['in', 'r'],
       ['in', 'r'],
       ['for', 'r'],
       ['as', 'r'],
       ['on', 'g'],
       ["'s", 'r'],
       [':', 'g'],
       ['said', 'b'],
       ['man', 'b'],
       ['across', 'r'],
       ['mother', 'b'],
       ['ca', 'c'],
       ['centers', 'r'],
       ['jan', 'b'],
       ['ally', 'b'],
       ['sterling', 'b'],
       ['f.a.', 'g']], dtype=object)
'''
flag_no = (tsne_new[:,0] >0) & (tsne_new[:,1] < -2) & (new_vec_label.colors != 'w')
vec_label.ix[flag_no,['word','colors']].values

'''
rndVectors
array([['the', 'r'],
       ['the', 'c'],
       ['the', 'b'],
       ['in', 'r'],
       ['in', 'r'],
       ['on', 'g'],
       ['two', 'r'],
       ['school', 'c'],
       ['play', 'g'],
       ['per', 'b'],
       ['foreign', 'r'],
       ['miles', 'b'],
       ['key', 'b'],
       ['spring', 'b'],
       ['price', 'b'],
       ['criminal', 'r'],
       ['champions', 'g'],
       ['centers', 'r'],
       ['add', 'r'],
       ['erickson', 'b']], dtype=object)
'''
