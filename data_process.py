import re
import random 
import os
import json
import sys
from collections import Counter

TEST_SIZE = 0.1
MAX_VOCAB_SIZE = 19996
MIN_FREQ = 1

if not os.path.exists('data'):
    os.makedirs('data/')
    
with open('data/pure_so_titles.txt','r') as f:
    queries = f.readlines()

test_list = random.sample(range(len(queries)), int(TEST_SIZE*len(queries)))
query_train = [q for i,q in enumerate(queries) if i not in test_list]
query_test = [q for i,q in enumerate(queries) if i in test_list]

word_counter = Counter(' '.join(query_train).split())
print(len(word_counter))
common_words = ['<PAD>','<UNK>','<SOS>','<EOS>'] + [w for w,i in word_counter.most_common(MAX_VOCAB_SIZE) if i > MIN_FREQ]
word_vocab = {w:i for i,w in enumerate(common_words)}
print(len(common_words))


def save_queries(path,vocab):
    with open(path,'w') as f:
        save_txt = []
        for q in query_train:
            words = q.split()
            new_q = ' '.join([str(vocab.get(w,1)) for w in words])
            save_txt.append(new_q)
        f.write('\n'.join(save_txt))


save_queries('data/so_train.txt', word_vocab)
# save_queries('data/so_test.txt', word_vocab)

with open('data/word_vocab.json','w') as f:
    json.dump(word_vocab,f)
