import torch
from nltk.stem import WordNetLemmatizer 

from torch.utils.data import Dataset


class QueryDataset(Dataset):
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    def __init__(self, query, word_vocab, query_len, is_docstring=False):
        super(QueryDataset, self).__init__()
        self.query_len = query_len
        self.word_vocab = word_vocab
        self.query = []
        lemmatizer = WordNetLemmatizer()  
        
        if isinstance(query,str):
            with open(query, 'r') as f:
                lines = f.readlines()
        else:
            lines = query
            
        for q in lines:
            words = q.lower().split()[:query_len]
            if is_docstring:
                if len(words)>0 and word_vocab.get(words[0],self.UNK) == self.UNK:
                    words[0] = lemmatizer.lemmatize(words[0],pos='v')
                words = [word_vocab.get(w,self.UNK) for w in words]+[self.EOS]
            else:
                words = [int(w) for w in words]+[self.EOS]
            padding = [self.PAD for _ in range(query_len - len(words)+2)]
            words.extend(padding)
            self.query.append(words)


    def __len__(self):
        return len(self.query)

    def __getitem__(self, item):
        return torch.tensor(self.query[item]),torch.tensor(self.query[item])

