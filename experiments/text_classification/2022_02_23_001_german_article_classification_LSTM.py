import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import math

data_file = '/home/gereka/data/nlp/German/text_classification/nouns_v2.csv'
split_seed = 4232
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = {
    'train_batch_size' : 32,
    'val_batch_size'   : 64,
    'num_epochs'       : 1,
    'lr'               : 1e-5,
    'n_layers'         : 1,
    'edim'             : 100,
}

print('Device is ', device)

def get_data(data_file):
    data = pd.read_csv(data_file)
    letters = pd.Series([l for w in data.noun.to_list() for l in w]).value_counts().rename_axis('letter').reset_index(name='count')
    letters['index'] = range(2,len(letters)+2) 
    PAD = 0
    UNK = 1
    letters.loc[letters['count']<200 ,'index']=UNK
    width = data.noun.str.len().max()
    l2i = {l:i for l,i in zip(letters['letter'].to_list() , letters['index'].to_list())}
    X = torch.tensor([[l2i[l] for l in n] + [PAD]* (width - len(n)) for n in data.noun.to_list()])
    a2i = {'n':0, 'm':1, 'f':2}
    y = torch.tensor(data.article.replace(a2i).to_list())
    return X, y, l2i, a2i

X, y, l2i, a2i  = get_data(data_file)
    
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=split_seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.16, stratify=y_train, random_state=split_seed)


class LSTM(nn.Module):
    def __init__(self, num_emb, edim=100, hdim=100, n_layers=2, drop=0.5):
        super(LSTM,self).__init__()
        self.embedding =  nn.Embedding(max(self.vocab.values())+1, edim)
        self.rnn = nn.LSTM(input_size = edim, hidden_size = hdim, num_layers = n_layers, batch_first=True, dropout=drop, bidirectional=True)
        self.fcL = nn.Linear(2*hdim, 3)
        self.dropout = nn.Dropout(p=drop)
    
    def forward(self, x):
        #lens = torch.count_nonzero(x, dim=1).unsequeeze(dim=1)
        x = self.embedding(x)
        x = self.dropout(x)
        x, _ = self.rnn(x)
        x = x.mean(dim=1)
        return self.fcL(x)



train_loss = []
train_acc = []
val_acc = []

net = LSTM(vocab = l2i, labels = a2i,  n_layers=args['n_layers'] , edim=args['edim']).to('cuda')
#crit = nn.CrossEntropyLoss(weight=cweights)
crit = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(net.parameters(), lr = args['lr'])


for epoch in range(args['num_epochs']):
    print('Epoch ', epoch, end=' ')
    corrects = 0 
    net.train()
    for it in range(math.ceil(len(X_train)/batch_size)):
        X_b = X_train[it*batch_size:(it+1)*batch_size]
        y_b = y_train[it*batch_size:(it+1)*batch_size]
        X_b = X_b.to(device)
        y_b = y_b.to(device)
        opt.zero_grad()
        
        pred = net(X_b)
        loss = crit(pred, y_b)
        train_loss.append(loss.item())
        pred = pred.argmax(dim=1)
        corrects = corrects + (pred==y_b).sum().item()
        loss.backward()
        opt.step()
    train_acc.append(corrects/len(y_train))
    print('train acc = ', train_acc[-1], end=' ')
    
    corrects = 0
    net.eval()
    for it in range(math.ceil(len(X_val)/val_batch_size)):
        X_b = X_val[it*val_batch_size:(it+1)*val_batch_size]
        y_b = y_val[it*val_batch_size:(it+1)*val_batch_size]
        X_b = X_b.cuda()
        y_b = y_b.cuda()
        with torch.no_grad():
            pred = net(X_b)
            pred = pred.argmax(dim=1)
            corrects = corrects + (pred==y_b).sum().item()
    val_acc.append(corrects/len(y_val))
    print('val acc = ', val_acc[-1])
    
            
