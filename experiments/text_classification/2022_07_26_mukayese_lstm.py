import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.cli import LightningCLI
from torchmetrics import Accuracy
from sklearn.model_selection import train_test_split
from datetime import datetime

#Default file locations
default_datafile  = os.path.expanduser('~/data/nlp/Turkish/spelling/mukayese/binary.csv')
default_rootdir   = os.path.expanduser('~/outputs/nlp/text_classification/mukayese_lstm')
default_vocabfile = os.path.expanduser('~/data/nlp/Turkish/spelling/mukayese/binary_letters.txt')

class Net(LightningModule):
    #TODO vocab_size should be instantiated from datamodule but couldn't figure out how to do that.
    def __init__(self, vocab_size=100, edim=100, hdim=100, n_layers=1, drop1=0.5, drop2=0.5, lr=3e-4, wd=2e-1, gamma=0.7):
        super().__init__()
        self.save_hyperparameters()
        self.test_acc = Accuracy()
        self.embedding = nn.Embedding(vocab_size+1, edim)
        self.dropout = nn.Dropout(p=drop1)
        self.rnn = nn.LSTM(input_size = edim, hidden_size = hdim, num_layers = n_layers,
                           batch_first=True, dropout=drop2, bidirectional=True)
        self.fcL = nn.Linear(2*edim,1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x, _ = self.rnn(x)
        x = x.mean(dim=1)
        return self.fcL(x)
    
    def training_step(self, batch, batch_idx):
        x, y, l = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.long())
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.test_acc(logits, y)
        self.log("train/acc", self.test_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, l = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.long())
        self.test_acc(logits, y)
        self.log("val/acc", self.test_acc)
        self.log("val/loss", loss)

    def test_step(self, batch, batch_idx):
        x, y, l = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.long())
        self.test_acc(logits, y)
        self.log("test/acc", self.test_acc)
        self.log("test/loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        return [optimizer], [torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.hparams.gamma)]

class MukayeseDataset(Dataset):
    def __init__(self, filename):
        self.data = pd.read_csv(filename)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row.label, row.text

    
class MukayeseDataModule(LightningDataModule):
    def __init__(self, datafile=default_datafile, vocabfile=default_vocabfile, num_workers=20 ,batch_size=32, test_batch_size=1000):
        super().__init__()
        self.save_hyperparameters()

    def load_vocab(self): 
        with open(self.hparams.vocabfile, 'r') as inpf:
            self.vocab = inpf.read().split('\n')
        self.vocab = ['<UNK>','<PAD>'] + self.vocab
        self.UNK_index = 0
        self.PAD_index = 1
        self.v2i = {w:i for i,w in enumerate(self.vocab)}
    
    def tokenize(self, s):
        return [letter for letter in s]

    def index_tokens(self, tokenlist):
        return [self.v2i.get(t, self.UNK_index) for t in tokenlist]

    def pad_indexlist(self, indexlist, length):
        paddinglength = (length - len(indexlist))
        #TODO throw exception if paddinglength is negative
        return indexlist + [self.PAD_index] * paddinglength
        
    def pad_indexlists(self, indexlists):
        length = [len(indexlist) for indexlist in indexlists]
        maxlen = max(length)
        paddedlists = [self.pad_indexlist(indexlist, maxlen) for indexlist in indexlists]
        return paddedlists, length

    def encode_labels(self, labser):
        self.labels = labser.unique()
        self.l2i = {l:i for i,l in enumerate(labser)}
        return [self.l2i[l] for l in labser]
    
    def setup(self, stage=None):
        self.load_vocab()
        data = pd.read_csv(self.hparams.datafile)
        data['text'] = data['text'].apply(self.tokenize).apply(self.index_tokens)
        data['label'] = self.encode_labels(data['label'])
        
        train, test = train_test_split(data,  test_size=0.2,  stratify=data.label,  random_state=100)
        train, val  = train_test_split(train, test_size=0.16, stratify=train.label, random_state=100)

        self.train = list(zip(train['text'].tolist(),train['label'].tolist()))
        self.val   = list(zip(val['text'].tolist(),val['label'].tolist()))
        self.test  = list(zip(test['text'].tolist(),test['label'].tolist()))
   
    def collate_fn(self, batch):
        text, label = batch
        text, length = self.pad_indexlists(text)
        return text, label, length
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.hparams.batch_size, collate_fn=self.collate_fn,
                                           num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.hparams.test_batch_size, collate_fn=self.collate_fn,
                                           num_workers=self.hparams.num_workers)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.hparams.test_batch_size, collate_fn=self.collate_fn,
                                           num_workers=self.hparams.num_workers)

    
    
class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(EarlyStopping, "estop")
        parser.set_defaults({"estop.monitor": "val/loss", "estop.patience": 20})

def cli_main():
    start = datetime.now()
    print("Starting at ", start)

    trainer_defaults = {
        'gpus':1,
        'default_root_dir': default_rootdir,
    }

    cli = MyLightningCLI(
        Net, MukayeseDataModule, seed_everything_default=438, save_config_overwrite=True, run=False, trainer_defaults=trainer_defaults
    )


    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
    finish = datetime.now()
    print("Finished at ", finish)
    print("Time elapsed", finish-start)

if __name__ == "__main__":
    cli_main()
            
