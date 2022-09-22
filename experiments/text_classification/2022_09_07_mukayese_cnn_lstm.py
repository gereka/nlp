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
default_rootdir   = os.path.expanduser('~/outputs/nlp/text_classification/mukayese_cnn_lstm')
default_vocabfile = os.path.expanduser('~/data/nlp/Turkish/spelling/mukayese/binary_letters.txt')

class Net(LightningModule):
    #TODO vocab_size should be instantiated from datamodule but couldn't figure out how to do that.
    def __init__(self, vocab_size=100, edim=128, hdim=128, fcdim=256, n_layers=1, drop1=0.5, lr=3e-4, wd=2e-3, schstep=4, gamma=0.95):
        super().__init__()
        self.save_hyperparameters()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.embedding = nn.Embedding(vocab_size+1, edim)
        self.drop = nn.Dropout(p=drop1)
        self.convseq2 = nn.Sequential(
            nn.Conv1d(in_channels=edim, out_channels=edim, kernel_size=2, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.Conv1d(in_channels=edim, out_channels=edim, kernel_size=2, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.Conv1d(in_channels=edim, out_channels=edim, kernel_size=2, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.Conv1d(in_channels=edim, out_channels=edim, kernel_size=2, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=2, padding=1),
            )
        self.convseq3 = nn.Sequential(
            nn.Conv1d(in_channels=edim, out_channels=edim, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3, padding=1),
            nn.Conv1d(in_channels=edim, out_channels=edim, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3, padding=1),
            )

        self.fcseq = nn.Sequential(
            nn.Linear(edim, fcdim),
            nn.LeakyReLU(),
            nn.Dropout(p=drop1),            
            nn.Linear(fcdim, fcdim),
            nn.LeakyReLU(),
            nn.Dropout(p=drop1),            
            nn.Linear(fcdim, 1),
            )

        self.rnn = nn.LSTM(input_size = edim, hidden_size = hdim, num_layers = n_layers,
                           batch_first=True, dropout=drop1, bidirectional=True)
        self.rnnfc = nn.Linear(2*edim,1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.drop(x)
        x1, _ = self.rnn(x)
        x = x.permute(0,2,1)
        x2 = self.convseq2(x).mean(dim=2)
        x3 = self.convseq3(x).mean(dim=2)
        x = 0.6*self.fcseq((x2+x3)/2) + 0.4*self.rnnfc(x1.mean(dim=1))
        return x.squeeze(1)
    
    def training_step(self, batch, batch_idx):
        x, y, l = batch
        logits = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        out = (logits>=0).long()
        self.train_acc(out, y.long())
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, l = batch
        logits = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        out = (logits>=0).long()
        self.val_acc(out, y.long())
        self.log("val/acc", self.val_acc)
        self.log("val/loss", loss)

    def test_step(self, batch, batch_idx):
        x, y, l = batch
        logits = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        out = (logits>=0).long()
        self.test_acc(out, y.long())
        self.log("test/acc", self.test_acc)
        self.log("test/loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        return [optimizer], [torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.schstep, gamma=self.hparams.gamma)]

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
        self.l2i = {l:i for i,l in enumerate(self.labels)}
        return [self.l2i[l] for l in labser]
    
    def setup(self, stage=None):
        self.load_vocab()
        data = pd.read_csv(self.hparams.datafile)
        data['text'] = data['text'].apply(self.tokenize).apply(self.index_tokens)
#        data['label'] = self.encode_labels(data['label'])
        
        train, test = train_test_split(data,  test_size=0.2,  stratify=data.label,  random_state=100)
        train, val  = train_test_split(train, test_size=0.16, stratify=train.label, random_state=100)

        self.train = list(zip(train['text'].tolist(),train['label'].tolist()))
        self.val   = list(zip(val['text'].tolist(),val['label'].tolist()))
        self.test  = list(zip(test['text'].tolist(),test['label'].tolist()))
   
    def collate_fn(self, batch):
        text, label =  zip(* batch)
        text, length = self.pad_indexlists(text)
        return torch.tensor(text), torch.tensor(label), torch.tensor(length)
    
        
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
            