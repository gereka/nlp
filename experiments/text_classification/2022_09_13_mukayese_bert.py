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
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

#Default file locations
default_datafile   = os.path.expanduser('~/data/nlp/Turkish/spelling/mukayese/binary.csv')
default_rootdir    = os.path.expanduser('~/outputs/nlp/text_classification/mukayese_bert')
default_pretrained = os.path.expanduser('~/data/models/bert-base-turkish-128k-uncased')

class Net(LightningModule):
    def __init__(self, pretrained=default_pretrained, lr=3e-4, wd=2e-3, schstep=4, gamma=0.95):
        super().__init__()
        self.save_hyperparameters()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.config = AutoConfig.from_pretrained(self.hparams.pretrained, num_labels=2)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams.pretrained, config=self.config)
        
    def forward(self, **inputs):
        return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
#        logits = self(**batch)
#        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        out = logits.argmax(dim=1)
        self.train_acc(out, batch['labels'])
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits
        out = logits.argmax(dim=1)
        self.val_acc(out, batch['labels'])
        self.log("val/acc", self.val_acc)
        self.log("val/loss", loss)

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits
        out = logits.argmax(dim=1)
        self.test_acc(out, batch['labels'])
        self.log("test/acc", self.test_acc)
        self.log("test/loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        return [optimizer], [torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.schstep, gamma=self.hparams.gamma)]

class MukayeseDataModule(LightningDataModule):
    def __init__(self, datafile=default_datafile, pretrained=default_pretrained, num_workers=20 ,batch_size=32, test_batch_size=2):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained, use_fast=True)
        self.PAD_index = self.tokenizer.pad_token_id

    def encode_text(self, ser):
        return self.tokenizer.batch_encode_plus(ser.tolist())


    def pad_indexlist(self, indexlist, length):
        paddinglength = (length - len(indexlist))
        #TODO throw exception if paddinglength is negative
        return indexlist + [self.PAD_index] * paddinglength
        
    def pad_indexlists(self, indexlists):
        length = [len(indexlist) for indexlist in indexlists]
        maxlen = max(length)
        paddedlists = [self.pad_indexlist(indexlist, maxlen) for indexlist in indexlists]
        return paddedlists

    def encode_labels(self, labser):
        self.labels = labser.unique()
        self.l2i = {l:i for i,l in enumerate(self.labels)}
        return [self.l2i[l] for l in labser]
    
    def setup(self, stage=None):
        data = pd.read_csv(self.hparams.datafile)
        encoded = self.encode_text(data['text'])
        for k in encoded.keys():
            data[k] = encoded[k]
        #data['label'] = self.encode_labels(data['label'])
        
        train, test = train_test_split(data,  test_size=0.2,  stratify=data.label,  random_state=100)
        train, val  = train_test_split(train, test_size=0.16, stratify=train.label, random_state=100)

        self.cols = list(encoded.keys()) + ['label']
        self.train = list(zip(*[train[c] for c in self.cols]))
        self.val   = list(zip(*[val[c] for c in self.cols]))
        self.test  = list(zip(*[val[c] for c in self.cols]))
   
    def collate_fn(self, batch):
        result =  {key: val for key,val in zip(self.cols, zip(* batch))}
        result2 = {key: self.pad_indexlists(result[key]) for key in self.cols if key!='label'}
        result2['labels'] = result['label']
        return {key: torch.tensor(val) for key, val in result2.items()}
    
        
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
            
