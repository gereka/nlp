"""given a path this should download and create a csv for the 1150haber dataset."""

import argparse
from glob import glob
import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split

#TODO write actual download code.

def create_csv(path):
    """Given downloaded model's root path this will create a csv file from it and save it there"""
    raw_texts = os.path.join(path, 'raw_texts')
    fnames = {re.search('[^/]+$', dir)[0]: sorted(glob(dir+'/*'))
              for dir in sorted(glob(os.path.join(raw_texts,'*')))}

    data = []
    for label, files in fnames.items():
        for file in files:
            with open(file, encoding='iso-8859-9') as inpf:
                text = inpf.read().replace("\'", "").replace('\x92', "").replace('\x91', "").replace("\n", " ")
                data.append({'label':label, 'text': text, 'filename':file})
    
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(path, '1150haber.csv'))
    traindev, test = train_test_split(data,test_size = 200, random_state=1848, shuffle=True, stratify=data.label)
    traindev.to_csv(os.path.join(path, '1150haber_traindev.csv'))
    test.to_csv(os.path.join(path, '1150haber_test.csv'))
    train,    dev  = train_test_split(traindev, test_size = 200, random_state=1848, shuffle=True, stratify=traindev.label)
    train.to_csv(os.path.join(path, '1150haber_train.csv'))
    dev.to_csv(os.path.join(path, '1150haber_dev.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and create dataset')
    parser.add_argument('path', type=str, help='where you want to download the dataset')
    args = parser.parse_args()
    create_csv(args.path)
