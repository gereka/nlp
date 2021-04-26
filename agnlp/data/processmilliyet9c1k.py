"""given a path this should download and create a csv for the 1150haber dataset."""

import argparse
from glob import glob
import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split

def create_csv(path):
    name = 'milliyet9c1k'
    """Given downloaded model's root path this will create a csv file from it and save it there"""
    fnames = {re.search('[^/]+$', dir)[0]: sorted(glob(dir+'/*'))
              for dir in sorted(glob(os.path.join(path,'*')))}

    data = []
    for label, files in fnames.items():
        for file in files:
            with open(file) as inpf:
                #TODO consider moving this preprocessing to the data reader.
                text = inpf.read().replace("\'", "").replace('\x92', "").replace('\x91', "").replace("\n", " ")
                data.append({'label':label, 'text': text, 'filename':file})
    
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(path, f'{name}.csv'), index=False)
    traindev, test = train_test_split(data,test_size = 200, random_state=1848, shuffle=True, stratify=data.label)
    traindev.to_csv(os.path.join(path, f'{name}_traindev.csv'), index=False)
    test.to_csv(os.path.join(path, f'{name}_test.csv'), index=False)
    train,    dev  = train_test_split(traindev, test_size = 200, random_state=1848, shuffle=True, stratify=traindev.label)
    train.to_csv(os.path.join(path, f'{name}_train.csv'), index=False)
    dev.to_csv(os.path.join(path, f'{name}_dev.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and create dataset')
    parser.add_argument('path', type=str, help='where you want to download the dataset')
    args = parser.parse_args()
    create_csv(args.path)
