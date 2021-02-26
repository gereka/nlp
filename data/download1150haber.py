"""given a path this should download and create a csv for the 1150haber dataset."""

import argparse
from glob import glob
import os
import re

def create_csv(path):
    """Given downloaded model's root path this will create a csv file from it and save it there"""
    raw_texts = os.path.join(path, 'raw_texts')
    fnames = {re.search('[^/]+$', dir)[0]: sorted(glob(dir+'/*'))
              for dir in sorted(glob(os.path.join(raw_texts,'*')))}
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and create dataset')
    parser.add_argument('path', type=str, help='where you want to download the dataset')
    args = parser.parse_args()
    create_csv(args.path)
