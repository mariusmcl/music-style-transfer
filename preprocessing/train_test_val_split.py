import os
import random
import sys
from pathlib import Path
from shutil import copy
import numpy as np

import dataset_factory as data
from utils.logger import Logger

logger = Logger('__name__', 'logs/data_split.log')


def copy_files(files, from_path, to_path):
    for f in files:
        out_file_path = to_path / f.relative_to(from_path)
        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        copy(f, out_file_path)


def split_data(input_path, output_path, split_ratio, file_types=None):
    assert sum(split_ratio) == 1.0 and all(np.array(split_ratio) > 0)

    if file_types is None:
        file_types = ['wav']

    input_files = data.EncodedFilesDataset.filter_paths(input_path.glob('**/*'), file_types)
    random.shuffle(input_files)

    logger.info(f'Found {len(input_files)} files')

    train_ratio, val_ratio, test_ratio = split_ratio
    n_train = int(len(input_files) * train_ratio)
    n_val = int(len(input_files) * val_ratio)
    n_test = int(len(input_files) * test_ratio)

    logger.info('Split as follows: Train - %s, Validation - %s, Test - %s', n_train, n_val, n_test)

    copy_files(input_files[:n_train], input_path, output_path / 'train')
    copy_files(input_files[n_train:n_train + n_val], input_path, output_path / 'val')
    copy_files(input_files[n_train + n_val:], input_path, output_path / 'test')


def main():
    random.seed(42)

    in_dir = Path("../dataset/musicnet/parsed")
    out_dir = Path("../dataset/musicnet/split")
    domains = os.listdir(in_dir)
    split_ratios = [0.8, 0.1, 0.1]  # [train, val, test]

    for domain in domains:
        input_path = in_dir / domain
        output_path = out_dir / domain

        split_data(input_path, output_path, split_ratios)


if __name__ == '__main__':
    main()
