'''
@File    :   base.py
@Time    :   2023/05/19 17:35:57
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw
@Desc    :   Base class for DiaASQ dataset. The dataset is designed for few-shot learning,
             so the default is to load test set, and then the k few-shot examples are loaded
             from the train set randomly (random seed control shall be implemented in subclasses).
'''
import json
from pathlib import Path
import os
from torch.utils.data import Dataset, DataLoader
# from argparse import ArgumentParser



class BaseDiaAsqDataset(Dataset):
    def __init__(self, src: str, data_root: os.PathLike, train_split_name:str = 'train',
                 test_split_name:str = 'valid'):
        self.src = src # language_source, en or zh
        self.test_split_name = test_split_name
        self.train_split_name = train_split_name
        # eg. data/diaasq/dataset/jsons_en/train.json
        self.data_root = data_root
        self._determine_path()

        self.train_data = self.read_json(self.train_file_path)
        self.data = self.read_json(self.test_file_path)


    def _determine_path(self):
        data_src_root = Path(f'{self.data_root}/jsons_{self.src}/')
        train_fp, test_fp = data_src_root / f'{self.train_split_name}.json',data_src_root / f'{self.test_split_name}.json'
        self.train_file_path, self.test_file_path = train_fp, test_fp

    def read_json(self, filepath: os.PathLike = None):
        with open(filepath, 'r') as f:
            jsonfile = json.load(f)
        return jsonfile

    def __getitem__(self, idx):
        raise NotImplementedError


    def __len__(self):
        return len(self.data)