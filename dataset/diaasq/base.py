
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
# from argparse import ArgumentParser

class BaseDiaAsq(Dataset):
    def __init__(self, src, split, data_root):
        # determine path
        self.src = src # en or zh
        self.split = split # eg. data/diaasq/dataset/jsons_en/train.json
        self.data_root = data_root
        self.filepath = self._determine_path()
        # determine data sampler
        self.data = self.read_json()

    def _determine_path(self):
        if self.split not in ['train', 'valid', 'test']:
            raise ValueError('split must be train, valid or test')
        data_src_root = Path(f'{self.data_root}/jsons_{self.src}/')
        return data_src_root / f'{self.split}.json'

    def read_json(self):
        with open(self.filepath, 'r') as f:
            jsonfile = json.load(f)
        return jsonfile

    def __getitem__(self, idx):
        raise NotImplementedError


    def __len__(self):
        return len(self.data)