from typing import List, Dict
from copy import deepcopy
import pickle
import json




def load_pkl(path: str):

    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_json(path: str) -> List[Dict]:

    with open(path, 'r') as f:
        data = json.load(f)
    return data

def remove_space(example: List[Dict]):
    _example = deepcopy(example)
    sentences = _example['sentences']
    for i, sent in enumerate(sentences):
        sentences[i] = _remove_space(sent)
    return _example

def _remove_space(text):
    # such that the texts look more natural
    text = text.split()
    text = ''.join(text)
    return text

def char_to_number(char: str):
    assert char < 'Z', 'char must be in [A, Z]'
    return ord(char) - ord('A')

def number_to_char(number: int):
    assert number < 26, 'number must be in [0, 25] such that it can be converted to a char'
    return chr(number + ord('A'))