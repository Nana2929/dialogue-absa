from typing import List, Dict
from copy import deepcopy



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

def add_speaker_prefix(data: List[Dict]):
    # check number of speakers
    speakers = data['speakers']
    n_speaker = len(set(speakers))
    assert n_speaker <= 26 # a better way is to convert 26 to AA, 27 to AB, etc.
    sentences = data['sentences'].copy()
    sp_sent_pairs = zip(speakers, sentences)
    speaker_list = set()
    for i, (sp, sent) in enumerate(sp_sent_pairs):
        sp = chr(ord('@')+int(sp)+1)
        speaker_list.add(sp)
        sent = f'{sp}: {sent}'
        sentences[i] = sent
    data_copy = data.copy()
    del data
    data_copy['sentences'] = sentences
    return data_copy, speaker_list
