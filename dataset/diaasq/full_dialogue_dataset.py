import os
from typing import List, Tuple
import numpy as np
import logging
from torch.utils.data import Dataset
from dataset.utils import *
from dataset.diaasq.base import BaseDiaAsqDataset
from dataset.constants import diaasq_instruct_terms as instruct_terms
# need yaml config to determine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_speaker_prefix(data: List[Dict]):
    # check number of speakers
    speakers = data['speakers']
    n_speaker = len(set(speakers))
    sentences = data['sentences'].copy()
    sp_sent_pairs = zip(speakers, sentences)
    speaker_list = set()
    for i, (sp, sent) in enumerate(sp_sent_pairs):
        sp = number_to_char(sp)
        speaker_list.add(sp)
        sent = f'{sp}: {sent}'
        sentences[i] = sent
    data_copy = data.copy()
    del data
    data_copy['sentences'] = sentences
    return data_copy, speaker_list


class FullDiaAsqDataset(BaseDiaAsqDataset):

    def __init__(self,
                 src: str,
                 data_root: os.PathLike,
                 train_split_name: str,
                 test_split_name: str,
                 k: int,
                 seed: int,
                 prompt_path=os.PathLike):
        super().__init__(src, data_root, train_split_name, test_split_name)
        self.k = k
        self.seed = seed
        self.data = self.read_json(self.test_filepath)
        self.train_data = self.read_json(self.train_file_path)
        self.prompt_prefix = self._load_instruction(prompt_path)
        self.instruct_terms = instruct_terms[self.src]

        np.random.seed(self.seed)

    def __getitem__(self, idx) -> Tuple[any]:
        # test sample
        sample = self.data[idx]
        defs, examples, test_sample, test_label = self._make_prompt(sample, self.k)

        # return string, full_sample (original data + answer)
        # print('in-context')
        # print(examples)
        return {
            'input': defs + '\n' + examples + '\n' + test_sample,
            'label': test_label,
            'full_sample': sample
        }

    def _load_instruction(self, path: os.PathLike):
        """
        load instruction
        """
        with open(path, 'r') as f:
            prompt_prefix = f.read()
        return prompt_prefix

    def _make_prompt(self, test_data: Dict[str, any], k: int) -> Tuple[str, str, str]:
        """The prompt roughly follows Yizhong Wang et al. (2022)'s paper
        Super-NaturalInstructions:
        Generalization via Declarative Instructions on 1600+ NLP Tasks.
        - Definitions
        - Examples (in paper, 1 positive, 1 negative)
        - Evaluation Instance
        """
        definitions = self.prompt_prefix
        k_shots = self._k_shot(k)
        np.random.shuffle(k_shots)  # order randomize
        ordered_k_shots = [
            self._form_example(self.train_data[id], with_ans=True)[0] for id in k_shots
        ]
        ordered_k_shots_string = '\n'.join(ordered_k_shots)
        test_sample, test_label = self._form_example(test_data, with_ans=False)

        return definitions, ordered_k_shots_string, test_sample, test_label

    def _find_legal_pool(self, ) -> List[int]:
        # first check if the attribute exists in this class
        if hasattr(self, '_legal_pool'):
            return self._legal_pool
        legal_pool = []
        for id, d in enumerate(self.train_data):
            senti_set = set()
            triplets = d['triplets']
            for triplet in triplets:
                pol = triplet[6]
                senti_set.add(pol)
            if len(senti_set) == 3:
                legal_pool.append(id)
        self._legal_pool = legal_pool
        # count legal pool
        # logger.info(f'Legal pool size: {len(legal_pool)}')
        return legal_pool

    def _form_example(self, data: Dict[str, any], with_ans: bool = True):
        """
        formulate 1 example.
        Parameters
        ----------
        data: Dict[str]
            data from json file
        with_ans: bool
            whether to include answer triplets or not.
            if True, return example with answer triplets, normally used for in-context examples.
            else, return example without answer triplets, normally used for evaluation/prompting for answer.
        """
        _data = data.copy()
        if self.src == 'zh':
            _data = remove_space(data)  # only for zh

        _data, _ = add_speaker_prefix(_data)
        preproc_sentences = _data['sentences']
        quads = _data['triplets']
        quad_string = ''
        for quad in quads:
            pol = quad[6]
            target_string = quad[7]
            aspect_string = quad[8]
            opn_string = quad[9]
            # fill with null if empty
            if target_string == '': target_string = 'null'
            if aspect_string == '': aspect_string = 'null'
            if opn_string == '': opn_string = 'null'

            formatter = '(%s,%s,%s,%s)'
            quad_string += formatter % (target_string, aspect_string, opn_string, pol) + '\n'

        sent_string = '\n'.join(preproc_sentences)
        interm = self.instruct_terms['input']
        outterm = self.instruct_terms['output']
        if with_ans:
            return interm + sent_string + '\n\n' + outterm + quad_string, quad_string
        # prompting for answer
        return interm + sent_string + '\n\n' + outterm, quad_string

    def _k_shot(self, k: int) -> List[int]:
        # sample k unique samples from legal pool
        legal_pool = self._find_legal_pool()
        assert len(legal_pool) >= k, f'Legal pool size: {len(legal_pool)} < k: {k}'
        k_shot = np.random.choice(legal_pool, k, replace=False)  # replace ~= repetition

        return k_shot


# data_root = './data/diaasq/dataset'
# train_split_name = 'train'
# test_split_name = 'valid'
# k = 3
# prompt_path = f'./prompt/experiment/diaasq-fulldiag'
# dataset = FullDiaAsqDataset('en', data_root, train_split_name,
#                             test_split_name, k, 0, prompt_path)
# input_string, full_sample = dataset[0]
# print(full_sample)
# print(input_string)
