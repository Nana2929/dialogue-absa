import os
from typing import List, Dict, Tuple
from dataset.diaasq.base import BaseDiaAsqDataset
from dataset.constants import diaasq_instruct_terms as instruct_terms
import logging
from dataset.utils import number_to_char, remove_space
import numpy as np


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

class SpeakerDiaAsqDataset(BaseDiaAsqDataset):
    def __init__(self, src: str,
                    data_root: os.PathLike,
                    ic_split_name: str, # in-context examples
                    data_split_name: str,
                    k: int,
                    seed: int,
                    prompt_path: os.PathLike,
                    in_context_strategy: str):
        super().__init__(src, data_root, ic_split_name, data_split_name)
        self.k = k
        self.in_context_strategy = in_context_strategy # 'same_speaker' or 'same_dialogue'
        self.seed = seed
        # base class may need refactor
        self.data = self.read_json(self.test_file_path)
        self.in_context_data = self.read_json(self.train_file_path)

        self.prompt_prefix = self._load_instruction(prompt_path)
        self.instruct_terms = instruct_terms[self.src]

        np.random.seed(self.seed)

    def __getitem__(self, idx: int) -> Tuple[any]:
        sample = self.data[idx]
        sample_speaker = sample['speaker']
        defs, in_context_examples, test_sample, test_label = self._make_prompt(sample, self.k)
        return {
            'input': defs + '/n' + in_context_examples + '/n' + test_sample,
            'label': test_label,
            'speaker': sample_speaker,
            'doc_id': sample['doc_id']}

    def _load_instruction(self, path: os.PathLike) -> str:

        with open(path, 'r') as f:
            prompt = f.read()
        return prompt

    def _make_prompt(self, sample: Dict[str, any], k: int) -> Tuple[str, str, str, str]:
        formulated_test_sample, test_label = self._form_example(sample, with_ans=False)
        k_shots = self._k_shot(k)
        k_shots_string = '\n'.join(k_shots)

        return  self.prompt_prefix, k_shots_string, formulated_test_sample, test_label

    def _k_shot(self, k: int) -> List[str]:
        assert k <= len(self.in_context_data)
        # reorder
        k_shots = np.random.choice(self.in_context_data, k, replace=False)
        formulated_k_shots = [
            self._form_example(data, with_ans=True)[0] for data in k_shots]
        return formulated_k_shots


    def _form_example(self, data: Dict[str, any],
                      with_ans: bool=True):
        # keys: speaker, doc_id, his_opn_quads, his_sentence_ids, full_dialog
        his_quads = data['his_opn_quads']


        # formatting sentences
        # if en, no need to remove space
        if self.src == 'zh':
            data = remove_space(data)
        preproc_data, _ = add_speaker_prefix(data)
        his_sentences = preproc_data['his_sentences']
        sent_string = '\n'.join(his_sentences)

        # formatting quads
        quad_string = ''
        for quad in his_quads:
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

        intterm = self.instruct_terms['input']
        outterm = self.instruct_terms['output']
        if with_ans:
            return intterm + sent_string + '\n' + outterm + quad_string, quad_string
        return intterm + sent_string + '\n' + outterm, quad_string
