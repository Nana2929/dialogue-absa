"""
@File    :   speaker_spec_dataset.py
@Time    :   2023/05/19 23:02:54
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw
@Desc    :   Speaker-angle dataset. Inference by identifying 1 speaker's angle.
             sentiment quadruples. The task can be furthered expanded
             to stance-detection, which is to identify the speaker's stance
             towards certain target/aspect (A more complicated task).
"""
import os
from typing import List, Dict, Tuple
from collections import defaultdict
import logging

import numpy as np
import yaml
from easydict import EasyDict as edict


from dataset.diaasq.base import BaseDiaAsqDataset
from dataset.utils import remove_space, number_to_char, char_to_number, load_pkl
from dataset.constants import diaasq_instruct_terms as instruct_terms





with open("configs/diaasq.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = edict(cfg)


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

class SpeakerDataset(BaseDiaAsqDataset):
    def __init__(
        self,
        src: str,
        data_root: os.PathLike,
        align_with: str,
        train_split_name: str = "train",
        test_split_name: str = "valid",
        k: int = 5,
        seed: int = 0,
        prompt_path: os.PathLike = None,
    ):
        """_summary
        Parameters
        ----------
        align_with : str, optional
            'speaker', 'dialogue' or None
            'speaker' means the in-context examples choose the same speaker as the test example
            'dialogue' means the in-context examples choose the same dialogue's other speakers
                    as the test example
            None means the in-context examples are randomly chosen from the train set,
            currently we do not control the quadruples' number in the in-context examples, so there may be
            poor examples for all setting for align_with.

        """
        super().__init__(
            src,
            data_root,
            train_split_name,
            test_split_name,
        )
        self.k = k
        self.seed = seed
        self.data = self.read_json(self.test_filepath)
        self.train_data = self.read_json(self.train_file_path)
        self.prompt_prefix = self._load_instruction(prompt_path)
        self.instruct_terms = instruct_terms[self.src]

        self.train_docid_2_nspeaker = load_pkl(cfg.meta.train)

        # ============================== #
        self.align_with = align_with  # 'dialogue', None
        # ============================== #

    def _check_align_with(self):
        assert self.align_with in [
            "speaker",
            "dialogue",
            None,
        ], f"align_with: {self.align_with} not in [speaker, dialogue, None]"

    def _preprocess_example(self, example: Dict[str, any]):
        data = example.copy()
        del example
        cleaned_data = remove_space(data)
        return add_speaker_prefix(cleaned_data)

    def __getitem__(self, idx: int):
        test_example = self.data[idx]

        # 1. preprocess test example
        preproc_test_example = self._preprocess_example(test_example)
        # 2. choose a speaker from the speaker_list
        test_example_n_speaker = self.train_docid_2_nspeaker[test_example['docid']]
        speaker_id = np.random.choice(range(test_example_n_speaker))
        speaker = number_to_char(speaker_id)


        # 3. formulate the test example with the chosen speaker without answer
        formulated_test_example = self._form_example(preproc_test_example, with_ans=False, speaker=speaker)
        # 4. formulate in-context examples
        if self.align_with == "speaker":
            # no need to worry about sampling the same sample as test example
            # because we sample from train set but our test example comes from valid set
            in_context_examples = self._k_shot_same_speaker(
                self.k,
                speaker = speaker,
            )
        elif self.align_with == "dialogue":
            in_context_examples = self._k_shot_same_dialog(
                self.k,
                speaker = speaker,
                data = test_example
            )
        # count the in_context_examples
        n_in_context_examples = len(in_context_examples)
        final_string = self._make_prompt(
            formulated_test_example,
            in_context_examples,
        )
        return {
            'data' : final_string,
            'n_in_context_examples' : n_in_context_examples, # used to check validity
            # in strict experiment setting, examples whose n_in_context_examples != k may
            # be discarded
        }

    def _load_instruction(self, path: os.PathLike):
        """
        load instruction
        """
        assert os.path.exists(path)
        with open(path, "r") as f:
            prompt_prefix = f.read()
        return prompt_prefix


    def _make_prompt(self, test_data: str, in_context_data: List[str]):
        """
        make prompt for each test example
        i.e. concatenate the instruction, in-context examples (if any) and the test example
        """
        prompt = self.prompt_prefix% + '\n'
        for ic in in_context_data:
            prompt += ic + '\n\n'
        prompt += test_data
        return prompt


    def _k_shot_same_dialog(self, k:int, speaker: str, data: Dict[str, any]) -> List[Dict[str, any]]:
        """ use the same dialogue and pick k other speakers,
        then formulate k examples with these k speakers
        """
        # 1. find all speakers in the dialogue
        speaker_list = data['speakers']
        n_speaker = len(set(speaker_list))
        # 2. find the speaker's index
        spkr_index = char_to_number(speaker)

        final_k = min(k, n_speaker - 1)
        sentence_indices = self._find_sentence_boundary(data)
        speaker_opn_quads, speaker_asso_sent_indices = self._find_all_speaker_data(
            data, sentence_indices
        )
        # randomize k other speakers
        other_speakers = list(speaker_opn_quads.keys())
        other_speakers.remove(speaker)
        other_speakers = np.random.choice(other_speakers, final_k, replace=False)

        k_formulated_examples = [
            self._form_example(data, with_ans=True, speaker=spkr) for spkr in other_speakers
        ]
        return k_formulated_examples


    def _k_shot_same_speaker(self, k: int, speaker: str) -> List[Dict[str, any]]:
        """
        Parameters
        ----------
        speaker: str

        Returns
        -------
        k formulated examples
        """
        # legal_pool: List[Dict[str, any]]
        # count all data['speaker'] nunique
        # map k to number
        spkr_index = char_to_number(speaker)
        legal_pool = []
        for data in self.train_data:
            doc_id = data['doc_id']
            n_speaker = self.train_docid_2_nspeaker[doc_id]
            if n_speaker > spkr_index:
                legal_pool.append(data)
        # randomly choose k examples from legal_pool
        k_examples = np.random.choice(legal_pool, min(k, len(legal_pool)), replace=False)
        # formulate k examples
        k_formulated_examples = [
            self._form_example(
                example,
                with_ans=True,
                speaker=speaker,
                ) for example in k_examples
            ]
        return k_formulated_examples # List[Dict[str, any]]

    def _form_example(self, data: Dict[str, any], speaker: str,
                      with_ans: bool = True):
        """formulate an example string based on original
        data, speaker, and with_ans flag.

        Parameters
        ----------
        data : Dict[str, any]
            a data in its original form
        speaker : str
            which speaker's angle to reformulate the data
        with_ans : bool, optional
            whether to include answer in the formulated example, by default True
        """
        speaker_opn_quads, speaker_asso_sent_indices = self._find_all_speaker_data(
            data, self._find_sentence_boundary(data))
        spkr_asso_sents = speaker_asso_sent_indices[speaker]
        spkr_opn_quads = speaker_opn_quads[speaker]
        sents = '\n'.join(s for s in spkr_asso_sents)

        interm = self.instruct_terms['input']
        outterm = self.instruct_terms['output']

        if not with_ans:
            return interm+sents + '\n\n' + outterm
        opn_quads = '\n'.join(q for q in spkr_opn_quads)
        speaker_instruct = '/n'%()
        return interm + speaker_instruct + sents + '\n\n' + outterm + opn_quads


    def _find_sentence_boundary(self, data: Dict[str, any]):
        """find the start and end index of each sentence in the data,
        this is for the purpose of identifying the quadruple's belonging sentence.
        Parameters
        ----------
        data : Dict[str, any]
            the `sentences` key must contain original sentences.
            No whitespaces removed, speaker added, or any preprocessing operations.
        """
        sentences = data["sentences"]
        sentence_indices = []
        start_index = 0
        for sent in sentences:
            end_index = start_index + len(sent.split())
            sentence_indices.append((start_index, end_index))
            start_index = end_index
        assert len(sentences) == len(sentence_indices)
        return sentence_indices

    def _find_holder(self, opn_sent: str):
        """Find the holder of the opinion span
        Since 1 dialogue would not have more than 10 sentences, so naturally < 10 speakers
        all encoded speakers are A, B, C, D, E, F, G, H, I, J (1-char).
        Therefore the holder is the first char of the sentence.

        Parameters
        ----------
        opn_sent : str
            the sentence containing the opinion span
        Returns
        -------
        str:
            the speaker of the opinion span (character-based, upper-case)
        """
        return opn_sent[0]

    def _find_speaker_data(
        self,
        speaker: str,
        speaker_opn_quads: Dict[str, any],
        speaker_asso_sent_indices: Dict[str, any],
    ):
        opn_quads = speaker_opn_quads.get(speaker, None)
        asso_sent_indices = speaker_asso_sent_indices.get(speaker, None)
        return opn_quads, asso_sent_indices

    def _find_all_speaker_data(self, data: Dict[str, any], sentence_indices: List[int]):
        speaker_opn_quads = defaultdict(list)
        speaker_asso_sent_indices = defaultdict(set)
        triplets = data["triplets"]
        sentences = data["sentences"]
        for trip_id, trip in enumerate(triplets):
            assert len(trip) == 10
            target_string = trip[7]
            aspect_string = trip[8]
            opn_string = trip[9]
            pol = trip[6]
            tgt_s, tgt_t = trip[0], trip[1]
            asp_s, asp_t = trip[2], trip[3]
            opn_s, opn_t = trip[4], trip[5]
            # if any of them is (-1, -1), asp_sent_idx will be -1
            tgt_sent_idx = -1
            asp_sent_idx = -1
            opn_sent_idx = -1

            for sent_id, (s, t) in enumerate(sentence_indices):
                if tgt_s >= s and tgt_t < t:
                    tgt_sent_idx = sent_id
                if asp_s >= s and asp_t < t:
                    asp_sent_idx = sent_id
                if opn_s >= s and opn_t < t:
                    opn_sent_idx = sent_id
            # check if triplet is in the same sentence: No
            # check if they are spoken by the same speaker: No
            # index the sentences if not -1
            # 可以嘗試一個 stance-holding dataset: opn string belongs to which speaker
            # tgt_sent = sentences[tgt_sent_idx] if tgt_sent_idx != -1 else None
            # asp_sent = sentences[asp_sent_idx] if asp_sent_idx != -1 else None
            opn_sent = sentences[opn_sent_idx] if opn_sent_idx != -1 else None
            # print

            # check the opinion is explicit
            # deal with explicit opn only, because implicit opn is not spoken by anyone; cannot specify the speaker
            if opn_sent is None:
                continue
            opn_holder = self._find_holder(opn_sent)
            speaker_opn_quads[opn_holder].append(
                (target_string, aspect_string, opn_string, pol)
            )
            # beware of -1 (代表not found, 不是倒數第一個)
            speaker_asso_sent_indices[opn_holder].update(
                [tgt_sent_idx, asp_sent_idx, opn_sent_idx]
            )
        return speaker_opn_quads, speaker_asso_sent_indices


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # need to check 0, 1, 2, 3 shots
    data_root = './data/diaasq/dataset'
    train_split_name = 'train'
    test_split_name = 'valid'
    k = 0
    prompt_path = f'./prompt/experiment/diaasq-speaker'
