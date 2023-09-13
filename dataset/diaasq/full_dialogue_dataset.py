import os
from collections import defaultdict
import numpy as np
import logging

import sys


from dataset.utils import *
from dataset.constants import diaasq_instruct_terms as instruct_terms

# need yaml config to determine
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_speaker_prefix(data: list[dict]) -> tuple[list[dict], set[str]]:
    # check number of speakers
    speakers = data["speakers"]
    sentences = data["sentences"].copy()
    sp_sent_pairs = zip(speakers, sentences)
    speaker_list = set()
    for i, (sp, sent) in enumerate(sp_sent_pairs):
        sp = number_to_char(sp)
        speaker_list.add(sp)
        sent = f"{sp}: {sent}"
        sentences[i] = sent
    data_copy = data.copy()
    del data
    data_copy["sentences"] = sentences
    return data_copy, speaker_list


pol_map = {
    "pos": "positive",
    "neg": "negative",
    "other": "neutral",
    "neu": "neutral",
}


# ==========================
class FullDiaAsqDataset:
    def __init__(
        self,
        src,
        data_path: os.PathLike,
        instruct_path=os.PathLike,
    ):
        self.src = src
        self.data_path = data_path
        self.data = load_json(self.data_path)
        self.instruct_prefix = self._load_instruction(instruct_path)
        self.instruct_terms = instruct_terms[self.src]

    def __getitem__(self, idx) -> tuple[any]:
        # test sample
        sample = self.data[idx]
        input_string, output_string, success_count = self._form_output(sample)
        char_speakers = [number_to_char(sp) for sp in sample["speakers"]]
        doc_id = sample["doc_id"]
        extra_metadata = {
            "success_quad_count": success_count,
            "quad_count": len(sample["triplets"]),
        }

        return {
            "instruction": self.instruct_prefix,
            "input": input_string,
            "output": output_string,
            "doc_id": doc_id,
            "char_speakers": char_speakers,
            "metadata": get_metadata(sample) | extra_metadata,
        }

    def _load_instruction(self, path: os.PathLike) -> str:
        """
        LLaMA has <SystemPrompt>, <Instruction>, <Input>, <Output> (for demo) parts as its total input.
        load instruction, to be placed at `instruction`
        """
        assert os.path.exists(path)
        with open(path, "r") as f:
            instruct_prefix = f.read()
        logger.info(f"Loaded instruction from {path}.")
        return instruct_prefix

    def __get_sentence_boundaries(self, raw_sentences) -> list[tuple[int, int]]:
        """
        Parameters
        ----------
        raw_sentences : _type_
            The DiaASQ dataset's original key 'sentences'
        Returns
        -------
        list(tuple(int, int))
            A list of start and end indices of each sentence when sentences are
            concatenated as " ".join(raw_sentences)
        """
        sentence_indices = []
        start_index = 0
        for sent in raw_sentences:
            end_index = start_index + len(sent.split())
            sentence_indices.append((start_index, end_index))
            start_index = end_index
        return sentence_indices

    def __get_opn_holder(
        self,
        opn_start_index: int,
        opn_end_index: int,
        sentence_indices: list[str],
        speakers: list[int],
    ) -> str:
        """

        Parameters
        ----------
        opn_start_index : int
        opn_end_index : int
        sentence_indices : list[str]
        speakers : list[int]

        Returns
        -------
        str
            The speaker of the sentence that contains the opinion spans in char form, e.g. 'A', 'B', 'C', etc.

        Raises
        ------
        ValueError
            If the sentence that contains the opinion spans cannot be found.
        """
        opn_sent_idx = None
        for sent_id, (s, t) in enumerate(sentence_indices):
            if opn_start_index >= s and opn_end_index < t:
                opn_sent_idx = sent_id
                break
        if opn_sent_idx is None:
            raise ValueError(
                f"Cannot find the sentence (therefore, holder) that contains the opinion spans."
            )
        opn_holder = speakers[opn_sent_idx]
        # holder as a char
        return number_to_char(opn_holder)

    def _form_output(self, data: Dict[str, any]) -> tuple[str, str, int]:
        from collections import namedtuple

        SentTuple = namedtuple(
            "SentTuple", ["opn_holder", "target", "aspect", "opn", "pol", "opn_start"]
        )
        """
        formulate 1 example.
        Parameters
        ----------
        data: Dict[str]
            data from json file
        Returns
        -------
        tuple[str, str, int]
            input string, output string, number of quadruples successfully converted

        """
        original_sentences = data["sentences"]
        speakers = data["speakers"]
        #  =========================
        _data = data.copy()
        if self.src == "zh":
            _data = remove_space(data)  # only for zh
        _data, _ = add_speaker_prefix(_data)
        preproc_sentences = _data["sentences"]
        # !!! Concatenate sentences with space
        sent_string = " ".join(preproc_sentences)
        quads = _data["triplets"]
        # ===========================
        sentence_indices = self.__get_sentence_boundaries(original_sentences)

        quad_strings = []
        for quad in quads:
            pol = quad[6]
            target_string = quad[7]
            aspect_string = quad[8]
            opn_string = quad[9]
            opn_start_index, opn_end_index = quad[4], quad[5]
            try:
                opn_holder = self.__get_opn_holder(
                    opn_start_index, opn_end_index, sentence_indices, speakers
                )
            except ValueError as e:
                # drop this quad
                logger.warning("Dropping quadruple due to error: %s" % e)
                continue
            # fill with null if empty
            if target_string.strip() == "":
                target_string = "null"
            if aspect_string.strip() == "":
                aspect_string = "null"
            if opn_string.strip() == "":
                opn_string = "null"

            sent_tuple = SentTuple(
                opn_holder,
                target_string,
                aspect_string,
                opn_string,
                pol,
                opn_start_index,
            )
            quad_strings.append(sent_tuple)
        formatter = "%s: (%s, %s, %s, %s)"
        # !!! Sort by opn_start_index (the appearance order in the dialogue)
        quad_strings = sorted(quad_strings, key=lambda x: x.opn_start)
        quad_string = " ".join(
            [
                formatter
                % (
                    quad.opn_holder,
                    quad.target,
                    quad.aspect,
                    quad.opn,
                    pol_map[quad.pol],
                )
                for quad in quad_strings
            ]
        )

        return sent_string, quad_string, len(quad_strings)
