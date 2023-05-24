#%%
from typing import Dict, List, Optional, Tuple, Union
from transformers import EvalPrediction
import numpy as np

from utils import (str_2_tuples,
                    calc_single_f1,
                    calc_pair_f1,
                    calc_strict_f1,)


class calc_sentiment_scores:
    """
    Calculate strict sentiment f1 and all the other metrics all at once for
    huggingface seq2seqTrainer.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, p: EvalPrediction) -> Dict:
        # https://zhuanlan.zhihu.com/p/363670628
        # the exact format of EvalPrediction is checked by pickling it out
        # this code works for `allenai/tk-instruct-base-def-pos`
        preds = p.predictions[0] # (batch_size, max_output_len, vocab_size)
        preds = np.argmax(preds, axis = -1)

        labels = p.label_ids
        # strings
        decoded_preds = self.tokenizer.batch_decode(
            preds, skip_special_tokens = True,
            clean_up_tokenization_spaces = True,
        )
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens = True,
            clean_up_tokenization_spaces = True,
        )

        # turn back to quadruple format

        # calculate target f1, aspect f1, opinion f1, sentiment f1, overall f1,
        # any 2 f1: Pair_t-a, Pair_t-o, Pair_a-o
        # quad micro f1
        target_f1 = calc_single_f1(decoded_preds, decoded_labels, idx=0)
        aspect_f1 = calc_single_f1(decoded_preds, decoded_labels, idx=1)
        opinion_f1 = calc_single_f1(decoded_preds, decoded_labels, idx=2)

        # pair f1
        pair_ta_f1 = calc_pair_f1(decoded_preds, decoded_labels, idx=(0,1))
        pair_to_f1 = calc_pair_f1(decoded_preds, decoded_labels, idx=(0,2))
        pair_ao_f1 = calc_pair_f1(decoded_preds, decoded_labels, idx=(1,2))

        # quad micro f1
        quad_micro_f1 = calc_strict_f1(decoded_preds, decoded_labels)
        return {
            'target_f1': target_f1,
            'aspect_f1': aspect_f1,
            'opinion_f1': opinion_f1,
            'pair_ta_f1': pair_ta_f1,
            'pair_to_f1': pair_to_f1,
            'pair_ao_f1': pair_ao_f1,
            'quad_micro_f1': quad_micro_f1,
        }


















