# %%
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from utils import (
    str_2_tuples,
    calc_single_f1,
    calc_pair_f1,
    calc_strict_f1,
)


class calc_sentiment_scores:
    """
    Calculate strict sentiment f1 and all the other metrics
    for llama generated text output
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, preds: list[str], golds: list[tuple]) -> Dict:
        # https://zhuanlan.zhihu.com/p/363670628
        # the exact format of EvalPrediction is checked by pickling it out
        # this code works for `allenai/tk-instruct-base-def-pos`

        # calculate target f1, aspect f1, opinion f1, sentiment f1, overall f1,
        # any 2 f1: Pair_t-a, Pair_t-o, Pair_a-o
        # quad micro f1
        target_f1 = calc_single_f1(preds, golds, idx=0)
        aspect_f1 = calc_single_f1(preds, golds, idx=1)
        opinion_f1 = calc_single_f1(preds, golds, idx=2)

        # pair f1
        pair_ta_f1 = calc_pair_f1(preds, golds, idx=(0, 1))
        pair_to_f1 = calc_pair_f1(preds, golds, idx=(0, 2))
        pair_ao_f1 = calc_pair_f1(preds, golds, idx=(1, 2))

        # quad micro f1
        quad_micro_f1 = calc_strict_f1(preds, golds)
        return {
            "target_f1": target_f1,
            "aspect_f1": aspect_f1,
            "opinion_f1": opinion_f1,
            "pair_ta_f1": pair_ta_f1,
            "pair_to_f1": pair_to_f1,
            "pair_ao_f1": pair_ao_f1,
            "quad_micro_f1": quad_micro_f1,
        }
