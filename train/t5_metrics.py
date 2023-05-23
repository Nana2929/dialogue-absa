#%%
from typing import Dict, List, Optional, Tuple, Union
import re
from transformers import EvalPrediction
import numpy as np


#%%
def str_2_tuples(string: str):
    pattern = r'\((.*?)\)'
    tuples = re.findall(pattern, string)
    tuples = [tuple(t.split(',')) for t in tuples]
    for i, t in enumerate(tuples):
        if len(t) < 4:
            # manually add empty string until 4 or
            # trim the tuple to 4
            t += ('',) * (4 - len(t))
        elif len(t) > 4:
            t = t[:4]
        tuples[i] = t
    return tuples


# string = "(1iej1,dfwhf,sss)"
# print(str_2_tuples(string))
#%%


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
        preds = p.predictions[0] # (batch_size, max_output_len, vocab_size)
        preds = np.argmax(preds, axis = -1)
        # print('preds:', preds.shape)

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
        target_f1 = self.calc_single_f1(decoded_preds, decoded_labels, idx=0)
        aspect_f1 = self.calc_single_f1(decoded_preds, decoded_labels, idx=1)
        opinion_f1 = self.calc_single_f1(decoded_preds, decoded_labels, idx=2)
        polaroity_f1 = self.calc_single_f1(decoded_preds, decoded_labels, idx=3)

        # pair f1
        pair_ta_f1 = self.calc_pair_f1(decoded_preds, decoded_labels, idx=(0,1))
        pair_to_f1 = self.calc_pair_f1(decoded_preds, decoded_labels, idx=(0,2))
        pair_ao_f1 = self.calc_pair_f1(decoded_preds, decoded_labels, idx=(1,2))

        # quad micro f1
        quad_micro_f1 = self.calc_strict_f1(decoded_preds, decoded_labels)
        return {
            'target_f1': target_f1,
            'aspect_f1': aspect_f1,
            'opinion_f1': opinion_f1,
            'polar_f1': polaroity_f1,
            'pair_ta_f1': pair_ta_f1,
            'pair_to_f1': pair_to_f1,
            'pair_ao_f1': pair_ao_f1,
            'quad_micro_f1': quad_micro_f1,
        }


    def calc_single_f1(self, predicted:List[str], gold: List[str], idx: int):
        """
        idx: from which position's element to extract
        """
        # following https://github.com/unikcc/DiaASQ/blob/master/src/run_eval.py
        fp, fn, tp = 0, 0, 0
        tol = 1e-10
        for pred_line, gold_line in zip(predicted, gold):
            pred_quads = str_2_tuples(pred_line)
            gold_quads = str_2_tuples(gold_line)
            pred_elems = [quad[idx] for quad in pred_quads]
            gold_elems = [quad[idx] for quad in gold_quads]

            fp += len(set(pred_elems) - set(gold_elems))
            fn += len(set(gold_elems) - set(pred_elems))
            tp += len(set(pred_elems) & set(gold_elems))
        p = tp / (tp + fp + tol)
        r = tp / (tp + fn + tol)
        f1 = 2*p*r/(p+r + tol)
        return f1

    def calc_pair_f1(self, predicted:List[str], gold: List[str], idx: Tuple[int, int]):
        """
        idx: from which 2 position's element to extract
        """
        fp, fn, tp = 0, 0, 0
        tol = 1e-10
        assert len(idx) == 2

        for pred_line, gold_line in zip(predicted, gold):
            pred_quads = str_2_tuples(pred_line)
            gold_quads = str_2_tuples(gold_line)
            pred_pairs = [(quad[idx[0]], quad[idx[1]]) for quad in pred_quads]
            gold_pairs = [(quad[idx[0]], quad[idx[1]]) for quad in gold_quads]

            fp += len(set(pred_pairs) - set(gold_pairs))
            fn += len(set(gold_pairs) - set(pred_pairs))
            tp += len(set(pred_pairs) & set(gold_pairs))
        p = tp / (tp + fp + tol)
        r = tp / (tp + fn + tol)
        f1 = 2*p*r/(p+r + tol)
        return f1

    def calc_strict_f1(self, predicted:List[str], gold: List[str]):
        fp, fn, tp = 0, 0, 0
        tol = 1e-10
        for pred_line, gold_line in zip(predicted, gold):
            pred_quads = str_2_tuples(pred_line)
            gold_quads = str_2_tuples(gold_line)

            fp += len(set(pred_quads) - set(gold_quads))
            fn += len(set(gold_quads) - set(pred_quads))
            tp += len(set(pred_quads) & set(gold_quads))
        p = tp / (tp + fp + tol)
        r = tp / (tp + fn + tol)
        f1 = 2*p*r/(p+r + tol)
        return f1

















