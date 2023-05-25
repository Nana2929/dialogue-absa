from typing import Dict, List, Optional, Tuple, Union
import sys
import os
from pprint import pprint
from argparse import ArgumentParser
from pathlib import Path
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from metrics.utils import (calc_single_f1,
                           calc_pair_f1,
                           calc_strict_f1)
from configs.utils import load_config
from dataset.utils import load_json, write_out
from dataset.diaasq.speaker_spec_dataset import SpeakerDiaAsqDataset
from dataset.diaasq.full_dialogue_dataset import FullDiaAsqDataset

# load the replies and concatenate
# data/diaasq/speaker_dataset/proc/jsons_en/t5_valid.json

type_switch = {
        'speaker': SpeakerDiaAsqDataset,
        'full': FullDiaAsqDataset,
    }
def determine_result_dir(cfg, dataset_name: str) -> os.PathLike:
    dataset_lang_name = f'{dataset_name}_{cfg.data.lang_src}'
    result_dir = Path(cfg.output_dir) / dataset_lang_name
    return result_dir

def concat_files(res_dir: os.PathLike) -> List[Dict]:
    res_dir = Path(res_dir)
    files = res_dir.glob('*.json')
    # sort the files by ending suffix
    # `replies_2.json`
    concat = []

    files = sorted(files, key=lambda x: int(str(x).split('.')[0].split('_')[-1]))
    print(files)
    logger.info(f'Found {len(files)} files to be concatenated ...')
    for file in files:
        concat.extend(load_json(file))
    return concat


def extract_msg(gpt_reply: Dict):
    return gpt_reply['choices'][0]['message']['content']

def load_testset(cfg: Dict):
    Dataset = type_switch[cfg.proc_data.type]
    testset = Dataset(src = cfg.data.lang_src,
                    data_root = cfg.proc_data.data_root,
                    ic_split_name = cfg.proc_data.test_ic_name,
                    data_split_name = cfg.proc_data.t5_test_split_name,
                    k = cfg.dataset.k,
                    seed = cfg.seed,
                    prompt_path = cfg.dataset.prompt_path,
                    in_context_strategy=cfg.dataset.in_context_strategy)
    return testset



def _test_sanity(gpt_replies: List[Dict], testset: List[Dict]):
    """
    這裡當初沒有寫很好，只有留下 doc_id，但 doc_id 在 speaker data 中不是 primary key
    所以只能比較照順序來看 doc_id 是不是一樣的（incomplete sanity check）
    """
    # assert len(gpt_replies) == len(testset)
    for gpt_reply, test in zip(gpt_replies, testset):
        assert gpt_reply['doc_id'] == test['doc_id'], \
        f'{gpt_reply["doc_id"]} != {test["doc_id"]}, 2 lists have different order!'
    logger.info('Sanity check passed!')

def main(args):
    cfg = load_config(args.cfg)
    # cfg = load_config('configs/diaasq-gpt-speaker-spec-en.yaml')

    testset = load_testset(cfg)
    dataset_name = testset.__class__.__name__

    result_dir = determine_result_dir(cfg, dataset_name = dataset_name)
    logger.info(f'Result Dir: {result_dir}')
    gpt_replies = concat_files(result_dir)
    _test_sanity(gpt_replies, testset)


    # make output file
    # test_file_path = Path(cfg.proc_data.data_root) / f'jsons_{cfg.data.lang_src}' / f'{cfg.proc_data.t5_test_split_name}.json'
    # test_file = load_json(test_file_path)
    outdir = Path(cfg.output_dir)
    outfile = outdir / f'{dataset_name}_gpt_eval.{args.suffix}'
    if args.outfile:
        outfile = args.outfile+ '.' + args.suffix
    # doc_id, speaker, sentences, labels, replies
    outs = []
    gpt_strings = []
    for reply in gpt_replies:
        gpt_strings.append(extract_msg(reply))

    for gpt_string, test_example in zip(gpt_strings, testset): # input, label, doc_id
        record = {
            'input_sequence': test_example['input'],
            'doc_id' : test_example['doc_id'],
            'speaker': test_example['speaker'],
            'sentences': test_example['his_sentences'],
            'labels': test_example['label'],
            'replies': gpt_string
        }


        outs.append(record)
    logger.info(f'Writing inference file to {outfile} ...')
    write_out(data = outs, path = outfile)

    gold_strings = [test['label'] for test in testset]
    logger.info(f'Starting evaluation on {cfg.proc_data.t5_test_split_name} ({len(testset)} examples)...')
    # calculate f1 scores
    target_f1 = calc_single_f1(gpt_strings, gold_strings, idx = 0)
    aspect_f1 = calc_single_f1(gpt_strings, gold_strings, idx = 1)
    opinion_f1 = calc_single_f1(gpt_strings, gold_strings, idx = 2)
    pair_ta_f1 = calc_pair_f1(gpt_strings, gold_strings, idx = (0,1))
    pair_to_f1 = calc_pair_f1(gpt_strings, gold_strings, idx = (0,2))
    pair_ao_f1 = calc_pair_f1(gpt_strings, gold_strings, idx = (1,2))
    quad_micro_f1 = calc_strict_f1(gpt_strings, gold_strings)

    # make
    eval_results = {
            'target_f1': target_f1,
            'aspect_f1': aspect_f1,
            'opinion_f1': opinion_f1,
            'pair_ta_f1': pair_ta_f1,
            'pair_to_f1': pair_to_f1,
            'pair_ao_f1': pair_ao_f1,
            'quad_micro_f1': quad_micro_f1,
        }
    pprint(eval_results)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/diaasq-gpt-speaker-spec-en.yaml')
    parser.add_argument('--outfile', type=str, required=False, help = 'full path; inference file stem')
    parser.add_argument('--suffix', type=str, default = 'csv', required=False, help = 'suffix of the inference file (No dot!), default `csv`. Also supporting `json`')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    main(args)



