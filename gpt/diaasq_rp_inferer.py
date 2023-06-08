from typing import Dict
import os
import sys
import time
import requests
import openai
import json
import dotenv
import logging
from datetime import datetime
from argparse import ArgumentParser
# append project root to sys.path
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataset.diaasq.speaker_spec_dataset import SpeakerDiaAsqDataset
from dataset.diaasq.full_dialogue_dataset import FullDiaAsqDataset
from configs.utils import load_config
from utils import write_json


def __askGPT(message: str,
             role_play: str,
             model_name: str,
             api_key: str,
             temperature: float,
             max_tokens: int):
    davinci_url = "https://api.openai.com/v1/completions"
    chatgpt_url = "https://api.openai.com/v1/chat/completions"
    if model_name == 'text-davinci-003':
        response = requests.post(url=davinci_url,
                                 headers={
                                     'Content-Type': 'application/json',
                                     'Authorization': f'Bearer {api_key}'
                                 },
                                 json={
                                     'model': model_name,
                                     'prompt': message,
                                     'temperature': temperature,
                                     'max_tokens': max_tokens
                                 })
    else:
        response = requests.post(
            url=chatgpt_url,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            },
            json={
                'model': model_name,
                'messages': [
                    {
                        'role': 'system',
                        'content': role_play  #put your prompt prefix here, credit to @YingJia
                    },
                    {
                    "role": "user",
                    "content": message
                    }
                    ],  #
                'temperature': temperature,
                'max_tokens': max_tokens
            })
    return response


def load_testset(cfg: Dict):
    type_switch = {
        'speaker': SpeakerDiaAsqDataset,
        'full': FullDiaAsqDataset,
    }

    Dataset = type_switch[cfg.data.type]
    testset = Dataset(src=cfg.data.lang_src,
                      data_root=cfg.data.data_root,
                      ic_split_name=cfg.data.ic_split_name,
                      data_split_name=cfg.data.data_split_name,
                      k=cfg.dataset.k,
                      seed=cfg.seed,
                      prompt_path=cfg.dataset.prompt_path,
                      in_context_strategy=cfg.dataset.in_context_strategy)

    return testset


def get_logger(logpath: str = None):

    # if exists
    if os.path.exists(logpath):
        os.remove(logpath)
    logger = logging.getLogger(__name__)
    # set handlers
    logger.handlers = [logging.FileHandler(logpath, mode='a'), logging.StreamHandler()]
    logger.level = logging.INFO
    # set format
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for h in logger.handlers:
        h.setFormatter(formatter)
    return logger


def load_definition(prompt_path: str):
    with open(prompt_path, 'r') as f:
        definition = f.read()
    return definition


def main(args):

    cfg = load_config(args.cfg)
    dotenv.load_dotenv(cfg.envfile)
    APIKEY = os.getenv("OPENAI_API_KEY")

    testset = load_testset(cfg)

    dataset_name = testset.__class__.__name__
    # output/diaasq/gpt-full-dialog/FullDiaAsqDataset_zh_k=1
    save_path = Path(cfg.output_dir) / (dataset_name + f'_{cfg.data.lang_src}_k={cfg.dataset.k}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load definition prefix
    role_play = load_definition(cfg.dataset.prompt_path)


    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logpath = os.path.join(save_path, f"{cfg.model.model_name}_{timestamp}.log")
    logger = get_logger(logpath)
    filename =   os.path.basename(__file__)
    logger.info(f'=== {filename} ===')

    # logging info
    logger.info(f'[Model] {cfg.model.model_name}')
    logger.info(f'[Max Tokens] {cfg.model.max_tokens}')
    logger.info(f'[Temperature] {cfg.model.temperature}')
    logger.info(f"Using role_play: {role_play}")
    logger.info(f'Saving replies to {save_path}')
    logger.info(f"k-shot #: {cfg.dataset.k}")
    current_batch = []
    idx = args.start_index
    logger.info(f"Total testdata size: {len(testset)}, start from {idx}")


    while idx < len(testset):
        sample = testset[idx]
        doc_id = sample['doc_id']
        no_def_inputs = sample['no_def_input']


        response = __askGPT(message=no_def_inputs,
                            role_play= role_play,
                            model_name = cfg.model.model_name,
                            api_key=APIKEY,
                            temperature=cfg.model.temperature,
                            max_tokens=cfg.model.max_tokens)
        if response.status_code != 200:
            logger.info(f"{idx} Failed to get response for {idx}, retry in 1 sec.")
            time.sleep(0.5)
            # save the current batch and clean the batch
            if len(current_batch) > 0:
                write_json(current_batch, os.path.join(save_path, f"replies_{idx}.json"))
                current_batch = []
            continue
        else:
            # else success
            response = response.json()
            response['doc_id'] = doc_id
            current_batch.append(response)
            # print(response)
            idx += 1

            if idx % args.log_step == 0 or idx == len(testset):
                logger.info(f"Processed the {idx}/{len(testset)} posts.")

            if idx % args.save_step == 0 or idx == len(testset):
                write_json(current_batch, os.path.join(save_path, f"replies_{idx}.json"))
                current_batch = []


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--cfg', type=str, default='')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--save_step", type=int, default=200)
    args = parser.parse_args()
    main(args)
