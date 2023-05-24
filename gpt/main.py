from typing import Dict
import os
import json
import yaml
import sys
import time
import requests
import openai
import dotenv
import logging
from datetime import datetime
from argparse import ArgumentParser
# append project root to sys.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataset.diaasq.speaker_spec_dataset import SpeakerDiaAsqDataset
from configs.utils import load_config
from utils import write_json





def __askGPT(message: str , model_name: str, api_key: str,
        temperature: float, max_tokens: int):
    davinci_url = "https://api.openai.com/v1/completions"
    chatgpt_url="https://api.openai.com/v1/chat/completions"
    if model_name == 'text-davinci-003':
        response = requests.post(url= davinci_url,
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
        response = requests.post(url= chatgpt_url,
                                 headers={
                                     'Content-Type': 'application/json',
                                     'Authorization': f'Bearer {api_key}'
                                 },
                                 json={
                                     'model': model_name,
                                     'messages': [{
                                         "role": "user",
                                         "content": message
                                     }], #
                                     'temperature': temperature,
                                     'max_tokens': max_tokens
                                 })
    return response

def load_testset(cfg: Dict):
    testset = SpeakerDiaAsqDataset(cfg.data.lang_src,
                               cfg.proc_data.data_root,
                               cfg.proc_data.test_ic_name,
                               cfg.proc_data.t5_test_split_name,
                               cfg.dataset.k,
                               cfg.seed,
                               cfg.dataset.prompt_path,
                               in_context_strategy=cfg.dataset.in_context_strategy)
    return testset

def get_logger(logpath: str = None):

    # if exists
    if os.path.exists(logpath):
        os.remove(logpath)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logpath, mode='a'),  # need to create the file first
            logging.StreamHandler()
        ])
    logger = logging.getLogger(__name__)
    return logger



def main(args):

    replies = []
    sent_messages = []
    cfg = load_config(args.cfg)
    dotenv.load_dotenv(cfg.envfile)
    APIKEY = os.getenv("OPENAI_API_KEY")

    testset = load_testset(cfg)

    save_path = os.path.join("results", cfg.dataset.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logpath = os.path.join(save_path, f"{cfg.model.model_name}_{timestamp}.log")
    logger = get_logger(logpath)
    logger.info(f'Saving replies to {save_path}')
    # logging info
    logger.info(f'[Model] {cfg.model.model_name}')
    logger.info(f'[Max Tokens] {cfg.model.max_tokens}')
    logger.info(f'[Url] {cfg.model.url}')
    logger.info(f'[Temperature] {cfg.model.temperature}')



    current_batch = []
    idx = args.start_index
    while idx <= len(testset):
        sample = testset[idx]
        doc_id = sample['doc_id']
        inputs = sample['input']
        labels = sample['label']

        response = __askGPT(inputs, cfg.model.model_name, api_key=APIKEY,
                temperature=cfg.model.temperature, max_tokens=cfg.model.max_tokens)
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
            current_batch.append()
            idx += 1

            if idx % args.log_step == 0 or idx == len(testset):
                logger.info(f"Processed the {idx}/{len(testset)} posts.")

            if idx % args.save_step == 0 or idx == len(testset):
                write_json(current_batch, os.path.join(save_path,f"replies{idx}.json"))
                current_batch = []




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/diaasq-gpt-speaker-spec-en.yaml')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--save_step", type=int, default=500)
    args = parser.parse_args()
    main(args)

