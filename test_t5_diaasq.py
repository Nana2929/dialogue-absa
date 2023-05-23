#%%
import warnings
import yaml
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from easydict import EasyDict as edict
import torch
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

from dataset.diaasq.speaker_spec_dataset import SpeakerDiaAsqDataset
from dataset.utils import SpeakerDiaAsqCollator
from train.t5_metrics import strict_sentiment_f1
warnings.filterwarnings('ignore')

# loading config
load_config = lambda path: edict(yaml.load(open(path, 'r'), Loader=yaml.FullLoader))
config_path = 'configs/diaasq-t5-speaker-spec-en.yaml'
cfg = load_config(config_path)
in_context_strategy = None
device = 'cuda:0'
# loading model
ckpt_dir = cfg.model.output_dir
model_name = cfg.model.model_name
ckpt = Path(ckpt_dir) / 'checkpoint-1000'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)

# making test loader
collate_fn = SpeakerDiaAsqCollator(tokenizer=tokenizer, max_len=cfg.model.max_length)
testset = SpeakerDiaAsqDataset(cfg.data.lang_src,
                               cfg.proc_data.data_root,
                               cfg.proc_data.test_ic_name,
                               cfg.proc_data.t5_test_split_name,
                               cfg.dataset.k,
                               cfg.seed,
                               cfg.dataset.prompt_path,
                               in_context_strategy=in_context_strategy)
testloader = DataLoader(testset,
                        batch_size=cfg.trainer.per_device_eval_batch_size,
                        shuffle=False,
                        num_workers=4,
                        collate_fn=collate_fn)
# evaluating
raw_predictions = []
# runnable test snippet
# text = 'This is a test'
# tokenized_text = tokenizer(text,
#                       return_tensors="pt")
# input_ids = tokenized_text.input_ids
# output_ids = model.generate(input_ids, do_sample=True, top_p=0.84, top_k=100, max_length=20)
# decoded_outputs = tokenizer.batch_decode(output_ids,
#                                             skip_special_tokens=True,
#                                             clean_up_tokenization_spaces=True)
# print(decoded_outputs)

for i, batch in enumerate(testloader):
    print(f'evaluating batch {i}/{len(testloader)}')
    batch = batch.to(device)
    model = model.to(device)
    # check model and batch in correct device
    assert batch.input_ids.device == model.device == torch.device(device)


    output_ids = model.generate(
        batch.input_ids,
        max_length=cfg.model.max_length,
    )
    decoded_outputs = tokenizer.batch_decode(output_ids,
                                             skip_special_tokens=True,
                                             clean_up_tokenization_spaces=True)
    raw_predictions.extend(decoded_outputs)
#%%
# matching back to the testset

def formulate_quad(quad: Union[List, Tuple]):
    """
    Parameters:quadruple following diaasq original triplet format
    quad: [
                164,
                165,
                165,
                166,
                167,
                169,
                "neg",
                "IQOO9",
                "scheduling",
                "quite conservative"
            ],
    Returns: Tuple
    in the format (target, aspect, opn, pol)
    ("IQOO9", "scheduling", "quite conservative", "neg"")
    """
    pol = quad[6]
    target_string = quad[7]
    aspect_string = quad[8]
    opn_string = quad[9]
    formatter = '(%s, %s, %s, %s)'
    print(formatter % (target_string, aspect_string, opn_string, pol))

#%%
assert len(raw_predictions) == len(testset)
for i, doc in enumerate(testset):
    # need to shuffle = False in testloader
    doc_id = doc['doc_id']
    pred = raw_predictions[i]

    output = doc['label']
    # formulated_his_opn_quads = [formulate_quad(quad) for quad in his_opn_quads]
    print(f'Prediction: {pred}')
    print(f'Gold: {output}')
    print('Order does not matter')



# # testing compute_metrics
# training_args = Seq2SeqTrainingArguments(
#     output_dir=cfg.model.output_dir,
#     do_train=False,
#     do_eval=True,
#     evaluation_strategy='epoch',
#     per_device_train_batch_size=cfg.trainer.per_device_train_batch_size,
#     per_device_eval_batch_size=cfg.trainer.per_device_eval_batch_size,
#     # gradient_accumulation_steps=cfg.model.gradient_accumulation_steps if cfg.model.gradient_accumulation_steps else 1,
#     learning_rate=cfg.trainer.learning_rate,
#     optim=cfg.trainer.optim,
#     warmup_steps=cfg.trainer.warmup_steps,
#     generation_max_length=cfg.trainer.generation_max_length,
#     num_train_epochs=cfg.trainer.epochs,
#     weight_decay=cfg.trainer.weight_decay,
#     remove_unused_columns=False,
#     logging_steps=cfg.trainer.logging_steps,
#     # report_to="wandb",
# )
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     tokenizer=tokenizer,
#     data_collator=collate_fn,
#     compute_metrics=strict_sentiment_f1,
# )
# trainer.predict(testset)

# print(raw_predictions[0])

# %%
