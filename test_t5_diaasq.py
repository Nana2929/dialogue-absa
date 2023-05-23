import warnings
import yaml

from pathlib import Path
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer,
                        AutoModelForSeq2SeqLM,
                        Seq2SeqTrainer, Seq2SeqTrainingArguments)


from dataset.diaasq.speaker_spec_dataset import SpeakerDiaAsqDataset
from dataset.utils import SpeakerDiaAsqCollator

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
ckpt = Path(ckpt_dir) / 'checkpoint-8500'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)

# making test loader
collate_fn = SpeakerDiaAsqCollator(tokenizer=tokenizer, max_len=cfg.model.max_seq_length)
testset = SpeakerDiaAsqDataset(cfg.data.lang_src,
                               cfg.proc_data.data_root,
                               cfg.proc_data.test_ic_name,
                               cfg.proc_data.t5_test_split_name,
                               cfg.dataset.k,
                               cfg.seed,
                               cfg.dataset.prompt_path,
                               in_context_strategy=in_context_strategy)
testloader = DataLoader(testset,
                        batch_size=cfg.model.per_device_eval_batch_size,
                        shuffle=False,
                        num_workers=4,
                        collate_fn=collate_fn)
# evaluating
raw_predictions = []
for batch in testloader:
    batch = batch.to(device)
    model = model.to(device)
    output_ids = model.generate(
        batch,
        max_length=cfg.model.max_length,
    )
    decoded_outputs = tokenizer.batch_decode(output_ids,
                                             skip_special_tokens=True,
                                             clean_up_tokenization_spaces=True)
    for output in decoded_outputs:
        raw_predictions.append(output)

# testing compute_metrics
training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.model.output_dir,
        do_train=False, 
        do_eval=True,
        evaluation_strategy='epoch',
        per_device_train_batch_size=cfg.trainer.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.trainer.per_device_eval_batch_size,
        # gradient_accumulation_steps=cfg.model.gradient_accumulation_steps if cfg.model.gradient_accumulation_steps else 1,
        learning_rate=cfg.trainer.learning_rate,
        optim=cfg.trainer.optim,
        warmup_steps=cfg.trainer.warmup_steps,
        generation_max_length=cfg.trainer.generation_max_length,
        num_train_epochs=cfg.trainer.epochs,
        weight_decay=cfg.trainer.weight_decay,
        remove_unused_columns=False,
        logging_steps=cfg.trainer.logging_steps,
        # report_to="wandb",
    )
trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=strict_sentiment_f1,
    )
trainer.predict(testset)


trainer = Seq2SeqTrainer(
    model=model,





# matching back to the testset
assert len(raw_predictions) == len(testset)



print(raw_predictions[0])
