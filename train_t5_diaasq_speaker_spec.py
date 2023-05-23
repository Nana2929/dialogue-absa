'''
@File    :   train_t5_diaasq_speaker_spec.py
@Time    :   2023/05/22 20:24:51
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw
@Desc    :   Train t5 with speaker-specific DiaASQ data (due to t5 seq limit = 512, which roughly contains
             definitions, 1 in-context-example, and 1 test example input.
'''

from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer)
from argparse import ArgumentParser
from configs.utils import load_config
from dataset.diaasq.speaker_spec_dataset import SpeakerDiaAsqDataset
from dataset.utils import SpeakerDiaAsqCollator
from train.t5_metrics import calc_sentiment_scores


def main(args):

    # load yaml
    cfg = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name,
                    truncation_side = cfg.tokenizer.truncation_side,)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.model_name)
    in_context_strategy = cfg.dataset.in_context_strategy

    trainset = SpeakerDiaAsqDataset(cfg.data.lang_src,
                                    cfg.proc_data.data_root,
                                    cfg.proc_data.train_ic_name,
                                    cfg.proc_data.t5_train_split_name,
                                    k=cfg.dataset.k,
                                    seed=cfg.seed,
                                    prompt_path=cfg.dataset.prompt_path,
                                    in_context_strategy=in_context_strategy)
    testset = SpeakerDiaAsqDataset(cfg.data.lang_src,
                                   cfg.proc_data.data_root,
                                   cfg.proc_data.test_ic_name,
                                   cfg.proc_data.t5_test_split_name,
                                   cfg.dataset.k,
                                   cfg.seed,
                                   cfg.dataset.prompt_path,
                                   in_context_strategy=in_context_strategy)

    print('trainset length:', len(trainset))
    print('testset length:', len(testset))

    # https://github.com/kevinscaria/InstructABSA/blob/main/run_model.py#L69
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.model.output_dir,
        do_train=True,
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
        report_to="wandb",
    )
    data_collator = SpeakerDiaAsqCollator(tokenizer=tokenizer, max_len=cfg.model.max_length)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=testset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics= calc_sentiment_scores(tokenizer = tokenizer),
    )

    trainer.train()
    trainer.save_model()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config',
                        '-cfg',
                        type=str,
                        default='configs/diaasq-t5-speaker-spec-en.yaml')
    args = parser.parse_args()
    main(args)
