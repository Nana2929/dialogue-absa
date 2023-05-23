
exp_config_name='diaasq-t5-speaker-spec-en'
log_dir='logs'
time_stamp=$(date "+%Y.%m.%d-%H.%M.%S")
python train_t5_diaasq_speaker_spec.py \
    -cfg=configs/${exp_config_name}.yaml > ${log_dir}/${exp_config_name}${time_stamp}.log