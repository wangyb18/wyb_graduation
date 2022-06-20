python train_nce.py -mode train\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=4 batch_size=8\
    epoch_num=50\
    exp_no=$2\
    cuda_device=$1\
    train_us=False\
    save_type=max_score\
    turn_level=True\
    input_history=False\
    input_prev_resp=True\
    evaluate_during_training=False