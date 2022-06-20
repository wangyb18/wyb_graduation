path=$2
python pretrain.py -mode test\
    -cfg gpt_path=$path  cuda_device=$1\
    fast_validate=True model_act=True dataset=1\
    debugging=False\
    fix_data=True\
    turn_level=True\
    input_history=False\
    input_prev_resp=True\
    test_unseen_act=False\
    eval_resp_prob=False\
    same_eval_as_cambridge=True\
    use_existing_result=True\
    eval_as_simpletod=True\
    eval_batch_size=32