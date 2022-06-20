path1=$3
path2=$4
exp_name=$2
python session.py -mode train\
 -cfg use_scheduler=False\
 lr=2e-5\
 weight_decay_count=100\
 seed=11\
 epoch_num=100\
 training_batch_size=16\
 rl_accumulation_steps=12\
 interaction_batch_size=32\
 DS_path=$path1 US_path=$path2\
 DS_device=$1 US_device=$1\
 rl_for_bspn=True\
 non_neg_reward=True\
 rl_dial_per_epoch=128\
 joint_train_ds=False\
 joint_train_us=True\
 simple_reward=True\
 simple_training=False\
 exp_no=$exp_name\
 rl_iterate=True\
 RL_ablation=False\
 iterative_update=False\
 on_policy=True\
 beam_search=True\
 beam_size=10

path2=RL_exp/${exp_name}/best_US
python session.py -mode train\
 -cfg use_scheduler=False\
 lr=2e-5\
 weight_decay_count=100\
 seed=11\
 epoch_num=100\
 training_batch_size=16\
 rl_accumulation_steps=12\
 interaction_batch_size=32\
 DS_path=$path1 US_path=$path2\
 DS_device=$1 US_device=$1\
 rl_for_bspn=True\
 non_neg_reward=True\
 rl_dial_per_epoch=128\
 joint_train_ds=True\
 joint_train_us=False\
 simple_reward=True\
 simple_training=False\
 exp_no=$exp_name\
 rl_iterate=True\
 RL_ablation=False\
 iterative_update=False\
 on_policy=True\
 beam_search=True\
 beam_size=10
