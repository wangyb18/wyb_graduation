# SGA-JRUD
This is the code of Jointly Reinforced User Simulator and Task-oriented Dialog System with Simplified Generative Architecture (SGA-JRUD).
## Requirements
After you create an environment with `python 3.6`, the following commands are recommended to install the corresponding package.
* pip install torch==1.5
* pip install transformers==3.5
* pip install spacy==3.1
* python -m spacy download en_core_web_sm
* pip install nltk
* pip install sklearn
* pip install tensorboard
* pip install future

Besides, you need to install the [standard evaluation repository](https://github.com/Tomiinek/MultiWOZ_Evaluation) for evaluation, in which we change the references in `mwzeval/utils.py/load_references()` to 'damd', since we adopt the same delexicalization as [DAMD](https://github.com/thu-spmi/damd-multiwoz), 

## Data Preprocessing
The processed data is provided in the directory `data/`, unzip it with
```
unzip data/data.zip -d data/
```
If you want to preprocess from scratch, you need to download [MultiWOZ2.1](https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.1.zip) to `data/` and unzip it. Then execute the following command
```
python data_analysis.py
python preprocess.py
```
In addition, the database files are needed
```
unzip db.zip
```
## Training

### Supervised Pretraining
To pretrain the dialog system (DS), run
```
bash pretrain_ds.sh $GPU ${your_exp_name}
```
e.g.
```
bash pretrain_ds.sh 0 DS-baseline
```
To pretrain the user simulator (US), run
```
bash pretrain_us.sh $GPU ${your_exp_name}
```
### Reinforcement Learning (RL)
To implement RL experiments, run
```
bash run_RL.sh $GPU ${your_exp_name} $DS_path $US_path
```
where DS_path and US_path are the paths of pretrained DS and US.
## Evaluation 
Online evaluation (Let DS and US interact with each other and evaluate the quality of generated dialogue)
```
bash eval_RL.sh $GPU $DS_path $US_path
```
Offline evaluation (Corpus-based evaluation of DS)
```
bash test.sh $GPU $DS_path
```
We also provide the generated result file of SGA-DS-SL in `result.json`.