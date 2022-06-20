import json
from sklearn.metrics import f1_score, accuracy_score
import sys
import numpy as np
import csv
from dst import ignore_none, default_cleaning, paser_bs
import argparse

def load_result(result_path):
    results=[]
    with open(result_path, 'r') as rf:
        reader=csv.reader(rf)
        for n,line in enumerate(reader):
            entry={}
            if n>0:
                if n==1:
                    field=line
                else:
                    for i,key in enumerate(field):
                        entry[key]=line[i]
                    results.append(entry)
    return results,field

def compute_jacc(data,default_cleaning_flag=True):
    num_turns = 0
    joint_acc = 0
    clean_tokens = ['<|endoftext|>', ]
    for turn_data in data:
        if 'user' in turn_data and turn_data['user']=='':
            continue
        turn_target = turn_data['bspn']
        turn_pred = turn_data['bspn_gen']
        turn_target = paser_bs(turn_target)
        turn_pred = paser_bs(turn_pred)
        for bs in turn_pred:
            if bs in clean_tokens + ['', ' '] or bs.split()[-1] == 'none':
                turn_pred.remove(bs)
        new_turn_pred = []
        for bs in turn_pred:
            for tok in clean_tokens:
                bs = bs.replace(tok, '').strip()
                new_turn_pred.append(bs)
        turn_pred = new_turn_pred
        turn_pred, turn_target = ignore_none(turn_pred, turn_target)
        if default_cleaning_flag:
            turn_pred, turn_target = default_cleaning(turn_pred, turn_target)
        join_flag = False
        if set(turn_target) == set(turn_pred):
            joint_acc += 1
            join_flag = True
        
        if not join_flag:
            turn_data['gtbs'] = turn_target
            turn_data['predbs'] = turn_pred
        num_turns += 1

    joint_acc /= num_turns
    
    #print('joint accuracy: {}'.format(joint_acc))
    return joint_acc

if __name__ == "__main__":
    result_path='/home/liuhong/UBAR-MultiWOZ/experiments/full_train/best_model_pri/result.csv'
    results,field=load_result(result_path)
    joint_acc=compute_jacc(results)
    print(joint_acc)