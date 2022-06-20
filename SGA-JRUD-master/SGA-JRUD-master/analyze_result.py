from typing import Sequence
from config import global_config as cfg
from eval import MultiWozEvaluator
from reader import MultiWozReader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json, random
import ontology
import torch
import numpy as np
from mwzeval.metrics import Evaluator
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
# change to your own model path
tokenizer=GPT2Tokenizer.from_pretrained('experiments_21/turn-level-DS/best_score_model')
reader = MultiWozReader(tokenizer)
evaluator = MultiWozEvaluator(reader)
std_evaluator=Evaluator(bleu=1, success=1, richness=0)

def compare_offline_result(path1, path2, show_num=10):
    succ1_unsuc2=[]
    succ2_unsuc1=[]
    data1=json.load(open(path1, 'r', encoding='utf-8'))
    data2=json.load(open(path2, 'r', encoding='utf-8'))
    dials1=evaluator.pack_dial(data1)
    dials2=evaluator.pack_dial(data2)
    counts = {}
    for req in evaluator.requestables:
        counts[req+'_total'] = 0
        counts[req+'_offer'] = 0
    dial_id_list=random.sample(reader.test_list, show_num)
    dial_samples=[]
    for dial_id in dials1:
        dial1=dials1[dial_id]
        dial2=dials2[dial_id]
        if dial_id+'.json' in dial_id_list:
            dial_samples.append({'dial1':dial1, 'dial2':dial2})
        reqs = {}
        goal = {}
        if '.json' not in dial_id and '.json' in list(evaluator.all_data.keys())[0]:
            dial_id = dial_id + '.json'
        for domain in ontology.all_domains:
            if evaluator.all_data[dial_id]['goal'].get(domain):
                true_goal = evaluator.all_data[dial_id]['goal']
                goal = evaluator._parseGoal(goal, true_goal, domain)
        # print(goal)
        for domain in goal.keys():
            reqs[domain] = goal[domain]['requestable']

        # print('\n',dial_id)
        success1, match1, _, _ = evaluator._evaluateGeneratedDialogue(dial1, goal, reqs, counts)
        success2, match2, _, _ = evaluator._evaluateGeneratedDialogue(dial2, goal, reqs, counts)
        if success1 and not success2:
            succ1_unsuc2.append(dial_id)
        elif success2 and not success1:
            succ2_unsuc1.append(dial_id)
    print('Success in data1 and unsuccess in data2:', len(succ1_unsuc2))#, succ1_unsuc2)
    print('Success in data2 and unsuccess in data1:', len(succ2_unsuc1))#, succ2_unsuc1)
    examples=[]
    for item in dial_samples:
        dialog=[]
        for turn1, turn2 in zip(item['dial1'], item['dial2']):
            if turn1['user']=='':
                continue
            entry={'user': turn1['user'], 'Oracle':turn1['resp'], 'Sup':turn1['resp_gen'], 'RL':turn2['resp_gen']}
            dialog.append(entry)
        examples.append(dialog)
    json.dump(examples, open('analysis/examples.json', 'w'), indent=2)
            
def compare_online_result(path1, path2):
    succ1_unsuc2=[]
    succ2_unsuc1=[]
    data1=json.load(open(path1, 'r', encoding='utf-8'))
    data2=json.load(open(path2, 'r', encoding='utf-8'))
    counts = {}
    for req in evaluator.requestables:
        counts[req+'_total'] = 0
        counts[req+'_offer'] = 0
    flag1=0
    flag2=0
    for i, dial_id in enumerate(reader.test_list):
        reqs = {}
        goal = {}
        dial1=data1[i]
        dial2=data2[i]
        if isinstance(dial1, list):
            data1[i]={dial_id:dial1}
            flag1=1
        elif isinstance(dial1, dict):
            dial1=dial1[dial_id]
        
        if isinstance(dial2, list):
            data2[i]={dial_id:dial2}
            flag2=1
        elif isinstance(dial2, dict):
            dial2=dial2[dial_id]

        init_goal=reader.data[dial_id]['goal']
        for domain in ontology.all_domains:
            if init_goal.get(domain):
                true_goal = init_goal
                goal = evaluator._parseGoal(goal, true_goal, domain)
        for domain in goal.keys():
            reqs[domain] = goal[domain]['requestable']
        success1, match2, _, _ = evaluator._evaluateGeneratedDialogue(dial1, goal, reqs, counts)
        success2, match2, _, _ = evaluator._evaluateGeneratedDialogue(dial2, goal, reqs, counts)
        if success1 and not success2:
            succ1_unsuc2.append(dial_id)
        elif success2 and not success1:
            succ2_unsuc1.append(dial_id)
    print('Success in data1 and unsuccess in data2:', len(succ1_unsuc2), succ1_unsuc2)
    print('Success in data2 and unsuccess in data1:', len(succ2_unsuc1), succ2_unsuc1)
    if flag1:
        json.dump(data1, open(path1, 'w'), indent=2)
    if flag2:
        json.dump(data2, open(path2, 'w'), indent=2)

def group_act(act):
    for domain in act:
        for intent, sv in act[domain].items():
            act[domain][intent]=set(sv)
    return act

def group_state(state):
    for domain, sv in state.items():
        state[domain]=set(sv)
    return state

def find_unseen_usr_act(path1=None, path2=None):
    data=json.load(open('data/multi-woz-2.1-processed/data_for_us.json', 'r', encoding='utf-8'))
    train_act_pool=[]
    unseen_act_pool=[]
    unseen_dials=[]
    for dial_id, dial in data.items():
        if dial_id in reader.train_list:
            for turn in dial:
                user_act=reader.aspan_to_act_dict(turn['usr_act'], 'user')
                user_act=group_act(user_act)
                if user_act not in train_act_pool:
                    train_act_pool.append(user_act)
    for dial_id, dial in data.items():
        if dial_id in reader.test_list:# or dial_id in reader.dev_list:
            unseen_turns=0
            for turn in dial:
                user_act=reader.aspan_to_act_dict(turn['usr_act'], 'user')
                user_act=group_act(user_act)
                if user_act not in train_act_pool:
                    unseen_act_pool.append(user_act)
                    unseen_turns+=1
            if unseen_turns>0:
                unseen_dials.append(dial_id)
    print('Total training acts:', len(train_act_pool), 'Unseen acts:', len(unseen_act_pool))
    print('Unseen dials:',len(unseen_dials))
    if path1 and path2:
        data1=json.load(open(path1, 'r', encoding='utf-8'))
        data2=json.load(open(path2, 'r', encoding='utf-8'))
        unseen_act_pool1=[]
        unseen_act_pool2=[]
        for dial1, dial2 in zip(data1, data2):
            dial1=list(dial1.values())[0]
            dial2=list(dial2.values())[0]
            for turn in dial1:
                user_act=reader.aspan_to_act_dict(turn['usr_act'], 'user')
                user_act=group_act(user_act)
                if user_act not in train_act_pool:
                    unseen_act_pool1.append(user_act)
            for turn in dial2:
                user_act=reader.aspan_to_act_dict(turn['usr_act'], 'user')
                user_act=group_act(user_act)
                if user_act not in train_act_pool:
                    unseen_act_pool2.append(user_act)
        print('Unseen acts in path1:', len(unseen_act_pool1))
        print('Unseen acts in path2:', len(unseen_act_pool2))
    return unseen_dials

def count_act(data):
    act_pool=[]
    for turn in data:
        if turn['user']=='':
            continue
        else:
            sys_act=reader.aspan_to_act_dict(turn['aspn_gen'], 'sys')
            sys_act=group_act(sys_act)
            if sys_act not in act_pool:
                act_pool.append(sys_act)
    print('Act num:', len(act_pool))

def count_online(data):
    user_act_pool=[]
    sys_act_pool=[]
    for dial in data:
        for turn in dial:
            user_act=reader.aspan_to_act_dict(turn['usr_act'], 'user')
            user_act=group_act(user_act)
            if user_act not in user_act_pool:
                user_act_pool.append(user_act)
            sys_act=reader.aspan_to_act_dict(turn['aspn'], 'sys')
            sys_act=group_act(sys_act)
            if sys_act not in sys_act_pool:
                sys_act_pool.append(sys_act)
    print('Total sys act:', len(sys_act_pool), 'total user acts:', len(user_act_pool))

def count_state(data):
    state_pool=[]
    act_pool=[]
    for turn in data:
        if turn['user']=='':
            continue
        state=reader.bspan_to_constraint_dict(turn['bspn_gen'])
        state=group_state(state)
        act=reader.aspan_to_act_dict(turn['aspn_gen'], 'sys')
        act=group_act(act)
        if state not in state_pool:
            state_pool.append(state)
            act_pool.append([act])
        elif act not in act_pool[state_pool.index(state)]:
            act_pool[state_pool.index(state)].append(act)
    print('Total states:',len(state_pool), 'Average actions per state:', np.mean([len(item) for item in act_pool]))

def find_unseen_sys_act():
    data=json.load(open('data/multi-woz-2.1-processed/data_for_us.json', 'r', encoding='utf-8'))
    train_act_pool=[]
    unseen_act_pool=[]
    test_act_pool=[]
    unseen_dials={}
    for dial_id, dial in data.items():
        if dial_id in reader.train_list:
            for turn in dial:
                sys_act=reader.aspan_to_act_dict(turn['sys_act'], 'sys')
                sys_act=group_act(sys_act)
                if sys_act not in train_act_pool:
                    train_act_pool.append(sys_act)
    for dial_id, dial in data.items():
        if dial_id in reader.test_list:# or dial_id in reader.dev_list:
            unseen_turns=[]
            for turn_id, turn in enumerate(dial):
                sys_act=reader.aspan_to_act_dict(turn['sys_act'], 'sys')
                sys_act=group_act(sys_act)
                if sys_act not in test_act_pool:
                    test_act_pool.append(sys_act)
                if sys_act not in train_act_pool:
                    unseen_act_pool.append(sys_act)
                    unseen_turns.append(turn_id)
            if len(unseen_turns)>0:
                unseen_dials[dial_id]=unseen_turns
    print('Total training acts:', len(train_act_pool), 'test acts:',len(test_act_pool),'Unseen acts:', len(unseen_act_pool))
    print('Unseen dials:',len(unseen_dials))
    json.dump(unseen_dials, open('analysis/unseen_turns.json', 'w'), indent=2)

    return unseen_dials

def calculate_unseen_acc(unseen_turns, path1=None, path2=None):
    data1=json.load(open(path1, 'r', encoding='utf-8'))
    data2=json.load(open(path2, 'r', encoding='utf-8'))
    total_unseen_act=0
    sup_acc=0
    rl_acc=0
    tp1=0
    fp1=0
    tp2=0
    fp2=0
    count=0
    for dial_id in unseen_turns:
        for t in unseen_turns[dial_id]:
            count+=1
    print('Total unseen act:', count)
    for turn1, turn2 in zip(data1, data2):
        dial_id=turn1['dial_id']+'.json'
        if dial_id in unseen_turns and turn1['user']!='' and turn1['turn_num'] in unseen_turns[dial_id]:
            total_unseen_act+=1
            #unseen_turns[dial_id]=unseen_turns[dial_id][1:]
            oracle_act=group_act(reader.aspan_to_act_dict(turn1['aspn'], side='sys'))
            sup_act=group_act(reader.aspan_to_act_dict(turn1['aspn_gen'], side='sys'))
            rl_act=group_act(reader.aspan_to_act_dict(turn2['aspn_gen'], side='sys'))
            if sup_act==oracle_act:
                sup_acc+=1
            if rl_act==oracle_act:
                rl_acc+=1
            for domain in sup_act:
                for intent, slots in sup_act[domain].items():
                    if domain not in oracle_act or intent not in oracle_act[domain]:
                        fp1+=len(slots)
                        continue
                    for slot in slots:
                        if slot in oracle_act[domain][intent]:
                            tp1+=1
                        else:
                            fp1+=1
            for domain in rl_act:
                for intent, slots in rl_act[domain].items():
                    if domain not in oracle_act or intent not in oracle_act[domain]:
                        fp2+=len(slots)
                        continue
                    for slot in slots:
                        if slot in oracle_act[domain][intent]:
                            tp2+=1
                        else:
                            fp2+=1
    print('Total unseen acts:{}, Sup acc:{}, RL acc:{}'.format(total_unseen_act, sup_acc, rl_acc))
    print(tp1, fp1, tp1/(tp1+fp1))
    print(tp2, fp2, tp2/(tp2+fp2))

def extract_goal():
    data=json.load(open('data/multi-woz-2.1-processed/data_for_damd_fix.json', 'r', encoding='utf-8'))
    goal_list=[]
    for dial_id, dial in data.items():
        goal=dial['goal']
        goal_list.append(goal)
    json.dump(goal_list, open('analysis/goals.json', 'w'), indent=2)

def prepare_for_std_eval(path=None, data=None):
    if path:
        data=json.load(open(path, 'r', encoding='utf-8'))
    new_data={}
    dials=evaluator.pack_dial(data)
    for dial_id in dials:
        new_data[dial_id]=[]
        dial=dials[dial_id]
        for turn in dial:
            if turn['user']=='':
                continue
            entry={}
            entry['response']=turn['resp_gen']
            entry['state']=reader.bspan_to_constraint_dict(turn['bspn_gen'])
            new_data[dial_id].append(entry)
    if path:
        new_path=path[:-5]+'std.json'
        json.dump(new_data, open(new_path, 'w'), indent=2)
    return new_data

def get_attentions(model_path, mode='bspn', encode_key=['user', 'bspn', 'db', 'aspn', 'resp'], turn_th=4):
    tok=GPT2Tokenizer.from_pretrained(model_path)
    model=GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    data=json.load(open('data/multi-woz-2.1-processed/data_for_rl.json', 'r', encoding='utf-8'))
    key_pool=['user', 'bspn', 'db', 'aspn', 'resp']
    num=len(encode_key)*(turn_th-1)+key_pool.index(mode)
    attention_list=[]
    count=0
    for dial_id, dial in data.items():
        if dial_id not in reader.test_list:
            continue
        if len(dial['log'])<turn_th:
            continue
        for turn in dial['log']:
            for key, sent in turn.items():
                if key in key_pool:
                    turn[key]=tok.encode(sent)
        sequence=[]
        st_idx_list=[]
        ed_idx_list=[]
        flag=0
        for id, turn in enumerate(dial['log']):
            if id<turn_th-1:
                for key in encode_key:
                    id1=len(sequence)+1
                    id2=len(sequence)+len(turn[key])-1
                    sequence+=turn[key]
                    st_idx_list.append(id1)
                    ed_idx_list.append(id2)
            elif id==turn_th-1:
                for key in key_pool:
                    id1=len(sequence)+1
                    id2=len(sequence)+len(turn[key])-1
                    sequence+=turn[key]
                    if key==mode:
                        st_idx=id1
                        ed_idx=id2
                        flag=1
                        break
                    st_idx_list.append(id1)
                    ed_idx_list.append(id2)
            if flag:
                break
        assert len(st_idx_list)==num
        if len(sequence)>1024:
            continue
        with torch.no_grad():
            outputs=model.forward(torch.tensor([sequence]), return_dict=True, output_attentions=True)
        attentions=outputs.attentions[-1] #last layer
        #ed_idx=min(st_idx+max_len, ed_idx)
        attention=torch.mean(attentions[0, :, st_idx:ed_idx,], dim=0) # T_b, T
        avg1=torch.mean(attention, dim=0) #T
        entry=[]
        for id1, id2 in zip(st_idx_list, ed_idx_list):
            avg2=avg1[id1:id2].mean().item()
            if np.isnan(avg2):
                avg2=0
            entry.append(avg2)
        attention_list.append(entry)
        count+=1

        '''
        attention=attention[:,1:st_idx]
        attention/=attention.max()
        plt.figure()
        plt.imshow(attention.numpy(), cmap=plt.cm.hot)
        plt.xlabel('Previous information')
        plt.ylabel('Belief state')
        plt.colorbar()
        #plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        #plt.yticks(np.arange(ed_idx-st_idx-1))
        plt.title('Attentions')
        plt.savefig('analysis/attention.png')
        #plt.show()
        break
        '''
    print('Count dials:', count)
    print(len(attention_list))
    print(list(np.mean(attention_list, axis=0)))
    print(list(np.var(attention_list, axis=0)))

def get_attentions1(model_path, mode='bspn', encode_key=['bspn','resp'], turn_th=4):
    tok=GPT2Tokenizer.from_pretrained(model_path)
    model=GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    data=json.load(open('data/multi-woz-2.1-processed/data_for_rl.json', 'r', encoding='utf-8'))
    key_pool=['user', 'bspn', 'db', 'aspn', 'resp']
    num=len(encode_key)+key_pool.index(mode)
    attention_list=[]
    count=0
    for dial_id, dial in data.items():
        if dial_id not in reader.test_list:
            continue
        if len(dial['log'])<turn_th:
            continue
        for turn in dial['log']:
            for key, sent in turn.items():
                if key in key_pool:
                    turn[key]=tok.encode(sent)
        sequence=[]
        st_idx_list=[]
        ed_idx_list=[]
        flag=0
        for id, turn in enumerate(dial['log']):
            if id==turn_th-2:
                for key in encode_key:
                    id1=len(sequence)+1
                    id2=len(sequence)+len(turn[key])-1
                    sequence+=turn[key]
                    st_idx_list.append(id1)
                    ed_idx_list.append(id2)
            elif id==turn_th-1:
                for key in key_pool:
                    id1=len(sequence)+1
                    id2=len(sequence)+len(turn[key])-1
                    sequence+=turn[key]
                    if key==mode:
                        st_idx=id1
                        ed_idx=id2
                        flag=1
                        break
                    st_idx_list.append(id1)
                    ed_idx_list.append(id2)
            if flag:
                break
        assert len(st_idx_list)==num
        if len(sequence)>1024:
            continue
        with torch.no_grad():
            outputs=model.forward(torch.tensor([sequence]), return_dict=True, output_attentions=True)
        attentions=outputs.attentions[-1] #last layer
        #ed_idx=min(st_idx+max_len, ed_idx)
        attention=torch.mean(attentions[0, :, st_idx:ed_idx,], dim=0) # T_b, T
        avg1=torch.mean(attention, dim=0) #T
        entry=[]
        for id1, id2 in zip(st_idx_list, ed_idx_list):
            avg2=avg1[id1:id2].mean().item()
            if np.isnan(avg2):
                avg2=0
            entry.append(avg2)
        attention_list.append(entry)
        count+=1
    print('Count dials:', count)
    print(len(attention_list))
    print(list(np.mean(attention_list, axis=0)))
    print(list(np.var(attention_list, axis=0)))

def find_attention_case(model_path, mode='bspn', encode_key=['user', 'bspn', 'db', 'aspn', 'resp'], turn_th=4):
    tok=GPT2Tokenizer.from_pretrained(model_path)
    model=GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    data=json.load(open('data/multi-woz-2.1-processed/data_for_rl.json', 'r', encoding='utf-8'))
    key_pool=['user', 'bspn', 'db', 'aspn', 'resp']
    num=len(encode_key)*(turn_th-1)+key_pool.index(mode)
    attention_list=[]
    count=0
    for dial_id, dial in data.items():
        if dial_id not in reader.test_list:
            continue
        if len(dial['log'])<turn_th:
            continue
        for turn in dial['log']:
            for key, sent in turn.items():
                if key in key_pool:
                    turn[key]=reader.modified_encode(sent, tok)
        sequence=[]
        st_idx_list=[]
        ed_idx_list=[]
        flag=0
        for id, turn in enumerate(dial['log']):
            if id<turn_th-1:
                for key in encode_key:
                    id1=len(sequence)+1
                    id2=len(sequence)+len(turn[key])-1
                    sequence+=turn[key]                  
                    st_idx_list.append(id1)
                    ed_idx_list.append(id2)
            elif id==turn_th-1:
                for key in key_pool:
                    id1=len(sequence)+1
                    id2=len(sequence)+len(turn[key])-1
                    sequence+=turn[key]
                    if key==mode:
                        st_idx=id1
                        ed_idx=id2
                        flag=1
                        break
                    #st_idx_list.append(id1)
                    #ed_idx_list.append(id2)
            if flag:
                break
        print('Previous variables:', len(st_idx_list))
        if len(sequence)>1024:
            continue
        with torch.no_grad():
            outputs=model.forward(torch.tensor([sequence]), return_dict=True, output_attentions=True)
        attentions=outputs.attentions[-1] #last layer
        #ed_idx=min(st_idx+max_len, ed_idx)
        attentions=torch.mean(attentions[0, :, st_idx:ed_idx,], dim=0) # T_b, T
        bs=tok.convert_ids_to_tokens(sequence[st_idx:ed_idx])
        bs=[item.strip('Ġ') for item in bs]
        for i, (id1, id2) in enumerate(zip(st_idx_list, ed_idx_list)):
            if id1==id2:
                continue
            sos_id=['<sos_u>'] if i%2==0 else ['<sos_b>']
            eos_id=['<eos_u>'] if i%2==0 else ['<eos_b>']
            temp=torch.zeros(attentions.size(0), 1)
            if i==0:
                attention1=torch.cat([temp, attentions[:,id1+1:id2+1], temp],dim=1)
                pv_bs1=sos_id+tok.convert_ids_to_tokens(sequence[id1:id2])+eos_id
            elif i==1:
                attention2=torch.cat([temp, attentions[:,id1+1:id2+1], temp],dim=1)
                pv_bs2=sos_id+tok.convert_ids_to_tokens(sequence[id1:id2])+eos_id
            elif i%2==0:
                attention1=torch.cat([attention1, temp, attentions[:,id1+1:id2+1], temp], dim=1)
                pv_bs1+=sos_id+tok.convert_ids_to_tokens(sequence[id1:id2])+eos_id
            else:
                attention2=torch.cat([attention2, temp, attentions[:,id1+1:id2+1], temp], dim=1)
                pv_bs2+=sos_id+tok.convert_ids_to_tokens(sequence[id1:id2])+eos_id
        pv_bs1=[item.strip('Ġ') for item in pv_bs1]
        pv_bs2=[item.strip('Ġ') for item in pv_bs2]
        plt.figure()
        ax=plt.gca()
        im=ax.imshow(attention1.numpy(), cmap=plt.cm.hot)
        # recttuple (left, bottom, right, top) default: (0, 0, 1, 1)
        # 0: most left or most bottom
        # 1: most right or most top
        plt.tight_layout(rect=(0.1, 0.1, 0.9, 1))
        plt.xticks(np.arange(len(pv_bs1)), pv_bs1, rotation=90, fontsize=7)
        plt.yticks(np.arange(len(bs)),labels=bs, fontsize=8)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        #plt.title('Attentions')
        plt.savefig('analysis/attention_u.png')

        plt.figure()
        ax=plt.gca()
        im=ax.imshow(attention2.numpy(), cmap=plt.cm.hot)
        plt.tight_layout(rect=(0.1, 0.1, 0.9, 1))
        plt.xticks(np.arange(len(pv_bs2)), pv_bs2, rotation=90, fontsize=7)
        plt.yticks(np.arange(len(bs)),labels=bs, fontsize=8)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        #plt.title('Attentions')
        plt.savefig('analysis/attention_b.png')
        '''
        for i, (id1, id2) in enumerate(zip(st_idx_list, ed_idx_list)):
            if id1==id2:
                continue
            attention=attentions[:,id1+1:id2+1]
            pv_bs=tok.convert_ids_to_tokens(sequence[id1:id2])
            pv_bs=[item.strip('Ġ') for item in pv_bs]
            print(pv_bs)
            #attention/=attention.max()
            plt.figure()
            plt.imshow(attention.numpy(), cmap=plt.cm.hot)
            #plt.xlabel('${b_%d}$'%(i+1))
            #plt.ylabel('${b_%d}$'%(i+2))
            plt.colorbar()
            plt.tight_layout(rect=(0.25, 0.25, 1, 1))
            plt.xticks(np.arange(len(pv_bs)), pv_bs, rotation=90)
            plt.yticks(np.arange(len(bs)),labels=bs)
            #plt.title('Attentions')
            plt.savefig('analysis/attention_%d.png'%(i+1))
        '''
        t=1
        break

def length_statistics():
    data=json.load(open('data/multi-woz-2.1-processed/new_db_se_blank_encoded.data.json', 'r', encoding='utf-8'))
    #session-level
    total=0
    exceed=0
    mean_len, max_len=0, 0
    len_list=[]
    for dial in data['train']:
        length=0
        total+=1
        for turn in dial:
            length+=len(turn['user']+turn['bspn']+turn['db']+turn['aspn']+turn['resp'])
        if length>1024:
            exceed+=1
            print(length)
        len_list.append(length)
        mean_len+=length/len(data['train'])
        if length>max_len:
            max_len=length
    print('Total training sequences:', total, 'sequences exceeding limit:', exceed)
    print('Mean length:{}, max length:{}'.format(mean_len, max_len))
    print(np.mean(len_list), np.sqrt(np.var(len_list)))
    #turn-level
    total=0
    exceed=0
    mean_len, max_len=0, 0
    len_list=[]
    total_turn=sum([len(dial) for dial in data['train']])
    for dial in data['train']:
        total+=1
        history_len=0
        for turn in dial:
            total+=1
            length = history_len+len(turn['user']+turn['bspn']+turn['db']+turn['aspn']+turn['resp'])
            #history_len+=len(turn['user']+turn['resp'])
            history_len=len(turn['bspn']+turn['resp'])
            if length>1024:
                exceed+=1
            len_list.append(length)
            mean_len+=length/total_turn
            if length>max_len:
                max_len=length
    print('Total training sequences:', total, 'sequences exceeding limit:', exceed)
    print('Mean length:{}, max length:{}'.format(mean_len, max_len))
    print(np.mean(len_list), np.sqrt(np.var(len_list)))

def get_success_list(path, prepared=False, dial_order=None):
    results=json.load(open(path, 'r'))
    input_data=prepare_for_std_eval(data=results) if not prepared else results
    if dial_order:
        new_data={}
        for dial_id in dial_order:
            if dial_id not in input_data:
                print('No dial id:', dial_id)
                continue
            new_data[dial_id]=input_data[dial_id]
        input_data=new_data
    std_evaluator.evaluate(input_data, return_all=True)
    return list(input_data.keys())

def compare_list(list1, list2):
    c1=0
    c2=0
    for t1, t2 in zip(list1, list2):
        if t1 and not t2:
            c1+=1
        elif not t1 and t2:
            c2+=1
    print(c1,c2)


if __name__=='__main__':
    '''
    path='/home/liuhong/myworkspace/experiments_21/RL-DS-baseline/best_score_model/result.json'
    data1=json.load(open(path, 'r', encoding='utf-8'))
    count_act(data1)
    count_state(data1)
    path='/home/liuhong/myworkspace/RL_exp/RL-1-10-beam-1/best_DS/result.json'
    data2=json.load(open(path, 'r', encoding='utf-8'))
    count_act(data2)
    count_state(data2)
    
    path='/home/liuhong/myworkspace/experiments_21/RL-DS-baseline/best_score_model/validate_result.json'
    data=json.load(open(path, 'r', encoding='utf-8'))
    count_online(data)
    path='/home/liuhong/myworkspace/RL_exp/RL-1-10-beam-1/best_DS/validate_result.json'
    data=json.load(open(path, 'r', encoding='utf-8'))
    count_online(data)
    '''
    #get_success_list('/home/liuhong/myworkspace/RL_exp/RL-1-5-only_aspn/best_DS/result.json')
    #dial_order=get_success_list('/home/liuhong/myworkspace/RL_exp/RL-12-30/best_DS/result.json')
    #get_success_list('/home/liuhong/MultiWOZ_Evaluation-master/predictions/ubar.json', prepared=True, dial_order=dial_order)
    #list1 = [1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1]
    list1 = [1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1]
    #list2 = [1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1]
    list2=[1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
    #compare_list(list1, list2)
    length_statistics()
    #find_unseen_sys_act()
    #find_attention_case('experiments_21/all_UBAR-wsl_sd11_lr0.0001_bs2_ga16/best_score_model', mode='bspn', turn_th=4, encode_key=['user', 'bspn'])
    #get_attentions1('experiments_21/turn-level-DS/best_score_model', encode_key=['bspn', 'resp'], mode='aspn', turn_th=4)
    #get_attentions('experiments_21/all_HRU-otl_sd11_lr0.0001_bs8_ga4/best_score_model', encode_key=['user', 'resp'], turn_th=5)
    #prepare_for_std_eval(path1)
    #path2='RL_exp/rl-10-19-use-scheduler/best_DS/result.json'
    #unseen_turns=find_unseen_sys_act()
    #calculate_unseen_acc(unseen_turns, path1, path2)
    #compare_offline_result(path1, path2, show_num=30)
    #path1='experiments_21/turn-level-DS/best_score_model/validate_result.json'
    #path2='RL_exp/rl-10-19-use-scheduler/best_DS/validate_result.json'
    #compare_online_result(path1, path2)
    #bspn='[restaurant] pricerange expensive area west'
    #print(reader.bspan_to_DBpointer(bspn, ['restaurant']))
    #unseen_dials=find_unseen_usr_act(path1, path2)
    #print(unseen_dials)
    #act='[taxi] [inform] destination cambridge train station [taxi] [request] car'
    #print(reader.aspan_to_act_dict(act, 'user'))
    #print(set(reader.aspan_to_act_dict(act, 'user')))
    #extract_goal()
