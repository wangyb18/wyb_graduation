import json
slot_list_act=[]
intent_list=[]
from reader import MultiWozReader
from transformers import GPT2Tokenizer
import json
import ontology
tokenizer=GPT2Tokenizer.from_pretrained('experiments_21/turn-level-DS/best_score_model')
reader = MultiWozReader(tokenizer)

def act_dict_to_aspn(act):
    #<sos_a> [restaurant] [offerbooked]  reference [general] [reqmore] <eos_a>
    slot_map={'post':'postcode','addr':'address','ref':'reference','dest':'destination','depart':'departure', 'fee':'price', 'entrance fee':'price', 'ticket':'price'}
    act_list=[]
    for key in act:
        domain=key.split('-')[0].lower()
        intent=key.split('-')[1].lower()
        act_list.append('['+domain+']')
        act_list.append('['+intent+']')
        if intent not in intent_list:
            intent_list.append(intent)
        if domain !='general':
            if intent=='inform':
                for item in act[key]:
                    slot=item[0].lower()
                    slot=slot_map.get(slot,slot)
                    act_list.append(slot)#slot
                    act_list.append(item[1].lower())#value
                    if slot not in slot_list_act:
                        #slot_list_act.append(item[0])
                        slot_list_act.append(slot)
            elif intent=='request':
                for item in act[key]:
                    slot=item[0].lower()
                    slot=slot_map.get(slot,slot)
                    act_list.append(slot)#slot
                    if slot not in slot_list_act:
                        slot_list_act.append(slot)
    aspn=' '.join(act_list)         
    return aspn

def goal_to_gpan(goal, slot_list_goal=None):
    slot_list_goal=[] if slot_list_goal==None else slot_list_goal
    token_map={'info':'[inform]','reqt':'[request]','fail_info':'[fail_info]','book':'[book]','fail_book':'[fail_book]'}
    goal_list=[]
    for domain in goal:
        goal_list.append('['+domain+']')
        for intent in goal[domain]:
            if intent in ['fail_book','fail_info']:
                # we do not consider the failing condition 
                continue
            goal_list.append(token_map.get(intent,''))
            if isinstance(goal[domain][intent],dict):
                for slot,value in goal[domain][intent].items():
                    slot=slot.lower()
                    value=value.lower() if isinstance(value, str) else value
                    if slot in ['pre_invalid','invalid']:
                        continue
                        '''
                        if slot=='pre_invalid':
                            goal_list.append('[pre_invalid]')
                        elif slot=='invalid':
                            goal_list.append('[invalid]')
                            value='yes' if value==True else 'no'
                            goal_list.append(value)
                        '''
                    else:
                        # some special case
                        if slot=='trainid':
                            slot='id'
                        if slot=='car type':
                            slot='car'
                        if slot in ['entrance fee', 'fee']:
                            slot='price'
                        if slot=='duration':
                            slot='time'
                        slot='arrive' if slot=='arriveby' else slot
                        slot='leave' if slot=='leaveat' else slot
                        goal_list.append(slot)
                        # some request goal may be in the form of dict
                        if value!='' and value!='?':
                            goal_list.append(str(value))
                    if slot not in slot_list_goal:
                        slot_list_goal.append(slot)
            elif isinstance(goal[domain][intent],list):
                for slot in goal[domain][intent]:
                    slot=slot.lower()
                    goal_list.append(slot)
                    if slot not in slot_list_goal:
                        slot_list_goal.append(slot)
    gpan=' '.join(goal_list)
    #print(slot_list_goal)
    return gpan

def count(data):
    intent_pool=[]
    slot_pool=[]
    goal_pool={}
    no_act_count=0
    for dial_id in data:
        goal=data[dial_id]['goal']
        for domain in goal:
            if domain in ['message','topic']:
                continue
            if domain not in goal_pool:
                goal_pool[domain]={}
            for intent in goal[domain]:
                if intent not in goal_pool[domain]:
                    goal_pool[domain][intent]=[]
                if isinstance(goal[domain][intent],dict):
                    for slot in goal[domain][intent].keys():
                        if slot not in goal_pool[domain][intent]:
                            goal_pool[domain][intent].append(slot)
                elif isinstance(goal[domain][intent],list):
                    for slot in goal[domain][intent]:
                        if slot not in goal_pool[domain][intent]:
                            goal_pool[domain][intent].append(slot)
        
        for turn_id,turn in enumerate(data[dial_id]['log']):
            if turn_id%2==0:#user
                if 'dialog_act' not in turn:
                    no_act_count+=1
                    continue
                usr_act=turn['dialog_act']
                for key in usr_act:
                    if key not in intent_pool:
                        intent_pool.append(key)
                    for item in usr_act[key]:
                        if item[0] not in slot_pool:
                            slot_pool.append(item[0])
        

    print(goal_pool)
    print(slot_pool)
    print(intent_pool)
    #print('user turns without dialog act:', no_act_count)

def prepare_us_data():
    path1='data/multi-woz-2.1-processed/data_for_damd_fix.json'
    path2='data/MultiWOZ_2.1/data.json'
    save_path='data/multi-woz-2.1-processed/data_for_us.json'
    data1=json.load(open(path1,'r', encoding='utf-8'))
    data2=json.load(open(path2,'r', encoding='utf-8'))
    #count(data2)
    slot_list_goal=[]
    new_data={}
    for dial_id in data1:
        dial_id_up=dial_id.split('.')[0].upper()+'.json'
        dial1=data1[dial_id]
        dial2=data2[dial_id_up]
        new_data[dial_id]={}
        goal=dial1['goal']
        goal=reader.goal_norm(goal)
        new_data[dial_id]=[]
        pv_user_act=None
        pv_constraint=None
        pv_sys_act=None
        for turn_id, turn in enumerate(dial1['log']):
            if pv_user_act is not None:
                #goal=reader.update_goal(goal, pv_user_act, pv_constraint)
                goal=reader.update_goal(goal, pv_user_act, sys_act=pv_sys_act)
            turn_domain=turn['turn_domain'].split()
            cur_domain=turn_domain[0] if len(turn_domain)==1 else turn_domain[1]
            cur_domain=cur_domain[1:-1] if cur_domain.startswith('[') else cur_domain
            gpan=reader.goal_to_gpan(goal, cur_domain)
            entry={}
            entry['goal']=gpan
            for field in ['user','resp','constraint','sys_act','turn_domain', 'turn_num']:
                entry[field]=turn[field]
            if 'dialog_act' in dial2['log'][2*turn_id]:
                entry['usr_act']=act_dict_to_aspn(dial2['log'][2*turn_id]['dialog_act'])
            else:
                entry['usr_act']=''
            pv_user_act=reader.aspan_to_act_dict(entry['usr_act'], side='user')
            pv_constraint=reader.bspan_to_constraint_dict(entry['constraint'])
            pv_sys_act=reader.aspan_to_act_dict(entry['sys_act'], side='sys')
            new_data[dial_id].append(entry)

    json.dump(new_data, open(save_path, 'w'), indent=2)

def temp():
    path='data/multi-woz-2.1-processed/data_for_us.json'
    path1='data/multi-woz-2.1-processed/data_for_damd_fix.json'
    data=json.load(open(path,'r', encoding='utf-8'))
    data1=json.load(open(path1,'r', encoding='utf-8'))
    new_data={}
    for dial_id in data:
        new_data[dial_id]={}
        new_data[dial_id]['goal']=data1[dial_id]['goal']
        new_data[dial_id]['log']=[]
        for turn in data[dial_id]:
            turn['gpan']= '<sos_g> '+ turn.pop('goal')+' <eos_g>'
            turn['user']='<sos_u> ' +turn['user']+' <eos_u>'
            turn['usr_act']='<sos_ua> ' +turn['usr_act']+' <eos_ua>'
            turn['bspn']='<sos_b> '+turn.pop('constraint')+' <eos_b>'
            turn['aspn']='<sos_a> '+turn.pop('sys_act')+' <eos_a>'
            turn['db']='<sos_db> '+reader.bspan_to_DBpointer(turn['bspn'], turn['turn_domain'].split())+' <eos_db>'
            turn['resp']='<sos_r> ' +turn['resp']+' <eos_r>'
            new_data[dial_id]['log'].append(turn)
    json.dump(new_data, open('data/multi-woz-2.1-processed/data_for_rl.json', 'w'), indent=2)

def prepare_modular_data():
    path='data/multi-woz-2.1-processed/data_for_rl.json'
    data=json.load(open(path,'r', encoding='utf-8'))
    dst_data={'train':[], 'dev':[], 'test':[]}
    dm_data={'train':[], 'dev':[], 'test':[]}
    nlg_data={'train':[], 'dev':[], 'test':[]}
    for dial_id in data:
        if dial_id in reader.dev_list:
            set='dev'
        elif dial_id in reader.test_list:
            set='test'
        else:
            set='train'
        dial=data[dial_id]['log']
        for turn_id, turn in enumerate(dial):
            nlg_sample = [turn['aspn'], turn['resp']]
            if turn_id==0:
                dst_sample=[turn['user'], turn['bspn']]
                dm_sample=[turn['user']+turn['bspn']+turn['db'], turn['aspn']+turn['resp']]
                #dm_sample=[turn['user']+turn['db'], turn['aspn']]
            else:
                pv_turn=dial[turn_id-1]
                dst_sample=[pv_turn['bspn']+pv_turn['resp']+turn['user'], turn['bspn']]
                #dst_sample=[pv_turn['bspn']+turn['user'], turn['bspn']]
                dm_sample=[pv_turn['bspn']+pv_turn['resp']+turn['user']+turn['bspn']+turn['db'], turn['aspn']+turn['resp']]
                #dm_sample=[pv_turn['aspn']+turn['user']+turn['db'], turn['aspn']]
            dst_data[set].append(dst_sample)
            nlg_data[set].append(nlg_sample)
            dm_data[set].append(dm_sample)
    json.dump(dst_data,open('data/multi-woz-2.1-processed/data_for_dst.json', 'w'), indent=2)
    json.dump(dm_data,open('data/multi-woz-2.1-processed/data_for_dm.json', 'w'), indent=2)
    #json.dump(nlg_data,open('data/multi-woz-2.1-processed/data_for_nlg.json', 'w'), indent=2)


if __name__ == "__main__":
    #prepare_us_data()
    #temp()
    prepare_modular_data()
    
    
