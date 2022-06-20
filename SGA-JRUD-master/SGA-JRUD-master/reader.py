import numpy as np
import os
import csv
import random
import logging
import json
import spacy
import utils
import ontology
import torch
from copy import deepcopy
from collections import OrderedDict
from db_ops import MultiWozDB
from torch.utils.data import Dataset, DataLoader
import transformers
from config import global_config as cfg
#from config21 import global_config as cfg

class _ReaderBase(object):

    def __init__(self):
        self.train, self.dev, self.test = [], [], []
        self.vocab = None
        self.db = None
        self.set_stats = {}

    def _bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len==0:
                continue
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)
        del_l = []
        for k in turn_bucket:
            if k >= 5:
                del_l.append(k)
            logging.debug("bucket %d instance %d" % (k, len(turn_bucket[k])))
        # for k in del_l:
        #    turn_bucket.pop(k)
        return OrderedDict(sorted(turn_bucket.items(), key=lambda i: i[0]))

    def _construct_mini_batch(self, data):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == cfg.batch_size:
                # print('batch size: %d, batch num +1'%(len(batch)))
                all_batches.append(batch)
                batch = []
        # if remainder > 1/2 batch_size, just put them in the previous batch, otherwise form a new batch
        # print('last batch size: %d, batch num +1'%(len(batch)))
        if (len(batch) % len(cfg.cuda_device)) != 0:
            batch = batch[:-(len(batch) % len(cfg.cuda_device))]

        if len(batch)>0:
            all_batches.append(batch)
        '''
        if len(batch) > 0.5 * cfg.batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)
        '''
        return all_batches
    def transpose_batch(self, batch):
        dial_batch = []
        turn_num = len(batch[0])
        for turn in range(turn_num):
            turn_l = {}
            for dial in batch:
                this_turn = dial[turn]
                for k in this_turn:
                    if k not in turn_l:
                        turn_l[k] = []
                    turn_l[k].append(this_turn[k])
            dial_batch.append(turn_l)
        return dial_batch
        
    def split_turn_batch(self, turn_batch, batch_size, other_batch=None):
        batches=[]
        other_batches=[]
        B=len(turn_batch['user'])
        for i in range(0, B, batch_size):
            new_turn_batch={}
            if other_batch:
                other_batches.append(other_batch[i:i+batch_size])
            for key in turn_batch:
                new_turn_batch[key]=turn_batch[key][i:i+batch_size]
            batches.append(new_turn_batch)
        if other_batch:
            return batches, other_batches
        else:
            return batches, None


    def inverse_transpose_turn(self, turn_list):
        """
        eval, one dialog at a time
        """
        dialogs = {}
        turn_num = len(turn_list)
        dial_id = turn_list[0]['dial_id']
        dialogs[dial_id] = []
        for turn_idx in range(turn_num):
            dial_turn = {}
            turn = turn_list[turn_idx]
            for key, value in turn.items():
                if key=='dial_id':
                    continue
                if key == 'pointer' and self.db is not None:
                    turn_domain = turn['turn_domain'][-1]
                    value = self.db.pointerBack(value, turn_domain)
                dial_turn[key] = value
            dialogs[dial_id].append(dial_turn)
        return dialogs

    def inverse_transpose_batch(self, turn_batch_list):
        """
        :param turn_batch_list: list of transpose dial batch
        """
        dialogs = []
        total_turn_num = len(turn_batch_list)
        # initialize
        for idx_in_batch, dial_id in enumerate(turn_batch_list[0]['dial_id']):
            dialog = []
            for turn_n in range(total_turn_num):
                dial_turn = {}
                turn_batch = turn_batch_list[turn_n]
                for key, v_list in turn_batch.items():
                    if idx_in_batch>=len(v_list):
                        print('list out of range',key, v_list)
                        continue
                    value = v_list[idx_in_batch]
                    dial_turn[key] = value
                dialog.append(dial_turn)
            dialogs.append(dialog)
        return dialogs
    
    def inverse_transpose_batch0(self, turn_batch_list):
        """
        :param turn_batch_list: list of transpose dial batch
        """
        dialogs = {}
        total_turn_num = len(turn_batch_list)
        # initialize
        for idx_in_batch, dial_id in enumerate(turn_batch_list[0]['dial_id']):
            dialogs[dial_id] = []
            for turn_n in range(total_turn_num):
                dial_turn = {}
                turn_batch = turn_batch_list[turn_n]
                for key, v_list in turn_batch.items():
                    if key == 'dial_id':
                        continue
                    value = v_list[idx_in_batch]
                    if key == 'pointer' and self.db is not None:
                        turn_domain = turn_batch['turn_domain'][idx_in_batch][-1]
                        value = self.db.pointerBack(value, turn_domain)
                    dial_turn[key] = value
                dialogs[dial_id].append(dial_turn)
        return dialogs

    def get_eval_data(self, set_name='dev'):
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]

        if set_name not in self.set_stats:
            self.set_stats[set_name] = {}
        num_turns = 0
        num_dials = len(dial)
        for d in dial:
            num_turns += len(d)

        self.set_stats[set_name]['num_turns'] = num_turns
        self.set_stats[set_name]['num_dials'] = num_dials

        return dial
        
    def get_batches(self, set_name,data=None):
        """
        compute dataset stats.
        """
        global dia_count
        log_str = ''

        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]
        if data:
            dial=data
        if cfg.low_resource and set_name == 'train':
            # dial = random.sample(dial, int(len(dial)*0.01))
            dial = random.sample(dial, 100)
            logging.info('Low Resource setting, finetuning size: {}'.format(len(dial)))
        turn_bucket = self._bucket_by_turn(dial)
        # self._shuffle_turn_bucket(turn_bucket)
        all_batches = []

        
        if set_name not in self.set_stats:
            self.set_stats[set_name] = {}
        num_training_steps = 0
        num_turns = 0
        num_dials = 0

        for k in turn_bucket:
            if set_name != 'test' and k == 1 or k >= 17:
                continue
            batches = self._construct_mini_batch(turn_bucket[k])
            if len(batches)==0:
                continue
            log_str += "turn num:%d, dial num: %d, batch num: %d last batch len: %d\n" % (
                k, len(turn_bucket[k]), len(batches), len(batches[-1]))
            # print("turn num:%d, dial num:v%d, batch num: %d, "%(k, len(turn_bucket[k]), len(batches)))
            num_training_steps += k * len(batches)
            num_turns += k * len(turn_bucket[k])
            num_dials += len(turn_bucket[k])
            all_batches += batches
        log_str += 'total batch num: %d\n' % len(all_batches)
        # print('total batch num: %d'%len(all_batches))
        # print('dialog count: %d'%dia_count)
        # return all_batches

        # log stats
        # logging.info(log_str)
        # cfg.num_training_steps = num_training_steps * cfg.epoch_num
        self.set_stats[set_name]['num_training_steps_per_epoch'] = num_training_steps
        self.set_stats[set_name]['num_turns'] = num_turns
        self.set_stats[set_name]['num_dials'] = num_dials
        if data is None:
            if set_name == 'train':
                random.shuffle(all_batches)
        return all_batches
    
    def get_nontranspose_data_iterator(self, all_batches):
        for i, batch in enumerate(all_batches):
            yield batch

    def get_data_iterator(self, all_batches):
        for i, batch in enumerate(all_batches):
            yield self.transpose_batch(batch)

    def save_result(self, write_mode, results, field, write_title=False,result_name=None):
        field=list(results[0].keys())
        result_name=result_name if result_name is not None else 'result.csv'
        with open(os.path.join(cfg.eval_load_path,result_name), write_mode) as rf:
            if write_title:
                rf.write(write_title+'\n')
            try:
                writer = csv.DictWriter(rf, fieldnames=field)
                writer.writeheader()
                writer.writerows(results)
            except Exception as e:
                print(e)
        return None
    
    def load_result(self,result_path):
        results=[]
        with open(result_path, 'r') as rf:
            reader=csv.reader(rf)
            is_field=True
            for n,line in enumerate(reader):
                entry={}
                if n==0 and line=='DECODED RESULTS:':
                    continue
                if is_field:
                    field=line
                    is_field=False
                else:
                    for i,key in enumerate(field):
                        entry[key]=line[i]
                    results.append(entry)
        return results,field


    def save_result_report(self, results):
        # if 'joint_goal' in results[0]:
        #     with open(cfg.result_path[:-4] + '_report_dst.txt', 'w') as rf:
        #         rf.write('joint goal\tslot_acc\tslot_f1\tact_f1\n')
        #         for res in results:
        #             a,b,c,d = res['joint_goal'], res['slot_acc'], res['slot_f1'], res['act_f1']
        #             rf.write('%2.1f\t%2.1f\t%2.1f\t%2.1f\n'%(a,b,c,d))
        # elif 'joint_goal_delex' in results[0]:
        #     with open(cfg.result_path[:-4] + '_report_bsdx.txt', 'w') as rf:
        #         rf.write('joint goal\tslot_acc\tslot_f1\tact_f1\n')
        #         for res in results:
        #             a,b,c,d = res['joint_goal_delex'], res['slot_acc_delex'], res['slot_f1_delex'], res['act_f1']
        #             rf.write('%2.1f\t%2.1f\t%2.1f\t%2.1f\n'%(a,b,c,d))
        ctr_save_path = cfg.result_path[:-4] + '_report_ctr%s.csv' % cfg.seed
        write_title = False if os.path.exists(ctr_save_path) else True
        if cfg.aspn_decode_mode == 'greedy':
            setting = ''
        elif cfg.aspn_decode_mode == 'beam':
            setting = 'width=%s' % str(cfg.beam_width)
            if cfg.beam_diverse_param > 0:
                setting += ', penalty=%s' % str(cfg.beam_diverse_param)
        elif cfg.aspn_decode_mode == 'topk_sampling':
            setting = 'topk=%s' % str(cfg.topk_num)
        elif cfg.aspn_decode_mode == 'nucleur_sampling':
            setting = 'p=%s' % str(cfg.nucleur_p)
        res = {'exp': cfg.eval_load_path, 'true_bspn': cfg.use_true_curr_bspn, 'true_aspn': cfg.use_true_curr_aspn,
               'decode': cfg.aspn_decode_mode, 'param': setting, 'nbest': cfg.nbest, 'selection_sheme': cfg.act_selection_scheme,
               'match': results[0]['match'], 'success': results[0]['success'], 'bleu': results[0]['bleu'], 'act_f1': results[0]['act_f1'],
               'avg_act_num': results[0]['avg_act_num'], 'avg_diverse': results[0]['avg_diverse_score']}
        with open(ctr_save_path, 'a') as rf:
            writer = csv.DictWriter(rf, fieldnames=list(res.keys()))
            if write_title:
                writer.writeheader()
            writer.writerows([res])


class MultiWozReader(_ReaderBase):

    def __init__(self, tokenizer):
        super().__init__()
        self.nlp = spacy.load('en_core_web_sm')

        self.db = MultiWozDB(cfg.dbs)

        # self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.gpt_path) # add special tokens later
        self.tokenizer = tokenizer
        self.add_sepcial_tokens()

        self.domain_files = json.loads(open(cfg.domain_file_path, 'r').read())
        self.slot_value_set = json.loads(
            open(cfg.slot_value_set_path, 'r').read())
        if cfg.multi_acts_training:
            self.multi_acts = json.loads(open(cfg.multi_acts_path, 'r').read())

        test_list = [l.strip().lower()
                     for l in open(cfg.test_list, 'r').readlines()]
        dev_list = [l.strip().lower()
                    for l in open(cfg.dev_list, 'r').readlines()]
        self.test_list=test_list
        self.dev_list=dev_list
        self.dev_files, self.test_files = {}, {}
        for fn in test_list:
            self.test_files[fn.replace('.json', '')] = 1
        for fn in dev_list:
            self.dev_files[fn.replace('.json', '')] = 1

        # for domain expanse aka. Cross domain
        self.exp_files = {}
        all_domains_list = list(self.domain_files.keys())
        if 'all' not in cfg.exp_domains:
            domains = self.get_exp_domains(cfg.exp_domains, all_domains_list)
            logging.info(domains)
            for domain in domains:
                fn_list = self.domain_files.get(domain)
                if not fn_list:
                    raise ValueError(
                        '[%s] is an invalid experiment setting' % domain)
                for fn in fn_list:
                    self.exp_files[fn.replace('.json', '')] = 1
        #

        self._load_data()


        self.multi_acts_record = None
        self.get_special_ids()
        self.clean_dial_id()

    def get_exp_domains(self, exp_domains, all_domains_list):
        if 'hotel' in exp_domains:
            if 'except' in exp_domains:
                # ['except', 'hotel']
                domains = [d for d in all_domains_list if 'hotel' not in d and 'multi' not in d]
            else:
                # ['hotel']
                domains = ['hotel_single', 'hotel_multi']
        if 'train' in exp_domains:
            if 'except' in exp_domains:
                # ['except', 'train']
                domains = [d for d in all_domains_list if 'train' not in d and 'multi' not in d]
            else:
                # ['train']
                domains = ['train_single', 'train_multi']
        if 'attraction' in exp_domains:
            if 'except' in exp_domains:
                # ['except', 'attraction']
                domains = [d for d in all_domains_list if 'attraction' not in d and 'multi' not in d]
            else:
                # ['attraction']
                domains = ['attraction_single', 'attraction_multi']
        if 'restaurant' in exp_domains:
            if 'except' in exp_domains:
                # ['except', 'restaurant']
                domains = [d for d in all_domains_list if 'restaurant' not in d and 'multi' not in d]
            else:
                # ['restaurant']
                domains = ['restaurant_single', 'restaurant_multi']
        if 'taxi' in exp_domains:
            if 'except' in exp_domains:
                # ['except', 'taxi']
                domains = [d for d in all_domains_list if 'taxi' not in d and 'multi' not in d]
            else:
                # ['taxi']
                domains = ['taxi_single', 'taxi_multi']
        return domains

    def clean_dial_id(self):
        new_list=[]
        for dial_id in self.dev_list:
            if dial_id in self.data:
                new_list.append(dial_id)
        self.dev_list=new_list

    def add_sepcial_tokens(self):
        """
            add special tokens to gpt tokenizer
            serves a similar role of Vocab.construt()
            make a dict of special tokens
        """
        special_tokens = []
        for word in ontology.all_domains + ['general']:
            word = '[' + word + ']'
            special_tokens.append(word)
        for word in ontology.all_acts:
            word = '[' + word + ']'
            special_tokens.append(word)
        # for word in ontology.all_slots:
            # to be determine whether slot should be [slot]
            # if slot, tokenizer having trouble decoding.
            # special_tokens.append(word)
        vocab_special_tokens=["[value_name]", "[value_choice]", "[value_area]", "[value_price]",
         "[value_type]", "[value_reference]", "[value_phone]", "[value_address]","[value_food]",
         "[value_leave]", "[value_postcode]", "[value_id]", "[value_arrive]", "[value_stars]",
         "[value_day]", "[value_destination]", "[value_car]", "[value_departure]","[value_time]",
         "[value_people]", "[value_stay]", "[value_pricerange]", "[value_department]", "[value_name]([value_phone]"]
        '''
        for word in self.vocab._word2idx.keys():
            if word.startswith('[value_') and word.endswith(']'):
                special_tokens.append(word)
        '''
        for word in vocab_special_tokens:
            if word!='[value_choice]':
                special_tokens.append(word)
            else:
                if cfg.delex_as_damd:
                    special_tokens.append(word)
        special_tokens.extend(ontology.special_tokens)
        
        if cfg.train_us:
            #special_tokens.extend(['[book]','[fail_book]','[fail_info]','[pre_invalid]','[invalid]','<sos_g>','<eos_g>','<sos_ua>','<eos_ua>'])
            special_tokens.extend(['<sos_g>','<eos_g>','<sos_ua>','<eos_ua>'])
        

        special_tokens_dict = {'additional_special_tokens': special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        logging.info('Added special tokens to gpt tokenizer.')

        cfg.pad_id = self.tokenizer.encode('<pad>')[0]
    
    def get_special_ids(self):
        self.sos_b_id=self.tokenizer.convert_tokens_to_ids('<sos_b>')
        self.sos_a_id=self.tokenizer.convert_tokens_to_ids('<sos_a>')
        self.sos_r_id=self.tokenizer.convert_tokens_to_ids('<sos_r>')
        self.eos_b_id=self.tokenizer.convert_tokens_to_ids('<eos_b>')
        self.eos_a_id=self.tokenizer.convert_tokens_to_ids('<eos_a>')
        self.eos_r_id=self.tokenizer.convert_tokens_to_ids('<eos_r>')
        self.sos_db_id=self.tokenizer.convert_tokens_to_ids('<sos_db>')
        self.eos_db_id=self.tokenizer.convert_tokens_to_ids('<eos_db>')
        self.sos_u_id=self.tokenizer.convert_tokens_to_ids('<sos_u>')
        self.eos_u_id=self.tokenizer.convert_tokens_to_ids('<eos_u>')
        if cfg.train_us:
            self.sos_g_id=self.tokenizer.convert_tokens_to_ids('<sos_g>')
            self.eos_g_id=self.tokenizer.convert_tokens_to_ids('<eos_g>')
            self.sos_ua_id=self.tokenizer.convert_tokens_to_ids('<sos_ua>')
            self.eos_ua_id=self.tokenizer.convert_tokens_to_ids('<eos_ua>')


    def _construct_bspn_constraint(self):
        bspn_masks = {}
        valid_domains = ['restaurant', 'hotel',
                         'attraction', 'train', 'taxi', 'hospital']
        all_dom_codes = [self.vocab.encode('['+d+']') for d in valid_domains]
        all_slot_codes = [self.vocab.encode(s) for s in ontology.all_slots]
        bspn_masks[self.vocab.encode(
            '<go_b>')] = all_dom_codes + [self.vocab.encode('<eos_b>'), 0]
        bspn_masks[self.vocab.encode('<eos_b>')] = [self.vocab.encode('<pad>')]
        bspn_masks[self.vocab.encode('<pad>')] = [self.vocab.encode('<pad>')]
        for domain, slot_values in self.slot_value_set.items():
            if domain == 'police':
                continue
            dom_code = self.vocab.encode('['+domain+']')
            bspn_masks[dom_code] = []
            for slot, values in slot_values.items():
                slot_code = self.vocab.encode(slot)
                if slot_code not in bspn_masks:
                    bspn_masks[slot_code] = []
                if slot_code not in bspn_masks[dom_code]:
                    bspn_masks[dom_code].append(slot_code)
                for value in values:
                    for idx, v in enumerate(value.split()):
                        if not self.vocab.has_word(v):
                            continue
                        v_code = self.vocab.encode(v)
                        if v_code not in bspn_masks:
                            # print(self.vocab._word2idx)
                            bspn_masks[v_code] = []
                        if idx == 0 and v_code not in bspn_masks[slot_code]:
                            bspn_masks[slot_code].append(v_code)
                        if idx == (len(value.split()) - 1):
                            for w in all_dom_codes + all_slot_codes:
                                if self.vocab.encode('<eos_b>') not in bspn_masks[v_code]:
                                    bspn_masks[v_code].append(
                                        self.vocab.encode('<eos_b>'))
                                if w not in bspn_masks[v_code]:
                                    bspn_masks[v_code].append(w)
                            break
                        if not self.vocab.has_word(value.split()[idx + 1]):
                            continue
                        next_v_code = self.vocab.encode(value.split()[idx + 1])
                        if next_v_code not in bspn_masks[v_code]:
                            bspn_masks[v_code].append(next_v_code)
        bspn_masks[self.vocab.encode('<unk>')] = list(bspn_masks.keys())

        with open('data/multi-woz-processed/bspn_masks.txt', 'w') as f:
            for i, j in bspn_masks.items():
                f.write(self.vocab.decode(i) + ': ' +
                        ' '.join([self.vocab.decode(int(m)) for m in j])+'\n')
        return bspn_masks

    def _construct_aspn_constraint(self):
        aspn_masks = {}
        aspn_masks = {}
        all_dom_codes = [self.vocab.encode('['+d+']')
                         for d in ontology.dialog_acts.keys()]
        all_act_codes = [self.vocab.encode('['+a+']')
                         for a in ontology.dialog_act_params]
        all_slot_codes = [self.vocab.encode(s)
                          for s in ontology.dialog_act_all_slots]
        aspn_masks[self.vocab.encode(
            '<go_a>')] = all_dom_codes + [self.vocab.encode('<eos_a>'), 0]
        aspn_masks[self.vocab.encode('<eos_a>')] = [self.vocab.encode('<pad>')]
        aspn_masks[self.vocab.encode('<pad>')] = [self.vocab.encode('<pad>')]
        # for d in all_dom_codes:
        #     aspn_masks[d] = all_act_codes
        for a in all_act_codes:
            aspn_masks[a] = all_dom_codes + all_slot_codes + \
                [self.vocab.encode('<eos_a>')]
        for domain, acts in ontology.dialog_acts.items():
            dom_code = self.vocab.encode('['+domain+']')
            aspn_masks[dom_code] = []
            for a in acts:
                act_code = self.vocab.encode('['+a+']')
                if act_code not in aspn_masks[dom_code]:
                    aspn_masks[dom_code].append(act_code)
        # for a, slots in ontology.dialog_act_params.items():
        #     act_code = self.vocab.encode('['+a+']')
        #     slot_codes = [self.vocab.encode(s) for s in slots]
        #     aspn_masks[act_code] = all_dom_codes + slot_codes + [self.vocab.encode('<eos_a>')]
        for s in all_slot_codes:
            aspn_masks[s] = all_dom_codes + all_slot_codes + \
                [self.vocab.encode('<eos_a>')]
        aspn_masks[self.vocab.encode('<unk>')] = list(aspn_masks.keys())

        with open('data/multi-woz-processed/aspn_masks.txt', 'w') as f:
            for i, j in aspn_masks.items():
                f.write(self.vocab.decode(i) + ': ' +
                        ' '.join([self.vocab.decode(int(m)) for m in j])+'\n')
        return aspn_masks

    def _load_data(self, save_temp=True):
        """
        load processed data and encode, or load already encoded data
        """
        if save_temp: # save encoded data
            if 'all' in cfg.exp_domains:
                #data.json is the encoded file of standard delexicalized data
                #data2.json is obtained by changing '[value_choice]' to corresponding db pointer
                #different data file needs different vocab file
                if cfg.train_us:
                    encoded_file = os.path.join(cfg.data_path, 'encoded_us_data.json')
                else:
                    if cfg.delex_as_damd:
                        encoded_file = os.path.join(cfg.data_path, 'new_db_se_blank_encoded.data.json')
                    else:
                        encoded_file = os.path.join(cfg.data_path, 'new_db_se_blank_encoded.data2.json')
                    if cfg.fix_data:
                        encoded_file=encoded_file[:-5]+'_fix.json'
                # encoded: no sos, se_encoded: sos and eos
                # db: add db results every turn
            else:
                xdomain_dir = './experiments_Xdomain/data'
                if not os.path.exists(xdomain_dir):
                    os.makedirs(xdomain_dir)
                encoded_file = os.path.join(xdomain_dir, '{}-encoded.data.json'.format('-'.join(cfg.exp_domains))) 
            
            self.data = json.loads(open(cfg.data_path + cfg.data_file, 'r', encoding='utf-8').read().lower())
            self.train_list=[]
            for key in self.data:
                if key not in self.dev_list and key not in self.test_list:
                    self.train_list.append(key)

            if cfg.rl_train:
                data_path='data/multi-woz-2.1-processed/data_for_rl.json'
                self.data=json.loads(open(data_path, 'r', encoding='utf-8').read().lower())
                logging.info('Reading data from {}'.format(data_path))
            if not cfg.rl_train or 'test' in cfg.mode:
                if os.path.exists(encoded_file):
                    logging.info('Reading encoded data from {}'.format(encoded_file))
                    encoded_data = json.loads(open(encoded_file, 'r', encoding='utf-8').read())
                    self.train = encoded_data['train']
                    self.dev = encoded_data['dev']
                    self.test = encoded_data['test']
                else:
                    logging.info('Encoding data now and save the encoded data in {}'.format(encoded_file))
                    # not exists, encode data and save
                    data_path=cfg.data_path+cfg.data_file
                    self.data = json.loads(open(data_path, 'r', encoding='utf-8').read().lower())
                    if cfg.fix_data and not cfg.train_us:
                        self.data=self.fix_dialog_state(self.data)
                        data_path=data_path[:-5]+'_fix.json'
                        json.dump(self.data, open(data_path, 'w'), indent=2)
                    self.train, self.dev, self.test = [], [], []
                    for fn, dial in self.data.items():
                        if '.json' in fn:
                            fn = fn.replace('.json', '')
                        if 'all' in cfg.exp_domains or self.exp_files.get(fn):
                            if self.dev_files.get(fn):
                                self.dev.append(self._get_encoded_data(fn, dial))
                            elif self.test_files.get(fn):
                                self.test.append(self._get_encoded_data(fn, dial))
                            else:
                                self.train.append(self._get_encoded_data(fn, dial))
                    
                    # save encoded data
                    encoded_data = {'train': self.train, 'dev': self.dev, 'test': self.test}
                    json.dump(encoded_data, open(encoded_file, 'w'), indent=2)
                    logging.info('encoded file saved in %s'%encoded_file)

        else: # directly read processed data and encode
            self.train, self.dev, self.test = [], [], []
            for fn, dial in self.data.items():
                if '.json' in fn:
                    fn = fn.replace('.json', '')
                if 'all' in cfg.exp_domains or self.exp_files.get(fn):
                    if self.dev_files.get(fn):
                        self.dev.append(self._get_encoded_data(fn, dial))
                    elif self.test_files.get(fn):
                        self.test.append(self._get_encoded_data(fn, dial))
                    else:
                        self.train.append(self._get_encoded_data(fn, dial))

        random.shuffle(self.train)
        # random.shuffle(self.dev)
        # random.shuffle(self.test)
        logging.info('train size:{}, dev size:{}, test size:{}'.format(len(self.train), len(self.dev), len(self.test)))

    def fix_dialog_state(self, data):
        count=0
        for dial_id in data:
            dial=data[dial_id]['log']
            for turn_id, turn in enumerate(dial):
                cons=turn['constraint']
                if 'name' in cons:
                    cons_dict=self.bspan_to_constraint_dict(cons)
                    for domain in cons_dict:
                        name_value=cons_dict[domain].get('name', None)
                        if name_value and name_value not in turn['user']:# not in the current turn
                            name_in_user=False
                            for i in range(turn_id):
                                if name_value in dial[i]['user']:# in previous turns
                                    name_in_user=True
                                    break
                            if not name_in_user:
                                count+=1
                                cons_dict[domain].pop('name')
                    turn['constraint']=self.cons_dict_to_bspn(cons_dict)
        #print(count)
        return data

    def cons_dict_to_bspn(self, cons_dict):
        bs_list=[]
        for domain in cons_dict:
            bs_list.append('['+domain+']')
            for slot in cons_dict[domain]:
                bs_list.append(slot)
                bs_list.append(cons_dict[domain][slot])
        return ' '.join(bs_list)

    def _get_encoded_data(self, fn, dial):
        if cfg.train_us:
            return self._get_encoded_us_data(fn,dial)
        encoded_dial = []
        for idx, t in enumerate(dial['log']):  # tokenize to list of ids
            enc = {}
            enc['dial_id'] = fn
            enc['user'] = self.modified_encode( '<sos_u> ' +t['user'] + ' <eos_u>')
            #enc['usdx'] = self.modified_encode('<sos_u> ' + t['user'] + ' <eos_u>')
            #if cfg.delex_as_damd:
            enc['bspn'] = self.modified_encode('<sos_b> ' +t['constraint'] + ' <eos_b>')
            #enc['bsdx'] = self.modified_encode('<sos_b> ' +t['cons_delex'] + ' <eos_b>')
            enc['resp'] = self.modified_encode('<sos_r> ' +t['resp'] + ' <eos_r>')
            enc['aspn'] = self.modified_encode('<sos_a> ' +t['sys_act'] + ' <eos_a>')
            enc['dspn'] = self.modified_encode('<sos_d> ' +t['turn_domain'] + ' <eos_d>')
            enc['pointer'] = [int(i) for i in t['pointer'].split(',')]
            enc['turn_domain'] = t['turn_domain'].split()
            enc['turn_num'] = t['turn_num']
            if cfg.multi_acts_training:
                enc['aspn_aug'] = []
                if fn in self.multi_acts:
                    turn_ma = self.multi_acts[fn].get(str(idx), {})
                    for act_type, act_spans in turn_ma.items():
                        enc['aspn_aug'].append([self.tokenizer.encode(
                            a.split()+['<eos_a>']) for a in act_spans])

            # add db results to enc, at every turn
            db_pointer = self.bspan_to_DBpointer(t['constraint'], t['turn_domain'].split())
            if not cfg.delex_as_damd:
                if '[value_choice]' in t['resp']:
                    t['resp']=t['resp'].replace('[value_choice]', db_pointer)
                    enc['resp'] = self.modified_encode('<sos_r> ' +t['resp'] + ' <eos_r>')
            # db_tokens = ['<sos_db>', '<eos_db>', '[db_nores]', '[db_0]', '[db_1]', '[db_2]', '[db_3]']
            enc['db'] = self.modified_encode('<sos_db> ' +db_pointer + ' <eos_db>')

            encoded_dial.append(enc)
        return encoded_dial
    

    def _get_encoded_us_data(self, fn, dial):
        encoded_dial=[]
        for idx, t in enumerate(dial):  # tokenize to list of ids
            enc = {}
            enc['dial_id'] = fn
            enc['goal']=self.modified_encode('<sos_g> ' +t['goal']+' <eos_g>')
            enc['user'] = self.modified_encode( '<sos_u> ' +t['user'] + ' <eos_u>')
            enc['resp'] = self.modified_encode('<sos_r> ' +t['resp'] + ' <eos_r>')
            enc['sys_act'] = self.modified_encode('<sos_a> ' +t['sys_act'] + ' <eos_a>')
            enc['usr_act'] = self.modified_encode('<sos_ua> ' +t['usr_act'] + ' <eos_ua>')
            enc['turn_domain'] = t['turn_domain'].split()
            enc['turn_num'] = t['turn_num']

            encoded_dial.append(enc)
        
        return encoded_dial

    def encode_data(self, data, tokenizer, modular='dst'):
        encoded_data={}
        for set in data:
            encoded_data[set]=[]
            for item in data[set]:
                encoded_data[set].append([self.modified_encode(item[0], tokenizer), self.modified_encode(item[1], tokenizer)])
        return encoded_data

    def update_goal(self, init_goal, user_act, constraint=None, sys_act=None):
        # constraint and sys_act are from last turn
        goal=deepcopy(init_goal)
        for domain in user_act:
            if domain not in goal:
                continue
            for intent, sv in user_act[domain].items():
                if intent=='inform':
                    for slot, value in sv.items():
                        # In user act, price can express both price and pricerange
                        if slot=='price' and intent in goal[domain] and slot not in goal[domain][intent]:
                            slot='pricerange'
                        if  'inform' in goal[domain] and slot in goal[domain]['inform']:
                            if goal[domain]['inform'][slot]==value:
                                goal[domain]['inform'].pop(slot)
                            if goal[domain]['inform']=={}:
                                goal[domain].pop('inform')
                        elif 'book' in goal[domain] and slot in goal[domain]['book']:
                            if goal[domain]['book'][slot]==value:
                                goal[domain]['book'].pop(slot)
                            if goal[domain]['book']=={}:
                                goal[domain].pop('book')
                elif intent=='request':
                    for slot in sv:
                        if slot=='price' and intent in goal[domain] and slot not in goal[domain][intent]:
                            slot='pricerange'
                        if 'request' in goal[domain] and slot in goal[domain]['request']:
                            goal[domain]['request'].pop(goal[domain]['request'].index(slot))
                            if goal[domain]['request']==[]:
                                goal[domain].pop('request')
            if goal[domain]=={}:
                goal.pop(domain)
        if constraint:
            for domain, sv in constraint.items():
                if domain in goal:
                    for slot, value in sv.items():
                        if  'inform' in goal[domain] and slot in goal[domain]['inform']:
                            if goal[domain]['inform'][slot]==value:
                                goal[domain]['inform'].pop(slot)
                            if goal[domain]['inform']=={}:
                                goal[domain].pop('inform')
                        elif 'book' in goal[domain] and slot in goal[domain]['book']:
                            if goal[domain]['book'][slot]==value:
                                goal[domain]['book'].pop(slot)
                            if goal[domain]['book']=={}:
                                goal[domain].pop('book')
                    if goal[domain]=={}:
                        goal.pop(domain)
        if sys_act:
            for domain in sys_act:
                if domain not in goal:
                    continue
                for intent, slots in goal[domain].items():
                    # if system has inform the slot in last turn then user simulator needn't request
                    if (intent=='inform' or intent=='recommend') and 'request' in goal[domain]:
                        for slot in slots:
                            if slot in goal[domain]['request']:
                                goal[domain]['request'].pop(goal[domain]['request'].index(slot))
                        if goal[domain]['request']==[]:
                            goal[domain].pop('request')
                if goal[domain]=={}:
                    goal.pop(domain)
                
        return goal

    def goal_to_gpan(self, goal, cur_domain=None):
        if goal=={}:
            return ''
        domain_gpan=[]
        domain_idx=0
        cur_domain_idx=-1
        for domain in goal:
            if domain==cur_domain:
                cur_domain_idx=domain_idx
            domain_idx+=1
            goal_list=[]
            goal_list.append('['+domain+']')
            for intent in goal[domain]:
                goal_list.append('['+intent+']')
                if isinstance(goal[domain][intent],dict):
                    for s, v in goal[domain][intent].items():
                        goal_list.append(s)
                        goal_list.append(v)
                elif isinstance(goal[domain][intent],list):
                    for s in goal[domain][intent]:
                        goal_list.append(s)
            domain_gpan.append(' '.join(goal_list))
        # current domain must be the last
        if cur_domain!='general' and cur_domain_idx>=0:
            domain_gpan[cur_domain_idx], domain_gpan[-1] = domain_gpan[-1], domain_gpan[cur_domain_idx]
        return ' '.join(domain_gpan)

    def goal_norm(self, goal):
        new_goal={}
        for domain in goal:
            '''
            if 'book' in goal[domain]:
                if 'reqt' in goal[domain]:
                    if 'reference' not in goal[domain]['reqt']:
                        goal[domain]['reqt'].append('reference')
                else:
                    goal[domain]['reqt']=['reference']
            '''
            new_goal[domain]={}
            for intent in goal[domain]:
                if intent in ['fail_book','fail_info']:
                    continue
                elif intent in ['info', 'book']:
                    new_intent='inform' if intent=='info' else intent
                    new_goal[domain][new_intent]={}
                    for slot, value in goal[domain][intent].items():
                        slot=slot.lower()
                        if slot in ['pre_invalid','invalid']:
                            continue
                        else:
                            if slot=='trainid':
                                slot='id'
                            elif slot=='car type':
                                slot='car'
                            elif slot in ['entrance fee', 'fee']:
                                slot='price'
                            elif slot=='duration':
                                slot='time'
                            elif slot=='arriveby':
                                slot='arrive'
                            elif slot=='leaveat':
                                slot='leave'
                        new_goal[domain][new_intent][slot]=value.lower()
                elif intent=='reqt':
                    new_goal[domain]['request']=[]
                    for slot in goal[domain][intent]:
                        new_goal[domain]['request'].append(slot.lower())
        
        return new_goal

    def bspan_to_constraint_dict(self, bspan, bspn_mode='bspn'):
        bspan = bspan.split() if isinstance(bspan, str) else bspan
        constraint_dict = {}
        domain = None
        conslen = len(bspan)
        for idx, cons in enumerate(bspan):
            cons = self.vocab.decode(cons) if type(cons) is not str else cons
            if cons == '<eos_b>':
                break
            if '[' in cons:
                if cons[1:-1] not in ontology.all_domains:
                    continue
                domain = cons[1:-1]
            elif cons in ontology.get_slot:
                if domain is None:
                    continue
                if cons == 'people':
                    # handle confusion of value name "people's portraits..." and slot people
                    try:
                        ns = bspan[idx+1]
                        ns = self.vocab.decode(ns) if type(
                            ns) is not str else ns
                        if ns == "'s":
                            continue
                    except:
                        continue
                if not constraint_dict.get(domain):
                    constraint_dict[domain] = {}
                if bspn_mode == 'bsdx':
                    constraint_dict[domain][cons] = 1
                    continue
                vidx = idx+1
                if vidx == conslen:
                    break
                vt_collect = []
                vt = bspan[vidx]
                vt = self.vocab.decode(vt) if type(vt) is not str else vt
                while vidx < conslen and vt != '<eos_b>' and '[' not in vt and vt not in ontology.get_slot:
                    vt_collect.append(vt)
                    vidx += 1
                    if vidx == conslen:
                        break
                    vt = bspan[vidx]
                    vt = self.vocab.decode(vt) if type(vt) is not str else vt
                if vt_collect:
                    constraint_dict[domain][cons] = ' '.join(vt_collect)

        return constraint_dict

    def bspan_to_DBpointer(self, bspan, turn_domain):
        constraint_dict = self.bspan_to_constraint_dict(bspan)
        # print(constraint_dict)
        matnums = self.db.get_match_num(constraint_dict)
        match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
        match_dom = match_dom[1:-1] if match_dom.startswith('[') else match_dom
        match = matnums[match_dom]
        # vector = self.db.addDBPointer(match_dom, match)
        vector = self.db.addDBIndicator(match_dom, match)
        return vector
    
    def aspan_to_act_list(self, aspan):
        aspan = aspan.split() if isinstance(aspan, str) else aspan
        acts = []
        domain = None
        conslen = len(aspan)
        for idx, cons in enumerate(aspan):
            cons = self.vocab.decode(cons) if type(cons) is not str else cons
            if cons in ['<eos_a>', '<eos_ua>']:
                break
            if '[' in cons and cons[1:-1] in ontology.dialog_acts:
                domain = cons[1:-1]

            elif '[' in cons and cons[1:-1] in ontology.dialog_act_params:
                if domain is None:
                    continue
                vidx = idx+1
                if vidx == conslen:
                    acts.append(domain+'-'+cons[1:-1]+'-none')
                    break
                vt = aspan[vidx]
                vt = self.vocab.decode(vt) if type(vt) is not str else vt
                no_param_act = True
                while vidx < conslen and vt not in ['<eos_a>', '<eos_ua>'] and '[' not in vt:
                    no_param_act = False
                    acts.append(domain+'-'+cons[1:-1]+'-'+vt)
                    vidx += 1
                    if vidx == conslen:
                        break
                    vt = aspan[vidx]
                    vt = self.vocab.decode(vt) if type(vt) is not str else vt
                if no_param_act:
                    acts.append(domain+'-'+cons[1:-1]+'-none')
        return acts

    def aspan_to_act_dict(self, aspan, side='sys'):
        assert side in ['sys', 'user'] # sys act or user act
        act_list=self.aspan_to_act_list(aspan)
        act_dict={}
        pv_slot=''
        if side=='sys':
            for act in act_list:
                if act.count('-')!=2:
                    continue
                domain, intent, slot = act.split('-')
                if domain not in act_dict:
                    act_dict[domain]={}
                if intent not in act_dict[domain]:
                    act_dict[domain][intent]=[]
                if slot not in act_dict[domain][intent]:
                    act_dict[domain][intent].append(slot)
        else:
            for act in act_list:
                if act.count('-')!=2:
                    continue
                domain, intent, slot = act.split('-')
                if domain not in act_dict:
                    act_dict[domain]={}
                if intent not in act_dict[domain]:
                    if intent=='inform':
                        act_dict[domain][intent]={}
                    elif intent=='request':
                        act_dict[domain][intent]=[]
                if intent=='inform':
                    if slot in ontology.all_slots:
                        act_dict[domain][intent][slot]='' 
                        pv_slot=slot                    
                    else:# slot is in fact a value in this condition
                        if pv_slot not in act_dict[domain][intent]:
                            continue
                        if act_dict[domain][intent][pv_slot]=='':
                            act_dict[domain][intent][pv_slot]=slot
                        else:
                            act_dict[domain][intent][pv_slot]+= ' '+slot
                elif intent=='request':
                    if slot not in act_dict[domain][intent]:
                        act_dict[domain][intent].append(slot)
        return act_dict

    def act_dict_to_aspan(self, act_dict):
        aspn=[]
        for domain in act_dict:
            aspn.append('['+domain+']')
            for intent in act_dict[domain]:
                aspn.append('['+intent+']')
                slot_list=act_dict[domain][intent]
                for slot in slot_list:
                    if slot!='none':
                        aspn.append(slot)
        return ' '.join(aspn)


    def dspan_to_domain(self, dspan):
        domains = {}
        dspan = dspan.split() if isinstance(dspan, str) else dspan
        for d in dspan:
            dom = self.vocab.decode(d) if type(d) is not str else d
            if dom != '<eos_d>':
                domains[dom] = 1
            else:
                break
        return domains

    def get_sys_batch(self, data, batch_size=16, mode='train'):
        assert mode in ['train', 'test']
        batches=[]
        batch=[]
        seq_num=0
        for dial in data:
            for turn in dial:
                if mode=='train':
                    batch.append(turn['resp'])
                elif mode=='test':
                    batch.append(turn['resp_gen'])
                if len(batch)>=batch_size:
                    seq_num+=len(batch)
                    batch_np, _ = utils.padSeqs_gpt(batch, cfg.pad_id)
                    batches.append(batch_np)
                    batch=[]
        if batch!=[]:
            seq_num+=len(batch)
            batch_np, _ = utils.padSeqs_gpt(batch, cfg.pad_id)
            batches.append(batch_np)
            batch=[]
        logging.info('Total responses:{}'.format(seq_num))
        return batches, seq_num


    def bs_filter(self,data,is_batch=True):
        #将生成的 bs 进行过滤
        special_tokens=['<eos_r>','<eos_a>','<eos_u>','<sos_r>','<sos_a>','<sos_u>','<sos_db>','<eos_db>']
        special_ids=self.tokenizer.convert_tokens_to_ids(special_tokens)
        eos_b_idx=self.tokenizer.convert_tokens_to_ids('<eos_b>')
        sos_b_idx=self.tokenizer.convert_tokens_to_ids('<sos_b>')
        total_turn=0
        count1=0#没有eos_b的数量
        count2=0#没有eos_b但含有其他特殊符的数量
        count3=0
        #logging.info('Starting filtering generated bspn')
        if is_batch:
            all_data=[]
            for dial_batch in data:
                all_data+=dial_batch
        else:
            all_data=data

        for dial in all_data:
            for turn in dial:
                total_turn+=1
                bspn=turn['bspn']
                if eos_b_idx not in bspn:
                    count1+=1
                    #说明生成时直接生成了50个token。如果中间有上述的special token，之后的信息就直接舍弃
                    loc=[]
                    for idx in special_ids:
                        if idx in bspn:
                            loc.append(bspn.index(idx))
                    if loc==[]:
                        bspn[-1]=eos_b_idx
                    else:
                        count2+=1
                        bspn=bspn[:min(loc)+1]
                        bspn[-1]=eos_b_idx
                else:
                    if bspn.count(sos_b_idx)>1:
                        last=bspn[::-1].index(sos_b_idx)+1
                        bspn=bspn[-last:]
                    if bspn[-1]!=eos_b_idx:
                        bspn=bspn[:bspn.index(eos_b_idx)+1]
                        count3+=1
                turn['bspn']=bspn#一定要再次赋值

        return all_data,(total_turn,count1,count2,count3)

    def convert_batch_tokens_to_ids(self, dial_batch, tokenizer):
        new_batch=[]
        for dial in dial_batch:
            if isinstance(dial,list):
                new_dial=[]
                for turn in dial:
                    new_turn={}
                    for key in turn:
                        if key in ['user','bspn','aspn','resp','db', 'usr_act', 'bspn_gen', 'aspn_gen', 
                            'resp_gen', 'db_gen', 'user_gen', 'usr_act_gen', 'gpan', 'pv_aspn']:
                            # GPT2Tokenizer of transformers3.5 needs to be modified
                            new_turn[key]=self.modified_encode(turn[key], tokenizer)
                        else:
                            new_turn[key]=turn[key]
                    new_dial.append(new_turn)
                new_batch.append(new_dial)
            elif isinstance(dial,dict):
                new_dial={}
                new_dial['goal']=self.modified_encode(dial['goal'], tokenizer)
                new_dial['log']=[]
                for turn in dial['log']:
                    new_turn={}
                    for key in turn:
                        if key in ['user','usdx','bspn','aspn','resp','bsdx','dspn','db', 'usr_act','bspn_gen', 'aspn_gen', 'resp_gen']:
                            # GPT2Tokenizer of transformers3.5 needs to be modified
                            new_turn[key]=self.modified_encode(turn[key], tokenizer)
                        else:
                            new_turn[key]=turn[key]
                    new_dial['log'].append(new_turn)
                new_batch.append(new_dial)
        return new_batch

    def convert_batch_ids_to_tokens(self, dial_batch):
        new_batch=[]
        for dial in dial_batch:
            new_dial=[]
            for turn in dial:
                new_turn={}
                for key in turn:
                    if key in ['user','bspn','aspn','resp','db', 'usr_act', 'goal', 'bspn_gen', 
                        'aspn_gen', 'resp_gen', 'db_gen','user_gen', 'usr_act_gen','sys_act']:
                        # GPT2Tokenizer of transformers3.5 needs to be modified
                        new_turn[key]=self.tokenizer.decode(turn[key])
                    else:
                        new_turn[key]=turn[key]
                new_dial.append(new_turn)
            new_batch.append(new_dial)
        return new_batch

    def transpose_ds_turn_batch(self, batch, rewards):
        turn_batches=[]
        label_batches=[]
        reward_batches=[]
        turn_batch=[]
        label_batch=[]
        reward_batch=[]
        if cfg.transpose_batch:
            p=0
            while(p*cfg.training_batch_size<len(batch)-1):
                batch_part=batch[p*cfg.training_batch_size:(p+1)*cfg.training_batch_size]
                reward_part=rewards[p*cfg.training_batch_size:(p+1)*cfg.training_batch_size]
                p+=1
                turn_id=0
                max_turn_num=max([len(dial) for dial in batch_part])
                for turn_id in range(max_turn_num):
                    if turn_id==0:
                        for dial, reward in zip(batch_part, reward_part):
                            if turn_id>len(dial)-1:
                                continue
                            turn=dial[turn_id]
                            R=reward[turn_id]
                            turn_batch.append(turn['user']+turn['bspn']+turn['db']+turn['aspn']+turn['resp'])
                            if cfg.rl_for_bspn:
                                label_batch.append([cfg.pad_id]*len(turn['user'])+turn['bspn']+turn['db']+\
                                    turn['aspn']+turn['resp'])
                            else:
                                label_batch.append([cfg.pad_id]*len(turn['user']+turn['bspn']+turn['db'])+\
                                    turn['aspn']+turn['resp'])
                            reward_batch.append(R)
                    else:
                        for dial, reward in zip(batch_part, reward_part):
                            if turn_id>len(dial)-1:
                                continue
                            turn=dial[turn_id]
                            pv_turn=dial[turn_id-1]
                            R=reward[turn_id]
                            turn_batch.append(pv_turn['bspn']+pv_turn['resp']+turn['user']+turn['bspn']+turn['db']+turn['aspn']+turn['resp'])
                            if cfg.rl_for_bspn:
                                label_batch.append([cfg.pad_id]*len(pv_turn['bspn']+pv_turn['resp']+\
                                    turn['user'])+turn['bspn']+turn['db']+turn['aspn']+turn['resp'])
                            else:
                                label_batch.append([cfg.pad_id]*len(pv_turn['bspn']+pv_turn['resp']+\
                                    turn['user']+turn['bspn']+turn['db'])+turn['aspn']+turn['resp'])
                            reward_batch.append(R)
                    
                    if len(turn_batch)>cfg.training_batch_size/2:
                        turn_batch_np, _ = utils.padSeqs_gpt(turn_batch, cfg.pad_id)
                        label_batch_np, _ = utils.padSeqs_gpt(label_batch, cfg.pad_id)
                        turn_batches.append(turn_batch_np)
                        label_batches.append(label_batch_np)
                        reward_batches.append(reward_batch)
                        turn_batch=[]
                        label_batch=[]
                        reward_batch=[]
    
        else:
            for dial, reward in zip(batch, rewards):
                pv_turn=None
                for turn, R in zip(dial, reward):
                    if pv_turn:
                        turn_batch.append(pv_turn['bspn']+pv_turn['resp']+\
                            turn['user']+turn['bspn']+turn['db']+turn['aspn']+turn['resp'])
                        if cfg.rl_for_bspn:
                            label_batch.append([cfg.pad_id]*len(pv_turn['bspn']+pv_turn['resp']+\
                                turn['user'])+turn['bspn']+turn['db']+turn['aspn']+turn['resp'])
                        else:
                            label_batch.append([cfg.pad_id]*len(pv_turn['bspn']+pv_turn['resp']+\
                                turn['user']+turn['bspn']+turn['db'])+turn['aspn']+turn['resp'])
                    else:
                        turn_batch.append(turn['user']+turn['bspn']+turn['db']+turn['aspn']+turn['resp'])
                        if cfg.rl_for_bspn:
                            label_batch.append([cfg.pad_id]*len(turn['user'])+turn['bspn']+turn['db']+\
                                turn['aspn']+turn['resp'])
                        else:
                            label_batch.append([cfg.pad_id]*len(turn['user']+turn['bspn']+turn['db'])+\
                                turn['aspn']+turn['resp'])
                    reward_batch.append(R)
                    if len(turn_batch)==cfg.training_batch_size:
                        turn_batch_np, _ = utils.padSeqs_gpt(turn_batch, cfg.pad_id)
                        label_batch_np, _ = utils.padSeqs_gpt(label_batch, cfg.pad_id)
                        turn_batches.append(turn_batch_np)
                        label_batches.append(label_batch_np)
                        reward_batches.append(reward_batch)
                        turn_batch=[]
                        label_batch=[]
                        reward_batch=[]
                    pv_turn=turn
            if turn_batch!=[]:
                turn_batch_np, _ = utils.padSeqs_gpt(turn_batch, cfg.pad_id)
                label_batch_np, _ = utils.padSeqs_gpt(label_batch, cfg.pad_id)
                turn_batches.append(turn_batch_np)
                label_batches.append(label_batch_np)
                reward_batches.append(reward_batch)
        return turn_batches, label_batches, reward_batches
    
    def transpose_us_turn_batch(self, batch, rewards, tokenizer):
        turn_batches=[]
        label_batches=[]
        reward_batches=[]
        turn_batch=[]
        label_batch=[]
        reward_batch=[]
        if cfg.transpose_batch:
            p=0
            while(p*cfg.training_batch_size<len(batch)-1):
                batch_part=batch[p*cfg.training_batch_size:(p+1)*cfg.training_batch_size]
                reward_part=rewards[p*cfg.training_batch_size:(p+1)*cfg.training_batch_size]
                p+=1
                turn_id=0
                max_turn_num=max([len(dial) for dial in batch_part])
                for turn_id in range(max_turn_num):
                    if turn_id==0:
                        for dial, reward in zip(batch_part, reward_part):
                            if turn_id>len(dial)-1:
                                continue
                            turn=dial[turn_id]
                            R=reward[turn_id]
                            turn_batch.append(turn['gpan']+turn['usr_act']+turn['user'])
                            label_batch.append([cfg.pad_id]*len(turn['gpan'])+turn['usr_act']+turn['user'])
                            reward_batch.append(R)
                    else:
                        for dial, reward in zip(batch_part, reward_part):
                            if turn_id>len(dial)-1:
                                continue
                            turn=dial[turn_id]
                            pv_turn=dial[turn_id-1]
                            R=reward[turn_id]
                            #pv_aspn=turn['pv_aspn'] if 'pv_aspn' in turn else pv_turn['aspn']
                            #turn_batch.append(pv_turn['resp']+pv_aspn+turn['gpan']+turn['usr_act']+turn['user'])
                            turn_batch.append(pv_turn['resp']+turn['gpan']+turn['usr_act']+turn['user'])
                            label_batch.append([cfg.pad_id]*len(pv_turn['resp']+turn['gpan'])+\
                                    turn['usr_act']+turn['user'])
                            reward_batch.append(R)
                    if len(turn_batch)>cfg.training_batch_size/2:
                        turn_batch_np, _ = utils.padSeqs_gpt(turn_batch, cfg.pad_id)
                        label_batch_np, _ = utils.padSeqs_gpt(label_batch, cfg.pad_id)
                        turn_batches.append(turn_batch_np)
                        label_batches.append(label_batch_np)
                        reward_batches.append(reward_batch)
                        turn_batch=[]
                        label_batch=[]
                        reward_batch=[]
        else:
            for dial, reward in zip(batch, rewards):
                pv_turn=None
                for turn, R in zip(dial, reward):
                    if pv_turn is None:
                        turn_batch.append(turn['gpan']+turn['usr_act']+turn['user'])
                        label_batch.append([cfg.pad_id]*len(turn['gpan'])+turn['usr_act']+turn['user'])
                    else:
                        #pv_aspn=turn['pv_aspn'] if 'pv_aspn' in turn else pv_turn['aspn']
                        turn_batch.append(pv_turn['resp']+turn['gpan']+turn['usr_act']+turn['user'])
                        label_batch.append([cfg.pad_id]*len(pv_turn['resp']+turn['gpan'])+\
                                turn['usr_act']+turn['user'])
                    reward_batch.append(R)
                    if len(turn_batch)==cfg.training_batch_size:
                        turn_batch_np, _ = utils.padSeqs_gpt(turn_batch, cfg.pad_id)
                        label_batch_np, _ = utils.padSeqs_gpt(label_batch, cfg.pad_id)
                        turn_batches.append(turn_batch_np)
                        label_batches.append(label_batch_np)
                        reward_batches.append(reward_batch)
                        turn_batch=[]
                        label_batch=[]
                        reward_batch=[]
                    pv_turn=turn
            if turn_batch!=[]:
                turn_batch_np, _ = utils.padSeqs_gpt(turn_batch, cfg.pad_id)
                label_batch_np, _ = utils.padSeqs_gpt(label_batch, cfg.pad_id)
                turn_batches.append(turn_batch_np)
                label_batches.append(label_batch_np)
                reward_batches.append(reward_batch)
        return turn_batches, label_batches, reward_batches

    def modified_encode(self, text, tokenizer=None):
        if tokenizer is None:
            tokenizer=self.tokenizer
        if int(transformers.__version__[0])>=3:
            if isinstance(text, str):
                word_list=text.split()
            elif isinstance(text, list):
                word_list=text
            else:             
                raise TypeError(text)
            special_token_pos=[]
            results=[]
            for idx, word in enumerate(word_list):
                if word in tokenizer.additional_special_tokens:
                    special_token_pos.append(idx)
            for j, idx in enumerate(special_token_pos):
                if j<len(special_token_pos)-1:
                    next_idx=special_token_pos[j+1]
                    results+=tokenizer.encode(word_list[idx]) + tokenizer.encode(' '+' '.join(word_list[idx+1:next_idx]))
                else:
                    results+=tokenizer.encode(word_list[idx])
                    if idx<len(word_list)-1:# the last word is not a special token
                        results+=tokenizer.encode(' '+' '.join(word_list[idx+1:]))
            return results

        else:
            return tokenizer.encode(text)

    def batch_align(self,contexts,left_len,return_attn=False):
        max_len=max([len(context) for context in contexts])
        max_len=min(1024-left_len,max_len)
        new_contexts=[]
        attentions=[]
        for id, context in enumerate(contexts):
            if len(context)<max_len:
                new_context=(max_len-len(context))*[cfg.pad_id]+context
                attention=(max_len-len(context))*[0]+len(context)*[1]
            else:
                new_context=context[-max_len:]
                attention=len(new_context)*[1]
            new_contexts.append(new_context)
            attentions.append(attention)
        if return_attn:
            return new_contexts, attentions
        return new_contexts

    def convert_batch_session(self, dial_batch,
        posterior_train=False, only_resp_label=False,bspn_label=False,bspn_pri=False,rl_train=False):
        """
        convert the whole session for training
        concat [U_0, B_0, A_0, R_0, ... , U_n, B_n, A_n, R_n]

        try: [user, bspn, aspn, resp]
        or
        try: [user, bspn, db, aspn, resp]
        """
        inputs = {}
        labels={}
        bspn_labels={}
        contexts = []
        label_contexts=[]
        bspn_label_contexts=[]
        if not posterior_train:
            if cfg.model_act:
                cell_list = ['user', 'bspn', 'db','aspn', 'resp']
                ignore_list= ['user','bspn','db','aspn'] if only_resp_label else ['user']
            else:
                cell_list = ['user', 'bspn', 'db', 'resp']
                ignore_list=['user','bspn','db'] if only_resp_label else ['user','db']
            
        else:
            if cfg.model_act:
                cell_list=['user','resp','bspn','db','aspn']
                ignore_list=['user','resp']
            else:
                cell_list=['user','resp','bspn']
                ignore_list=['user','resp']

        if rl_train:
            cell_list=['user', 'bspn', 'db','aspn', 'resp']
            if cfg.turn_level_reward and not cfg.rl_with_us:#only calculate cross-entropy on response:
                ignor_list=['user','bspn','db','aspn']
            else:
                ignore_list=['user'] if cfg.rl_for_bspn else ['user','bspn','db']
        
        for idx, dial in enumerate(dial_batch):
            context = []
            label_context=[]
            bspn_label_context=[]
            Dial=dial['log'] if isinstance(dial, dict) else dial
    
            for turn_num, turn in enumerate(Dial):
                for cell in cell_list:
                    if cell=='bspn' and bspn_pri and 'bspn_pri' in turn:
                        cell='bspn_pri'
          
                    if cell=='db':
                        if bspn_pri and 'db_pri' in turn:
                            cell='db_pri'
                        else:
                            db_result=self.bspan_to_DBpointer(self.tokenizer.decode(turn['bspn']), turn['turn_domain'])
                            turn[cell] = self.tokenizer.encode('<sos_db> '+ db_result + ' <eos_db>')
                    context.extend(turn[cell])
                    if cell in ignore_list:
                        label_context.extend(len(turn[cell])*[cfg.pad_id])#pad_id在计算损失时被自动忽略
                    else:
                        label_context.extend(turn[cell])
                    if bspn_label:
                        bspn_cell_list=['bspn','db','aspn'] if cfg.model_act else ['bspn']
                        if cell in bspn_cell_list:
                            bspn_label_context.extend(turn[cell])
                        else:
                            bspn_label_context.extend(len(turn[cell])*[cfg.pad_id])
            
            contexts.append(context)
            label_contexts.append(label_context)
            bspn_label_contexts.append(bspn_label_context)

        
        inputs['contexts'] = contexts
        inputs['contexts_np'], inputs['lengths'] = utils.padSeqs_gpt(inputs['contexts'], cfg.pad_id)
        labels['contexts']=label_contexts
        labels['contexts_np'], labels['lengths']=utils.padSeqs_gpt(labels['contexts'], cfg.pad_id)

        if not bspn_label:
            return inputs,labels
        else:
            bspn_labels['contexts']=bspn_label_contexts
            bspn_labels['contexts_np'],bspn_labels['lengths']=utils.padSeqs_gpt(bspn_labels['contexts'], cfg.pad_id)
            return inputs,labels,bspn_labels

    def get_pv_batch(self, pv_batch, user=None, resp=None, bspn=None, aspn=None, side='sys'):
        assert side in ['sys', 'user']
        new_pv_batch=[] # pv_batch for next turn
        if side=='sys':
            if pv_batch is None:# first turn
                for u, r, b in zip(user, resp, bspn): 
                    if cfg.input_history:
                        new_pv_batch.append(u+r)
                    elif cfg.input_prev_resp:
                        new_pv_batch.append(b+r)
                    else:
                        new_pv_batch.append(b)
            else:
                for hist, u, r, b in zip(pv_batch,user, resp, bspn):
                    if cfg.input_history:
                        new_pv_batch.append(hist+u+r)
                    elif cfg.input_prev_resp:
                        new_pv_batch.append(b+r)
                    else:
                        new_pv_batch.append(b)
        else:# user's pv batch
            for r, a in zip(resp, aspn):
                new_pv_batch.append(r)
        return new_pv_batch


    def convert_batch_turn(self, 
        turn_batch, 
        pv_batch, 
        first_turn=False, 
        rl_train=False, 
        mode='oracle', 
        side='sys', 
        posterior=False,
        seg_label=False
        ):
        '''
        Args:
        Returns:
        '''
        assert mode in ['oracle', 'gen']
        assert side in ['sys', 'user']
        inputs = {}
        labels = {}
        contexts = []
        label_contexts = []
        if rl_train:
            rl_labels={}
            rl_label_contexts=[]
        if seg_label:
            seg_labels={}
            seg_contexts=[]
        if side=='sys':
            if first_turn:
                if mode=='oracle':
                    batch_zipped = zip(turn_batch['user'], turn_batch['bspn'], 
                        turn_batch['db'], turn_batch['aspn'], turn_batch['resp'])
                else:
                    batch_zipped=zip(turn_batch['user'], turn_batch['bspn_gen'], 
                        turn_batch['db_gen'], turn_batch['aspn_gen'], turn_batch['resp_gen'])
                    
                for u, b, db, a, r in batch_zipped:
                    if posterior:
                        context=u+r + b+db+a
                        label_context=len(u+r)*[cfg.pad_id] + b+db+a
                    else:
                        context = u+b+db+a+r
                        label_context=len(u)*[cfg.pad_id]+b+db+a+r
                    contexts.append(context)
                    label_contexts.append(label_context)
                    if rl_train:
                        # 1 for belief state, 2 for system act, 3 for response and 0 for others
                        rl_label_context=len(u)*[0]+len(b)*[1]+len(db)*[0]+len(a)*[2]+len(r)*[3]
                        rl_label_contexts.append(rl_label_context)
                    if seg_label:
                        # 1 for hidden state, 2 for response, 0 for others
                        seg_context=len(u)*[0] + len(b+db+a)*[1] + len(r)*[2]
                        seg_contexts.append(seg_context)
            else:
                if mode=='oracle':
                    batch_zipped = zip(pv_batch,turn_batch['user'], turn_batch['bspn'], 
                        turn_batch['db'], turn_batch['aspn'], turn_batch['resp'])
                else:
                    batch_zipped = zip(pv_batch,turn_batch['user'], turn_batch['bspn_gen'], 
                        turn_batch['db_gen'], turn_batch['aspn_gen'], turn_batch['resp_gen'])
                for ur, u, b, db, a, r in batch_zipped:
                    if posterior:
                        context = ur + u + r + b + db + a
                        label_context=len(ur+u+r)*[cfg.pad_id] + b+db+a
                    else:
                        context = ur + u + b + db + a + r
                        label_context=len(ur+u)*[cfg.pad_id]+b+db+a+r
                    contexts.append(context)
                    label_contexts.append(label_context)
                    if rl_train:
                        # 1 for belief state, 2 for system act, 3 for response and 0 for others
                        rl_label_context=len(ur+u)*[0]+len(b)*[1]+len(db)*[0]+len(a)*[2]+len(r)*[3]
                        rl_label_contexts.append(rl_label_context)
                    if seg_label:
                        # 1 for hidden state, 2 for response, 0 for others
                        seg_context=len(ur+u)*[0] + len(b+db+a)*[1] + len(r)*[2]
                        seg_contexts.append(seg_context)
        
        elif side=='user':
            if first_turn:
                if mode=='oracle':
                    batch_zipped = zip(turn_batch['goal'], turn_batch['usr_act'], turn_batch['user'])
                else:
                    batch_zipped=zip(turn_batch['goal'], turn_batch['usr_act_gen'], turn_batch['user_gen'])                   
                for g, ua, u in batch_zipped:
                    context = g + ua + u
                    label_context = len(g)*[cfg.pad_id]+ua+u
                    #context = g + [self.sos_r_id, self.eos_r_id] + ua + u
                    #label_context=(len(g)+2)*[cfg.pad_id]+ua+u
                    contexts.append(context)
                    label_contexts.append(label_context)
            else:
                if mode=='oracle':
                    batch_zipped = zip(pv_batch, turn_batch['goal'], turn_batch['usr_act'], turn_batch['user'])
                else:
                    batch_zipped=zip(pv_batch, turn_batch['goal'], turn_batch['usr_act_gen'], turn_batch['user_gen'])                   
                for pv, g, ua, u in batch_zipped:
                    context = pv + g + ua + u
                    label_context=len(pv+g)*[cfg.pad_id]+ua+u
                    contexts.append(context)
                    label_contexts.append(label_context)

        inputs['contexts'] = contexts
        inputs['contexts_np'], inputs['lengths'] = utils.padSeqs_gpt(inputs['contexts'], cfg.pad_id)
        labels['contexts']=label_contexts
        labels['contexts_np'], labels['lengths']=utils.padSeqs_gpt(labels['contexts'], cfg.pad_id)
        if seg_label and side=='sys':
            seg_labels['contexts']=seg_contexts
            seg_labels['contexts_np'], seg_labels['lengths']=utils.padSeqs_gpt(seg_labels['contexts'], cfg.pad_id)
            return inputs, labels, seg_labels
        if rl_train and side=='sys':
            rl_labels['contexts']=rl_label_contexts
            rl_labels['contexts_np'], rl_labels['lengths']=utils.padSeqs_gpt(rl_labels['contexts'], cfg.pad_id)
            return inputs, labels, rl_labels
        else:
            return inputs, labels


    def convert_eval_batch_turn(self, turn_batch, pv_batch, mode='gen_bspn', bspn_gen=None, db_gen=None, posterior=False):
        eval_batch=[]
        assert mode in ['gen_bspn', 'gen_ar']
        if pv_batch is None:
            if mode=='gen_bspn':
                for u, r in zip(turn_batch['user'], turn_batch['resp']):
                    context=u+r+[self.sos_b_id] if posterior else u+[self.sos_b_id]
                    eval_batch.append(context)
            else:
                for u, b, d, r in zip(turn_batch['user'], bspn_gen, db_gen, turn_batch['resp']):
                    context=u+r+b+d+[self.sos_a_id] if posterior else u+b+d+[self.sos_a_id]
                    eval_batch.append(context)
        else:
            if mode=='gen_bspn':
                for hist, u, r in zip(pv_batch, turn_batch['user'], turn_batch['resp']):
                    context=hist+u+r+[self.sos_b_id] if posterior else hist+u+[self.sos_b_id]
                    eval_batch.append(context)
            else:
                for hist, u, b, d, r in zip(pv_batch, turn_batch['user'], bspn_gen, db_gen, turn_batch['resp']):
                    context=hist+u+r+b+d+[self.sos_a_id] if posterior else hist+u+b+d+[self.sos_a_id]
                    eval_batch.append(context)
        return eval_batch
    
    def convert_dst_eval_batch(self, turn_batch, pv_bspn, pv_resp):
        eval_batch=[]
        if pv_bspn is None:#first turn
            for u in turn_batch['user']:
                eval_batch.append(u+[self.sos_b_id])
        else:
            for b, r, u in zip(pv_bspn, pv_resp, turn_batch['user']):
                eval_batch.append(b+r+u+[self.sos_b_id])
        return eval_batch
    
    def convert_dm_eval_batch(self, turn_batch, bspn_batch, db_batch, pv_bspn, pv_resp):
        eval_batch=[]
        if pv_bspn is None:
            for u, b, db in zip(turn_batch['user'], bspn_batch, db_batch):
                eval_batch.append(u+b+db+[self.sos_a_id])
        else:
            for pv_b, pv_r, u, b, db in zip(pv_bspn, pv_resp, turn_batch['user'], bspn_batch, db_batch):
                eval_batch.append(pv_b+pv_r+u+b+db+[self.sos_a_id])
        return eval_batch

    def convert_nlg_eval_batch(self, turn_batch, bspn_batch, db_batch, aspn_batch, pv_bspn, pv_resp):
        eval_batch=[]
        if pv_bspn is None:
            for u, b, db, a in zip(turn_batch['user'], bspn_batch, db_batch, aspn_batch):
                eval_batch.append(u+b+db+a+[self.sos_r_id])
        else:
            for pv_b, pv_r, u, b, db, a in zip(pv_bspn, pv_resp, turn_batch['user'], bspn_batch, db_batch, aspn_batch):
                eval_batch.append(pv_b+pv_r+u+b+db+a+[self.sos_r_id])
        return eval_batch


    def convert_eval_batch_turn_us(self, turn_batch, pv_batch, user_act=None):
        eval_batch=[]
        if user_act is None:# generate user act (and utterance)
            if pv_batch==None: # first turn
                for g in turn_batch['goal']:
                    eval_batch.append(g + [self.sos_ua_id])
            else:
                for g, pv in zip(turn_batch['goal'], pv_batch):
                    eval_batch.append(pv + g + [self.sos_ua_id])
        else:# generate user utterance
            if pv_batch==None:
                for g, ua in zip(turn_batch['goal'], user_act):
                    eval_batch.append(g + ua + [self.sos_u_id])
            else:
                for g, pv, ua in zip(turn_batch['goal'], pv_batch, user_act):
                    eval_batch.append(pv + g + ua + [self.sos_u_id])
        return eval_batch


    def convert_us_batch_session(self, dial_batch):
        
        inputs = {}
        labels={}
        contexts = []
        label_contexts=[]
        cell_list = ['usr_act','user','resp']

        for idx, dial in enumerate(dial_batch):
            context = []
            label_context=[]
            context.extend(dial['goal'])
            label_context.extend(len(dial['goal'])*[cfg.pad_id])
            for turn_num, turn in enumerate(dial['log']):
                for cell in cell_list:                  
                    context.extend(turn[cell])
                    label_context.extend(turn[cell])           
            contexts.append(context)
            label_contexts.append(label_context)
      
        inputs['contexts'] = contexts
        inputs['contexts_np'], inputs['lengths'] = utils.padSeqs_gpt(inputs['contexts'], cfg.pad_id)
        labels['contexts']=label_contexts
        labels['contexts_np'], labels['lengths']=utils.padSeqs_gpt(labels['contexts'], cfg.pad_id)

        return inputs,labels
    
    def wrap_result_lm(self, result_dict, eos_syntax=None):
        results = []
        eos_syntax = ontology.eos_tokens if not eos_syntax else eos_syntax
        sos_syntax = ontology.sos_tokens
        # ground truth bs, as, ds.. generate response
        field = ['dial_id', 'turn_num', 'user', 'bspn', 'bspn_gen', 'db', 'db_gen', 'aspn_gen', 'aspn', 'resp_gen', 'resp', 'dspn']

        for dial_id, turns in result_dict.items():
            entry = {'dial_id': dial_id, 'turn_num': len(turns)}
            for f in field[2:]:
                entry[f] = '' # ???
            results.append(entry)
            for turn_idx, turn in enumerate(turns):
                entry = {'dial_id': dial_id}
                for key in field:
                    if key in ['dial_id']:
                        continue
                    v = turn.get(key, '')
                    if key == 'turn_domain':
                        v = ' '.join(v)

                    if key in eos_syntax and v != '':
                        # remove eos tokens
                        v = self.tokenizer.decode(v)
                        v = v.split()
                        # remove eos/sos in span
                        if eos_syntax[key] in v:
                            v.remove(eos_syntax[key])
                        if sos_syntax[key] in v:
                            v.remove(sos_syntax[key])
                        # if key != 'resp_gen':
                        #     # remove eos/sos in span
                        #     if eos_syntax[key] in v:
                        #         v.remove(eos_syntax[key])
                        #     if sos_syntax[key] in v:
                        #         v.remove(sos_syntax[key])
                        # else: # 'resp_gen'
                        #     sos_index = 0
                        #     eos_index = -1
                        #     if sos_syntax[key] in v:
                        #         sos_index = v.index(sos_syntax[key])
                        #     if eos_syntax[key] in v:
                        #         eos_index = v.index(eos_syntax[key])
                        #     else:
                        #         pass # take too long
                        #         # no <eos_r> found, stop at any eos_tokens
                        #         # for i in range(sos_index+1, len(v)):
                        #         #     if v[i] in sos_syntax.values() or v[i] in eos_syntax.values():
                        #         #         eos_index = i
                        #     v = v[sos_index+1: eos_index]


                        # v = self.tokenizer.convert_tokens_to_string(v)
                        v = " ".join(v)
                    else: 
                        pass # v = v
                    entry[key] = v

                results.append(entry)

        return results, field

class tod_dataset(Dataset):
    def __init__(self, data):
        self.data=data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def train_collate_fn(batch):
    # item[0]: input text
    # item[1]: target text
    data=[item[0]+item[1] for item in batch]
    label=[[cfg.pad_id]*len(item[0])+item[1] for item in batch]
    data_np, _=utils.padSeqs_gpt(data, cfg.pad_id)
    label_np, _=utils.padSeqs_gpt(label, cfg.pad_id)
    data_tensor=torch.from_numpy(data_np).long()
    label_tensor=torch.from_numpy(label_np).long()
    return [data_tensor, label_tensor]

def test_collate_fn(batch):
    # prediction
    sos_id=batch[0][1][0]
    data=[item[0]+[sos_id] for item in batch]
    label=[item[1] for item in batch]
    return [data, label]

if __name__ == '__main__':
    reader = MultiWozReader()
    # for aspan in ["[general] [bye] [welcome] <eos_a>","[train] [inform] trainid destination arrive leave [offerbook] [general] [reqmore] <eos_a>",]:
    #     act = reader.aspan_to_constraint_dict(aspan.split())
    #     print('！！！')
    #     print(act)

    for bspan in ["[taxi] destination golden house departure broughton house gallery arrive 19:30 [attraction] type museum name whipple museum of the history of science people 5 day monday", "[taxi] destination golden house departure broughton house gallery arrive 19:30 [attraction] type museum name whipple museum of the history of science people 5 day monday <eos_b>"]:
        encoded = reader.vocab.sentence_encode(bspan.split())
        print(encoded)
        cons = reader.bspan_to_constraint_dict(encoded, bspn_mode='bspn')
        print(cons)
    for bspan in ["[taxi] destination departure leave [hotel] name [attraction] name people day", "[taxi] destination departure leave [hotel] name [attraction] name people day <eos_b>"]:
        encoded = reader.vocab.sentence_encode(bspan.split())
        print(encoded)
        cons = reader.bspan_to_constraint_dict(encoded, bspn_mode='bsdx')
        print(cons)