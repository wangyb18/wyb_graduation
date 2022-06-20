import json
from lib2to3.pgen2 import token

from sklearn.model_selection import train_test_split

# 读取json文件内容,返回字典格式
with open('/mnt/workspace/wangyb/SGA-JRUD-master/data/multi-woz-2.1-processed/new_db_se_blank_encoded.data_fix.json', 'r', encoding='utf8')as fp:
    result = json.load(fp)

train_result = result['train']
print(len(train_result))
print(train_result[0][0])
len_dic = {i: 0 for i in range(1000)}
turn_sum = 0
tokens_sum = 0


def compute_len(turn_dict):
    tokens_num = 0
    tokens_num += len(turn_dict['user'])
    tokens_num += len(turn_dict['bspn'])
    tokens_num += len(turn_dict['resp'])
    tokens_num += len(turn_dict['aspn'])
    tokens_num += len(turn_dict['dspn'])
    tokens_num += len(turn_dict['pointer'])
    tokens_num += len(turn_dict['turn_domain'])
    tokens_num += len(turn_dict['db'])
    return tokens_num


for i in range(len(train_result)):
    for j in range(len(train_result[i])):
        token_num = compute_len(train_result[i][j])
        len_dic[token_num] += 1
        tokens_sum += token_num
        turn_sum += 1

for i in range(1000):
    len_dic[i] = len_dic[i]/turn_sum
print(len_dic)
print('average token num:', tokens_sum/turn_sum)

with open("/mnt/workspace/wangyb/SGA-JRUD-master/data/multi-woz-2.1-processed/len_dic.json", "w") as f:
    json.dump(len_dic, f)
    print("加载入文件完成...")
