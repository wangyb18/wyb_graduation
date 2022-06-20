import json
import ontology
from transformers import GPT2Tokenizer
import transformers
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained('tokenizer_file')

informable_slots = {
    "taxi": ["leave", "destination", "departure", "arrive"],
    "police": [],
    "hospital": ["department"],
    "hotel": ["type", "parking", "pricerange", "price", "internet", "stay", "day", "people", "area", "stars", "name"],
    "attraction": ["area", "type", "name"],
    "train": ["destination", "day", "arrive", "departure", "people", "leave"],
    "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"]
}


def divide(bspn_words, prob):
    # 得到domain对应的word级别words words_gen prob_gen
    # 现在这个函数的写法有一定风险
    word_index = 0
    domain_words, slot_words, value_words = [], [], []
    domain_prob, slot_prob, value_prob = [], [], []
    while word_index < len(bspn_words):
        domain_words.append(bspn_words[word_index])
        domain_prob.append(prob[word_index])
        domain_index = word_index
        word_index += 1
        while word_index < len(bspn_words) and '[' not in bspn_words[word_index]:
            slot_words.append(bspn_words[word_index])
            slot_prob.append(prob[word_index])
            word_index += 1
            # print('bspn_words[domain_index]:', bspn_words[domain_index])
            if bspn_words[word_index] not in informable_slots[bspn_words[domain_index][1:-1]]:
                value_words.append(bspn_words[word_index])
                value_prob.append(prob[word_index])
                word_index += 1
    # print("domain_words:", domain_words, "domain_prob:", domain_prob)
    # print("slot_words:", slot_words, "slot_prob", slot_prob)
    # print("value_words:", value_words, 'value_prob:', value_prob)
    return domain_words, slot_words, value_words, domain_prob, slot_prob, value_prob


def get_words_prob(bspn_words, bspn_tokens, bspn_prob):
    if bspn_tokens[0] == '<sos_b>':
        bspn_tokens = process_tokens(bspn_tokens)
    bspn_words_prob_gen = []
    gen_index = 0
    # print("bspn_words:", bspn_words)
    # print("bspn_tokens:", bspn_tokens)
    for word in bspn_words:
        temp_prob = 1
        # print("word:", word)
        word_index = 0
        # print("gen_index:", gen_index)
        # print('bspn_tokens[gen_index]:', bspn_tokens[gen_index])
        # print('word[word_index:word_index + len(bspn_tokens[gen_index])]:',
        #       word[word_index:word_index + len(bspn_tokens[gen_index])])
        while bspn_tokens[gen_index] == word[word_index:word_index + len(bspn_tokens[gen_index])]:
            # print('bspn_tokens[gen_index]:', bspn_tokens[gen_index])
            # print('word[word_index:word_index + len(bspn_tokens[gen_index])]:',
            #       word[word_index:word_index + len(bspn_tokens[gen_index])])
            # print("gen_index:", gen_index)
            temp_prob *= bspn_prob[gen_index]
            gen_index += 1
            if gen_index < len(bspn_prob) and word_index + len(bspn_tokens[gen_index]) < len(word):
                word_index += len(bspn_tokens[gen_index])
            else:
                break
            # print("word_index:", word_index)
        bspn_words_prob_gen.append(temp_prob)
    return bspn_words_prob_gen


def process_tokens(tokens):
    for i in range(len(tokens)):
        if "\u0120" in tokens[i]:
            tokens[i] = tokens[i][1:]
    tokens = tokens[1:-1]
    return tokens


def constraint_dict_to_list(bspn_dict):
    bspn_word = []
    for domain in bspn_dict.keys():
        word_domain = '[' + domain + ']'
        bspn_word.append(word_domain)
        for slot in bspn_dict[domain].keys():
            bspn_word.append(slot)
            bspn_word.append(bspn_dict[domain][slot])
    return bspn_word


def bspan_to_constraint_dict(bspan):
    bspan = bspan.split()
    constraint_dict = {}
    domain = None
    conslen = len(bspan)
    for idx, cons in enumerate(bspan):
        if cons == '<eos_b>':
            break
        if '[' in cons:
            if cons[1:-1] not in ontology.all_domains:
                continue
            domain = cons[1:-1]
            if domain not in constraint_dict:
                constraint_dict[domain] = {}
        elif cons in ontology.get_slot:
            if domain is None:
                continue
            if cons == 'people':
                try:
                    ns = bspan[idx + 1]
                    if ns == "'s":
                        continue
                except:
                    continue
            if not constraint_dict.get(domain):
                constraint_dict[domain] = {}
            vidx = idx + 1
            if vidx == conslen:
                break
            vt_collect = []
            vt = bspan[vidx]
            while vidx < conslen and vt != '<eos_b>' and '[' not in vt and vt not in ontology.get_slot:
                vt_collect.append(vt)
                vidx += 1
                if vidx == conslen:
                    break
                vt = bspan[vidx]
            if vt_collect:
                constraint_dict[domain][cons] = ' '.join(vt_collect)

    return constraint_dict


def cons_dict_to_bspn(cons_dict):
    bs_list = []
    for domain in cons_dict:
        bs_list.append('[' + domain + ']')
        for slot in cons_dict[domain]:
            bs_list.append(slot)
            bs_list.append(cons_dict[domain][slot])
    return ' '.join(bs_list)


def get_tokens_position(temp_token, bspn_tokens):
    i = 1
    start = 0
    while i < len(temp_token):
        if bspn_tokens.index(temp_token[i], start) - bspn_tokens.index(temp_token[i - 1], start
                                                                       ) != 1:
            start = min(bspn_tokens.index(temp_token[i], start), bspn_tokens.index(temp_token[i - 1], start)) + 1
            i = 1
        else:
            i += 1
    temp_position = [bspn_tokens.index(temp_token[0], start) + i for i in range(len(temp_token))]
    return temp_position


def reorder(cons_dict, bspn_word):
    # 记录排序结果的列表(tokenize化后)
    bspn_word_sort = []
    # 记录位置变化的列表
    pos_list = []
    # 首先按领域排序
    for domain in informable_slots.keys():
        if domain in cons_dict.keys():
            word_domain = '[' + domain + ']'
            bspn_word_sort.append(word_domain)
            pos_list.append(bspn_word.index(word_domain))
            for slot in informable_slots[domain]:
                if slot in cons_dict[domain].keys():
                    # 若该键位于bspn中，将其与之后的值tokennize之后加入排序后列表，并将其对应索引加入
                    bspn_word_sort.append(slot)
                    bspn_word_sort.append(cons_dict[domain][slot])
                    temp_position = get_tokens_position([slot, cons_dict[domain][slot]], bspn_word)
                    # print(temp_position)
                    pos_list += temp_position
    return bspn_word_sort, pos_list


def is_slot(token):
    for domain in informable_slots:
        if token in informable_slots[domain]:
            return True
    return False


def get_domain(index, tokens):
    for j in reversed(range(index - 1)):
        if tokens[j] in informable_slots.keys():
            return tokens[j], j
    print("未搜寻到领域！")
    return tokens[0], 0


def sort_words(bspn):
    # 第一步：将bspn转化为字典形式（cons）
    # print("bspn:", bspn)
    bspn_dict = bspan_to_constraint_dict(bspn)
    # print("bspn_dict:", bspn_dict)
    bspn_word = constraint_dict_to_list(bspn_dict)
    # print("bspn_word:", bspn_word)
    # 第二步：根据上述informable_slots里的domain和slot的顺序对bspn排序，返回排序后的bspn_tokens_sort
    # 并在排序过程中记录位置变化
    bspn_words_sort, pos_list = reorder(bspn_dict, bspn_word)
    # print("bspn_word_sort:", bspn_words_sort)
    # print("pos_list:", pos_list)

    return bspn_words_sort, pos_list


tokenizer = GPT2Tokenizer.from_pretrained('tokenizer_file')

# 得到bspn_tokens
bspn = '[hotel] parking beijing type juzi [taxi] destination here leave 9:00'
bspn_words_sort, pos_list = sort_words(bspn)
print("bspn_words_sort:", bspn_words_sort)
print("pos_list:", pos_list)
bspn_words = ['my', 'sister', 'is', 'a', 'girl']
bspn_tokens = ['m', 'y', 'sister', 'is', 'a', 'gi', 'rl']
bspn_prob = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
print(get_words_prob(bspn_words, bspn_tokens, bspn_prob))
print(bspan_to_constraint_dict('[attraction] [restaurant] area centre pricerange expensive'))
##
l = [1, 1, 1]
print(l.index(1))
print(l.index(1, 1))
print(l.index(1, 2))
#
# bspn_tokens = tokenizer.tokenize(bspn)
# for i in range(len(bspn_tokens)):
#     if "\u0120" in bspn_tokens[i]:
#         bspn_tokens[i] = bspn_tokens[i][1:]
# print(bspn_tokens)
#
# # 第一步：将bspn转化为字典形式（cons）
# cons = bspan_to_constraint_dict(bspn)
# print(cons)
#
# # 第二步：根据上述informable_slots里的domain和slot的顺序对bspn排序，返回排序后的bspn_tokens_sort
# # 并在排序过程中记录位置变化
# bspn_tokens_sort, pos_list = reorder(cons, bspn_tokens)
# print("bspn_tokens_sort:", bspn_tokens_sort)
# print("pos_list:", pos_list)
# # 第三步：将排序完成的cons转回序列形式，并且对其tokenize
# bspn_sort = cons_dict_to_bspn(cons_sort)
# bspn_tokens_sort = tokenizer.tokenize(bspn_sort)
# for i in range(len(bspn_tokens_sort)):
#     if "\u0120" in bspn_tokens_sort[i]:
#         bspn_tokens_sort[i] = bspn_tokens_sort[i][1:]
# print(bspn_tokens_sort)
#
# # 第四步（待完成）：根据上述重排后的tokens列表对相应的概率进行重排
# # 返回对应的概率位置的数组
# # 假定文本中不会出现领域值与槽值
# prob_index = []
# for i in range(len(bspn_tokens_sort)):
#     if bspn_tokens.count(bspn_tokens_sort[i]) == 1:
#         prob_index.append(bspn_tokens.index(bspn_tokens_sort[i]))
#         print(prob_index)
#     # 如果不是槽值
#     elif not is_slot(bspn_tokens_sort[i]):
#         prob_index.append(prob_index[i - 1])
#     # 如果是槽值
#     else:
#         # 首先得到排序后列表该槽值的领域
#         sort_domain, sort_domain_index = get_domain(i, bspn_tokens_sort)
#         # 得到排序后列表槽值坐标距离领域坐标的距离
#         domain_slot_distance = i - sort_domain_index
#         start = 0
#         bspn_domain, bspn_domain_index = get_domain(bspn_tokens.index(bspn_tokens_sort[i], start), bspn_tokens)
#         while bspn_domain != sort_domain and domain_slot_distance != bspn_tokens.index(bspn_tokens_sort[i],
#                                                                                        start) - bspn_domain_index:
#             start = bspn_tokens.index(bspn_tokens_sort[i], start) + 1
#             bspn_domain, bspn_domain_index = get_domain(bspn_tokens.index(bspn_tokens_sort[i], start), bspn_tokens)
#         prob_index.append(bspn_tokens.index(bspn_tokens_sort[i], start))
