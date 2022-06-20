import numpy as np
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
from levenshtein import levenshtein
from utils import sort_words, constraint_dict_to_list, bspan_to_constraint_dict, get_words_prob, informable_slots, \
    divide
import openpyxl


def get_list_in_result(index, key):
    l = result[index][key]
    l = eval(l)
    return l


def record_confidence_accuracy(bspn_words_sort, bspn_words_gen_sort, confidence_list, accuracy_list, bspn_prob_gen,
                               confidence_bin_border):
    # 得到confidence所在区间
    distance, bspn_right_array = levenshtein(bspn_words_sort, bspn_words_gen_sort)
    # print('distance', distance)
    # print('bspn_right_array', bspn_right_array)
    for i in range(len(bspn_prob_gen)):
        temp_list = confidence_bin_border < bspn_prob_gen[i]
        confidence_bin_position = (temp_list.sum() - 1)[0]
        if confidence_bin_position == -1:
            continue
        if bspn_prob_gen[i] <= confidence_bin_border.iloc[confidence_bin_position, 0]:
            print(bspn_prob_gen[i])
            print(confidence_bin_position)
            exit(0)
        confidence_list[confidence_bin_position].append(bspn_prob_gen[i])
        positive_negative_flag = bspn_right_array[i]
        accuracy_list[confidence_bin_position].append(positive_negative_flag)
        # print("confidence_list:", confidence_list)
        # print("accuracy_list:", accuracy_list)
    return confidence_list, accuracy_list


def compute_ECE(accuracy_list, confidence_list, bin_num, T, ECE_list, word_type=''):
    # print("accuracy_list:", accuracy_list)
    # print("bin_num:", bin_num)
    accuracy_array = np.array(
        [sum(accuracy_list[i]) / len(accuracy_list[i]) if len(accuracy_list[i]) != 0 else 0 for i in
         range(bin_num)]).squeeze()
    # print("accuracy_array:", accuracy_array)
    accuracy_num_array = np.array([len(accuracy_list[i]) for i in range(bin_num)]).squeeze()
    confidence_array = np.array(
        [sum(confidence_list[i]) / len(confidence_list[i]) if len(confidence_list[i]) != 0 else 0 for i in
         range(bin_num)]).squeeze()
    bin_sum = [len(confidence_list[i]) for i in range(bin_num)]
    print(word_type, " accuracy_array:", accuracy_array)
    print(word_type, " confidence_array:", confidence_array)
    ECE_list[str(T)] = sum(abs(confidence_array - accuracy_array) * accuracy_num_array) / sum(accuracy_num_array)
    print(word_type, " T:", T, " ECE:", ECE_list[str(T)])
    if T == '1.00':
        # 将accuracy与confidence数据写入excel
        write_acc_con_to_excel(confidence_array, accuracy_array, bin_sum, word_type)
        # 绘制不同区间内acc与con的对比曲线
        plot_acc_con_figure(confidence_array, accuracy_array, word_type)
    return accuracy_array, confidence_array, bin_sum, ECE_list


def write_acc_con_to_excel(confidence_array, accuracy_array, bin_sum, word_type):
    # 将accuracy与confidence数据写入excel
    data = pd.DataFrame()
    data['confidence_array'] = confidence_array
    data['accuracy_array'] = accuracy_array
    data['bin_sum'] = bin_sum
    data.to_excel(word_type + '_data.xlsx')
    return data


def write_ECE_to_excel(ECE_list, word_type):
    # 将ECE数据写入excel
    data = pd.DataFrame()
    data['T'] = ECE_list.keys()
    data['ECE'] = ECE_list.values()
    # print("data:", data)
    data.to_excel(word_type + 'ECE' + '.xlsx')
    return data


def plot_ECE_figure(word_type):
    # 画图
    global figure_num
    plt.figure(figure_num)
    figure_num += 1
    data = pd.read_excel(word_type + 'ECE' + '.xlsx')
    plt.plot(data['T'], data['ECE'])
    plt.xlabel('T')
    plt.ylabel('ECE')
    plt.title(word_type + ': ECE-T figure')
    plt.show()


def plot_acc_con_figure(confidence_array, accuracy_array, word_type):
    # 绘制图像
    global figure_num
    plt.figure(figure_num)
    figure_num += 1
    total_width, n = 10, 2 * len(confidence_array)
    width = total_width / n
    tick_list = [round(low + i * (high - low) / bin_num, 4) for i in range(bin_num)]
    confidence_x = range(len(confidence_array))
    accuracy_x = [confidence_x[i] + width for i in range(len(confidence_x))]
    plt.bar(accuracy_x, accuracy_array, width=width, label='accuracy', fc='b', tick_label=tick_list)
    plt.bar(confidence_x, confidence_array, width=width, label='confidence', fc='r', tick_label=tick_list)
    plt.xlabel('bin')
    plt.title(word_type + ': The comparison of accuracy and confidence')
    plt.legend()
    plt.show()


def plot_alarm_missing_curve(word_type, FN_array, FP_array, TN_array, TP_array):
    # 计算missing_alarm与false_alarm
    MA_array = FN_array / (TP_array + FN_array)
    FA_array = FP_array / (TP_array + FP_array)
    plt.plot(MA_array, FA_array)
    plt.xlabel('MA')
    plt.ylabel('FA')
    plt.title(word_type, 'FA-MA figure')
    plt.show()


# 读取json文件内容,返回字典格式
with open('result-ood.json', 'r', encoding='utf8')as fp:
    result = json.load(fp)
figure_num = 0
ID_num = 0
turn_num = 0
domain_position = 'ood_'
# 记录confidence与accuracy的列表
low = 0
high = 1
bin_num = 20
T_num = 40
T_begin = 1
T_end = 3
T_list = ['{:.2f}'.format(T_begin + i * (T_end - T_begin) / T_num) for i in range(T_num)]
ECE_list = {}
domain_ECE_list, slot_ECE_list, value_ECE_list = {}, {}, {}
confidence_list = [[] for i in range(bin_num)]
accuracy_list = [[] for i in range(bin_num)]
domain_confidence_list = [[] for i in range(bin_num)]
domain_accuracy_list = [[] for i in range(bin_num)]
slot_confidence_list = [[] for i in range(bin_num)]
slot_accuracy_list = [[] for i in range(bin_num)]
value_confidence_list = [[] for i in range(bin_num)]
value_accuracy_list = [[] for i in range(bin_num)]
tick_list = [round(low + i * (high - low) / bin_num, 4) for i in range(bin_num)]
confidence_bin_border = pd.DataFrame(tick_list)
type_list = ['domain', 'slot', 'value']
print(confidence_bin_border)
print(T_list)

##
T_list = ["1.00"]
# 提取出每一个步进点的温度
for T in T_list:
    bspn_prob_index = 'bspn_prob_' + str(T)
    # 按对话循环,用索引实现
    result_index = 0
    while result_index < len(result):
        turn_index = 1
        while turn_index <= result[result_index]['turn_num']:

            # 计算总共有多少轮
            turn_num += 1

            # 对原有的与生成的bspn中的word进行排序排序
            bspn_words_sort, pos_list = sort_words(result[result_index + turn_index]['bspn'])
            bspn_words_gen_sort, pos_gen_list = sort_words(result[result_index + turn_index]['bspn_gen'])

            # 得到去掉头尾的token级的概率列表
            bspn_prob_gen = get_list_in_result(result_index + turn_index, bspn_prob_index)
            # print("len(bspn_prob_gen):", len(bspn_prob_gen))
            bspn_prob_gen = bspn_prob_gen[1:-1]

            # 将概率合并为word级
            bspn_tokens_gen = get_list_in_result(result_index + turn_index, 'bspn_tokens_gen')
            bspn_words_gen = constraint_dict_to_list(
                bspan_to_constraint_dict(result[result_index + turn_index]['bspn_gen']))
            bspn_words_prob_gen = get_words_prob(bspn_words_gen, bspn_tokens_gen, bspn_prob_gen)

            # 得到排序后对应的概率列表
            bspn_prob_gen = [bspn_words_prob_gen[i] for i in pos_gen_list]

            domain_words, slot_words, value_words, domain_prob, slot_prob, value_prob = divide(bspn_words_sort,
                                                                                               np.zeros(len(
                                                                                                   bspn_words_sort)))
            domain_words_gen, slot_words_gen, value_words_gen, domain_prob_gen, slot_prob_gen, value_prob_gen = divide(
                bspn_words_gen, bspn_words_prob_gen)

            # # 剔除去域内数据
            # if domain_position == 'ood_' and '[attraction]' not in domain_words and len(domain_words) != 0:
            #     print("domain_words:", domain_words)
            #     turn_index += 1
            #     ID_num += 1
            #     continue

            # # 剔除去域内数据
            # if domain_position == 'ood_' and domain_words != ['[attraction]']:
            #     # print("domain_words:", domain_words)
            #     turn_index += 1
            #     ID_num += 1
            #     continue

            # 调试所用代码
            if len(bspn_prob_gen) != 0 and len(bspn_prob_gen) == sum(bspn_prob_gen):
                exit(0)

            # 得到confidence所在区间
            domain_confidence_list, domain_accuracy_list = record_confidence_accuracy(domain_words,
                                                                                      domain_words_gen,
                                                                                      domain_confidence_list,
                                                                                      domain_accuracy_list,
                                                                                      domain_prob_gen,
                                                                                      confidence_bin_border)
            slot_confidence_list, slot_accuracy_list = record_confidence_accuracy(slot_words,
                                                                                  slot_words_gen,
                                                                                  slot_confidence_list,
                                                                                  slot_accuracy_list,
                                                                                  slot_prob_gen,
                                                                                  confidence_bin_border)
            value_confidence_list, value_accuracy_list = record_confidence_accuracy(value_words,
                                                                                    value_words_gen,
                                                                                    value_confidence_list,
                                                                                    value_accuracy_list,
                                                                                    value_prob_gen,
                                                                                    confidence_bin_border)

            turn_index += 1
        result_index += turn_index

    # 计算ECE
    domain_accuracy_array, domain_confidence_array, domain_bin_sum, domain_ECE_list = compute_ECE(domain_accuracy_list,
                                                                                                  domain_confidence_list,
                                                                                                  bin_num, T,
                                                                                                  domain_ECE_list,
                                                                                                  'domain')
    slot_accuracy_array, slot_confidence_array, slot_bin_sum, slot_ECE_list = compute_ECE(slot_accuracy_list,
                                                                                          slot_confidence_list,
                                                                                          bin_num, T,
                                                                                          slot_ECE_list, 'slot')
    value_accuracy_array, value_confidence_array, value_bin_sum, value_ECE_list = compute_ECE(value_accuracy_list,
                                                                                              value_confidence_list,
                                                                                              bin_num, T,
                                                                                              value_ECE_list, 'value')

for word_type in type_list:
    # print(eval(word_type + "_ECE_list"))
    write_ECE_to_excel(eval(word_type + "_ECE_list"), word_type)
    plot_ECE_figure(word_type)

print("ID_num:", ID_num)
print("turn_num:", turn_num)
##
# 画图
data = pd.read_excel('ECE' + str(T_begin) + '_' + str(T_end) + '.xlsx')
plt.plot(data['T'], data['ECE'])
plt.xlabel('T')
plt.ylabel('ECE')
plt.title('ECE-T figure')
plt.show()
##
# 写出数据
data = pd.DataFrame()
data['confidence_array'] = confidence_array
data['accuracy_array'] = accuracy_array
data['bin_sum'] = bin_sum
data.to_excel('data1.xlsx')
##
import matplotlib.pyplot as plt

# 绘制图像
total_width, n = 10, 2 * len(confidence_array)
width = total_width / n
tick_list = [round(low + i * (high - low) / bin_num, 4) for i in range(bin_num)]
confidence_x = range(len(confidence_array))
accuracy_x = [confidence_x[i] + width for i in range(len(confidence_x))]
plt.bar(accuracy_x, accuracy_array, width=width, label='accuracy', fc='b', tick_label=tick_list)
plt.bar(confidence_x, confidence_array, width=width, label='confidence', fc='r', tick_label=tick_list)
plt.xlabel('bin')
plt.title('The comparison of accuracy and confidence')
plt.legend()
plt.show()

print("accuracy_num_array =", accuracy_num_array)

##
# # 第一次应将键值修整的整洁
# if T == 1.0:
#     result[result_index + turn_index]['bspn_prob_1.7'] = result[result_index + turn_index].pop(
#         'bspn_prob_1.7000000000000002')
#     result[result_index + turn_index]['bspn_prob_1.95'] = result[result_index + turn_index].pop(
#         'bspn_prob_1.9500000000000002')
