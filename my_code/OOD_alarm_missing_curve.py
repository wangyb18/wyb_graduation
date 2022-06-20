import numpy as np
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
from levenshtein import levenshtein
from utils import sort_words, constraint_dict_to_list, bspan_to_constraint_dict, get_words_prob, informable_slots, \
    divide
import openpyxl


def plot_MA_FA_curve(TP_array, FP_array, TN_array, FN_array, word_type):
    # 绘制MA_FA曲线
    MA_array = FN_array / (TP_array + FN_array)
    FA_array = FP_array / (TP_array + FP_array)
    plt.plot(MA_array, FA_array)
    plt.xlabel('MA')
    plt.ylabel('FA')
    plt.title(word_type, 'FA-MA figure')
    plt.show()
    return MA_array, FA_array


def classify(P_or_N, prob, TP, FP, TN, FN):
    # 归类至四类中某一类
    # TP
    if P_or_N == 1 and prob > confidence_threshold:
        TP += 1
    # FP
    elif P_or_N == 1 and prob <= confidence_threshold:
        FP += 1
    # TN
    elif P_or_N == 0 and prob <= confidence_threshold:
        TN += 1
    # FN
    elif P_or_N == 0 and prob > confidence_threshold:
        FN += 1
    else:
        print("分类错误！")
    return TP, FP, TN, FN


def get_list_in_result(index, key):
    l = result[index][key]
    l = eval(l)
    return l


def record_confidence_accuracy(bspn_right_array, confidence_list, accuracy_list, bspn_prob_gen,
                               confidence_bin_border):
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
    # 绘制accuracy与confidence图像
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


alarm_missing_figure = 10
def plot_alarm_missing_curve(word_type, FN_array, FP_array, TN_array, TP_array):
    # 绘制missing_alarm与false_alarm曲线
    global figure_num
    plt.figure(figure_num)
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
domain_position = 'ood_'
type_list = ['domain_', 'slot_', 'value_']
# type_list = ['domain_']
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
bspn_words_sort, pos_gen_list = [], []
domain_bspn_right_array, slot_bspn_right_array, value_bspn_right_array = [], [], []
print(confidence_bin_border)
print(T_list)
# 置信度域值
threshold_num = 20
threshold_low = 0.9
threshold_high = 1

confidence_threshold_list = [threshold_low + i * (threshold_high - threshold_low) / threshold_num for i in
                             range(threshold_num)]
domain_TP_array, domain_FP_array, domain_TN_array, domain_FN_array = np.zeros(len(confidence_threshold_list)), np.zeros(
    len(confidence_threshold_list)), np.zeros(len(confidence_threshold_list)), np.zeros(len(confidence_threshold_list))
slot_TP_array, slot_FP_array, slot_TN_array, slot_FN_array = np.zeros(len(confidence_threshold_list)), np.zeros(
    len(confidence_threshold_list)), np.zeros(len(confidence_threshold_list)), np.zeros(len(confidence_threshold_list))
value_TP_array, value_FP_array, value_TN_array, value_FN_array = np.zeros(len(confidence_threshold_list)), np.zeros(
    len(confidence_threshold_list)), np.zeros(len(confidence_threshold_list)), np.zeros(len(confidence_threshold_list))

##
# T_list = ["1.00"]
# 提取出每一个步进点的温度
for T in T_list:
    bspn_prob_index = 'bspn_prob_' + str(T)
    # 按对话循环,用索引实现
    result_index = 0
    while result_index < len(result):
        turn_index = 1
        while turn_index <= result[result_index]['turn_num']:

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

            # 调试所用代码
            if len(bspn_prob_gen) != 0 and len(bspn_prob_gen) == sum(bspn_prob_gen):
                exit(0)

            # 得到right_array
            if T == '1.00':
                domain_distance, domain_bspn_right_array = levenshtein(domain_words, domain_words_gen)
                slot_distance, slot_bspn_right_array = levenshtein(slot_words, slot_words_gen)
                value_distance, value_bspn_right_array = levenshtein(value_words, value_words_gen)

            # 得到confidence所在区间
            domain_confidence_list, domain_accuracy_list = record_confidence_accuracy(domain_bspn_right_array,
                                                                                      domain_confidence_list,
                                                                                      domain_accuracy_list,
                                                                                      domain_prob_gen,
                                                                                      confidence_bin_border)
            slot_confidence_list, slot_accuracy_list = record_confidence_accuracy(slot_bspn_right_array,
                                                                                  slot_confidence_list,
                                                                                  slot_accuracy_list,
                                                                                  slot_prob_gen,
                                                                                  confidence_bin_border)
            value_confidence_list, value_accuracy_list = record_confidence_accuracy(value_bspn_right_array,
                                                                                    value_confidence_list,
                                                                                    value_accuracy_list,
                                                                                    value_prob_gen,
                                                                                    confidence_bin_border)

            if T == '1.00' or T == '2.00':
                for threshold_index, confidence_threshold in enumerate(confidence_threshold_list):
                    for word_type in type_list:
                        for i in range(len(eval(word_type + 'prob_gen'))):
                            # 归类至四类中某一类
                            eval(word_type + 'TP_array')[threshold_index], eval(word_type + 'FP_array')[
                                threshold_index], eval(word_type + 'TN_array')[threshold_index], \
                            eval(word_type + 'FN_array')[
                                threshold_index] = classify(eval(word_type + 'bspn_right_array')[i],
                                                            eval(word_type + 'prob_gen')[i],
                                                            eval(word_type + 'TP_array')[threshold_index],
                                                            eval(word_type + 'FP_array')[threshold_index],
                                                            eval(word_type + 'TN_array')[threshold_index],
                                                            eval(word_type + 'FN_array')[
                                                                threshold_index])

            turn_index += 1
        result_index += turn_index

    # 计算ECE
    domain_accuracy_array, domain_confidence_array, domain_bin_sum, domain_ECE_list = compute_ECE(
        domain_accuracy_list,
        domain_confidence_list,
        bin_num, T,
        domain_ECE_list,
        domain_position +
        type_list[0])
    slot_accuracy_array, slot_confidence_array, slot_bin_sum, slot_ECE_list = compute_ECE(slot_accuracy_list,
                                                                                          slot_confidence_list,
                                                                                          bin_num, T,
                                                                                          slot_ECE_list,
                                                                                          domain_position +
                                                                                          type_list[
                                                                                              1])
    value_accuracy_array, value_confidence_array, value_bin_sum, value_ECE_list = compute_ECE(value_accuracy_list,
                                                                                              value_confidence_list,
                                                                                              bin_num, T,
                                                                                              value_ECE_list,
                                                                                              domain_position +
                                                                                              type_list[2])
##
for word_type in type_list:
    # print(eval(word_type + "_ECE_list"))
    write_ECE_to_excel(eval(word_type + "ECE_list"), domain_position + word_type)
    plot_ECE_figure(domain_position + word_type)
    plot_alarm_missing_curve("T:" + T_list[0] + ' ' + word_type, eval(word_type + 'FN_array'),
                             eval(word_type + 'FP_array'), eval(word_type + 'TN_array'), eval(word_type + 'TP_array'))
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
