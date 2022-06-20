import numpy as np
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
from levenshtein import levenshtein
from utils import sort_words, constraint_dict_to_list, bspan_to_constraint_dict, get_words_prob
import openpyxl

# 读取json文件内容,返回字典格式
with open('result-ood.json', 'r', encoding='utf8')as fp:
    result = json.load(fp)


def get_list_in_result(index, key):
    l = result[index][key]
    l = eval(l)
    return l


def classify(P_or_N, prob, TP, FP, TN, FN, confidence_threshold):
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


# 记录confidence与accuracy的列表
low = 0
high = 1
bin_num = 20
T_num = 40
T_begin = 1
T_end = 3
figure_num = 1
word_type = 'OOD'
T_list = ['{:.2f}'.format(T_begin + i * (T_end - T_begin) / T_num) for i in range(T_num)]
ECE_list = {}
confidence_list = [[] for i in range(bin_num)]
accuracy_list = [[] for i in range(bin_num)]
tick_list = [round(low + i * (high - low) / bin_num, 4) for i in range(bin_num)]
confidence_bin_border = pd.DataFrame(tick_list)
print(confidence_bin_border)
print(T_list)

# 置信度域值
threshold_num = 20
threshold_low = 0.9
threshold_high = 1

confidence_threshold_list = [threshold_low + i * (threshold_high - threshold_low) / threshold_num for i in
                             range(threshold_num)]
TP_array, FP_array, TN_array, FN_array = np.zeros(len(confidence_threshold_list)), np.zeros(
    len(confidence_threshold_list)), np.zeros(len(confidence_threshold_list)), np.zeros(len(confidence_threshold_list))

##
for threshold_index, confidence_threshold in enumerate(confidence_threshold_list):
    T_list = ["1.00"]
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
                # print("len(bspn_tokens_gen):", len(bspn_tokens_gen))
                bspn_words_gen = constraint_dict_to_list(
                    bspan_to_constraint_dict(result[result_index + turn_index]['bspn_gen']))
                # print("bspan_to_constraint_dict(result[result_index + turn_index]['bspn_gen'])",
                #       bspan_to_constraint_dict(result[result_index + turn_index]['bspn_gen']))
                # print("result[result_index + turn_index]['bspn_gen']:", result[result_index + turn_index]['bspn_gen'])
                # print("bspn_words_gen", bspn_words_gen)
                # print("bspn_words_gen:", bspn_words_gen)
                bspn_words_prob_gen = get_words_prob(bspn_words_gen, bspn_tokens_gen, bspn_prob_gen)
                # print("len(bspn_words_prob_gen):", len(bspn_words_prob_gen))

                # 得到排序后对应的概率列表
                bspn_prob_gen = [bspn_words_prob_gen[i] for i in pos_gen_list]
                # print("bspn_prob_gen", bspn_prob_gen)
                if len(bspn_prob_gen) != 0 and len(bspn_prob_gen) == sum(bspn_prob_gen):
                    exit(0)

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

                    # 归类至四类中某一类
                    TP_array[threshold_index], FP_array[threshold_index], TN_array[threshold_index], FN_array[
                        threshold_index] = classify(bspn_right_array[i], bspn_prob_gen[i], TP_array[threshold_index],
                                                    FP_array[threshold_index], TN_array[threshold_index],
                                                    FN_array[threshold_index], confidence_threshold)

                turn_index += 1
            result_index += turn_index

        # 计算ECE
        accuracy_array = np.array(
            [sum(accuracy_list[i]) / len(accuracy_list[i]) if len(accuracy_list[i]) != 0 else 0 for i in
             range(bin_num)]).squeeze()
        accuracy_num_array = np.array([len(accuracy_list[i]) for i in range(bin_num)]).squeeze()
        confidence_array = np.array(
            [sum(confidence_list[i]) / len(confidence_list[i]) if len(confidence_list[i]) != 0 else 0 for i in
             range(bin_num)]).squeeze()
        bin_sum = [len(confidence_list[i]) for i in range(bin_num)]
        print("accuracy_array:", accuracy_array)
        print("confidence_array:", confidence_array)
        ECE_list[str(T)] = sum(abs(confidence_array - accuracy_array) * accuracy_num_array) / sum(accuracy_num_array)
        print("T:", T, " ECE:", ECE_list[str(T)])
##
# 计算missing_alarm与false_alarm
MA_array = FN_array / (TP_array + FN_array)
FA_array = FP_array / (TP_array + FP_array)
plt.plot(MA_array, FA_array)
plt.xlabel('MA')
plt.ylabel('FA')
plt.title('OOD FA-MA figure')
plt.show()
##
data = pd.DataFrame()
data['T'] = ECE_list.keys()
data['ECE'] = ECE_list.values()
data.to_excel(word_type + 'ECE' + str(T_begin) + '_' + str(T_end) + '.xlsx')
##
# 画图
data = pd.read_excel(word_type + 'ECE' + str(T_begin) + '_' + str(T_end) + '.xlsx')
plt.plot(data['T'], data['ECE'])
plt.xlabel('T')
plt.ylabel('ECE')
plt.title('OOD ECE-T figure')
plt.show()
##
# 写出数据
data = pd.DataFrame()
data['confidence_array'] = confidence_array
data['accuracy_array'] = accuracy_array
data['bin_sum'] = bin_sum
data.to_excel(word_type + str(low) + '_' + str(high) + '_data.xlsx')


##
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


plot_acc_con_figure(confidence_array, accuracy_array, word_type)

##
# # 第一次应将键值修整的整洁
# if T == 1.0:
#     result[result_index + turn_index]['bspn_prob_1.7'] = result[result_index + turn_index].pop(
#         'bspn_prob_1.7000000000000002')
#     result[result_index + turn_index]['bspn_prob_1.95'] = result[result_index + turn_index].pop(
#         'bspn_prob_1.9500000000000002')
