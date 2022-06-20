import numpy as np
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
from levenshtein import levenshtein
import openpyxl


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


# 读取json文件内容,返回字典格式
with open('result_1_3.json', 'r', encoding='utf8')as fp:
    result = json.load(fp)

# 记录confidence与accuracy的列表
low = 0
high = 1
bin_num = 20
T_num = 40
T_begin = 1
T_end = 3
T_list = ['{:.2f}'.format(T_begin + i * (T_end - T_begin) / T_num) for i in range(T_num)]
ECE_list = {}
confidence_list = [[] for i in range(bin_num)]
accuracy_list = [[] for i in range(bin_num)]
tick_list = [round(low + i * (high - low) / bin_num, 4) for i in range(bin_num)]
confidence_bin_border = pd.DataFrame(tick_list)
print(confidence_bin_border)
print(T_list)

##
figure_num = 1
#T_list = ["1.00"]
# 提取出每一个步进点的温度
for T in T_list:
    bspn_prob_index = 'bspn_prob_' + str(T)
    # 按对话循环,用索引实现
    result_index = 0
    while result_index < len(result):
        turn_index = 1
        while turn_index <= result[result_index]['turn_num']:
            bspn_tokens = result[result_index + turn_index]['bspn_tokens']
            bspn_tokens_gen = result[result_index + turn_index]['bspn_tokens_gen']
            # # 第一次应将键值修整的整洁
            # if T == 1.0:
            #     result[result_index + turn_index]['bspn_prob_1.7'] = result[result_index + turn_index].pop(
            #         'bspn_prob_1.7000000000000002')
            #     result[result_index + turn_index]['bspn_prob_1.95'] = result[result_index + turn_index].pop(
            #         'bspn_prob_1.9500000000000002')
            bspn_prob_gen = result[result_index + turn_index][bspn_prob_index]
            bspn_tokens = eval(bspn_tokens)
            bspn_tokens_gen = eval(bspn_tokens_gen)
            bspn_prob_gen = eval(bspn_prob_gen)
            # 消除\u字符
            for i in range(len(bspn_tokens)):
                if "\u0120" in bspn_tokens[i]:
                    bspn_tokens[i] = bspn_tokens[i][1:]
            for i in range(len(bspn_tokens_gen)):
                if "\u0120" in bspn_tokens_gen[i]:
                    bspn_tokens_gen[i] = bspn_tokens_gen[i][1:]
            # 将第一个token及其概率去掉
            bspn_tokens = bspn_tokens[1:-1]
            bspn_tokens_gen = bspn_tokens_gen[1:-1]
            bspn_prob_gen = bspn_prob_gen[1:-1]
            # print(bspn_tokens)
            # print(bspn_tokens_gen)
            # print(bspn_prob_gen)
            # 得到confidence所在区间
            distance, bspn_right_array = levenshtein(bspn_tokens, bspn_tokens_gen)
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
    print(accuracy_array)
    print(confidence_array)
    ECE_list[str(T)] = sum(abs(confidence_array - accuracy_array) * accuracy_num_array) / sum(accuracy_num_array)
    print("T:", T, " ECE:", ECE_list[str(T)])
    if T == '1.00':
        plot_acc_con_figure(confidence_array, accuracy_array, 'tokens_unsorted_levenshtein')

##
data = pd.DataFrame()
data['T'] = ECE_list.keys()
data['ECE'] = ECE_list.values()
data.to_excel('ECE' + str(T_begin) + '_' + str(T_end) + '.xlsx')
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
tick_list = [round(low + i * (high - low) / bin_num, 4) for i in range(bin_num)]
plt.bar(range(len(confidence_array)), confidence_array, label='confidence', fc='r', tick_label=tick_list)
plt.bar(range(len(accuracy_array)), accuracy_array, label='accuracy', fc='b', tick_label=tick_list)
plt.xlabel('bin')
plt.title('The comparison of accuracy and confidence')
plt.legend()
plt.show()

print("accuracy_num_array =", accuracy_num_array)

##
