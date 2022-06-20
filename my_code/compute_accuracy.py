# 此为先计算accuracy的版本
# 该方案已被弃之不用

import numpy as np
import json
import math
import pandas as pd
import matplotlib.pyplot as plt

# 读取json文件内容,返回字典格式
with open('result.json', 'r', encoding='utf8')as fp:
    result = json.load(fp)

accuracy_list = []
confidence_list = []
confidence_num_list = []
batch_size = 5
result_index = 0

# 最终不能超出result的长度
# 对batch的循环
while result_index < len(result):
    # batch内的循环
    batch_index = 0
    positive_count = 0
    negative_count = 0
    confidence_num = 0
    confidence_sum = 0
    while batch_index < batch_size:
        # dialog内的循环
        turn_index = 0
        dialog_begin = result_index
        while turn_index < result[dialog_begin]['turn_num']:
            result_index += 1
            turn_index += 1
            bspn = result[result_index]['bspn']
            print('result_index:', result_index)
            print('turn_index:', turn_index)
            bspn_tokens = result[result_index]['bspn_tokens']
            bspn_prob = result[result_index]['bspn_prob']
            bspn_tokens = eval(bspn_tokens)
            bspn_prob = eval(bspn_prob)
            # 消除\u字符
            for i in range(len(bspn_tokens)):
                if "\u0120" in bspn_tokens[i]:
                    bspn_tokens[i] = bspn_tokens[i][1:]
            # 将第一个token及其概率去掉
            bspn_tokens = bspn_tokens[1:-1]
            bspn_prob = bspn_prob[1:-1]
            print(bspn)
            print(bspn_tokens)
            print(bspn_prob)
            # 对每一个turn计算正例与负例
            for i, bspn_token in enumerate(bspn_tokens):
                if bspn_token in bspn:
                    positive_count += 1
                else:
                    negative_count += 1
                confidence_num += 1
                confidence_sum += bspn_prob[i]
        result_index += 1
        batch_index += 1
        if result_index >= len(result):
            break
    print('confidence_num:', confidence_num)
    # 计算accuracy
    cur_accuracy = positive_count / (positive_count + negative_count)
    accuracy_list.append(cur_accuracy)
    # 计算confidence
    cur_confidence = confidence_sum / confidence_num
    confidence_list.append(cur_confidence)
    confidence_num_list.append(confidence_num)
    print("accuracy_list:", accuracy_list)
    print("confidence_list:", confidence_list)
    print("confidence_num_list", confidence_num_list)
    print("length of confidence_list:", len(confidence_list))
    print("length of accuracy_list:", len(accuracy_list))
##
# 计算ECE
confidence_array = np.array(confidence_list)
accuracy_array = np.array(accuracy_list)
confidence_num_array = np.array(confidence_num_list)
temp_array = abs(confidence_array - accuracy_array)
temp_array = temp_array * confidence_num_array
ECE = temp_array.sum() / confidence_num_array.sum()
print('ECE:', ECE)

##
# 因为没有0-0.5的accuracy,就没有绘制
low = 0.8
high = 1
bin_num = 10
accuracy_bin = np.zeros(bin_num)
confidence_bin = np.zeros(bin_num)
df_confidence = pd.DataFrame(confidence_list)
df_accuracy = pd.DataFrame(accuracy_list)
tick_list = []
precision = (high - low) / bin_num
bin_sum = []
# 左开右闭区间
for i in range(bin_num):
    left = low + i * precision
    right = low + (i + 1) * precision
    tick_list.append(round((left + right) / 2, 4))
    is_in_bin = df_accuracy[df_accuracy > left]
    is_in_bin = (is_in_bin <= right)
    print("is_in_bin:", is_in_bin.sum())
    confidence_bin[i] = (is_in_bin[0] * df_confidence[0]).sum() / is_in_bin[0].sum()
    accuracy_bin[i] = (is_in_bin[0] * df_accuracy[0]).sum() / is_in_bin[0].sum()
    bin_sum.append(is_in_bin[0].sum())
print("confidence_bin:", confidence_bin)
print("accuracy_bin", accuracy_bin)
print("bin_sum:", bin_sum)
plt.bar(range(len(confidence_bin)), confidence_bin, label='confidence', fc='r', tick_label=tick_list)
plt.bar(range(len(accuracy_bin)), accuracy_bin, label='accuracy', fc='b', tick_label=tick_list)
plt.xlabel('bin')
plt.title('The comparison of accuracy and confidence')
plt.legend()
plt.show()
##
import openpyxl
# 写出数据
data = pd.DataFrame()
data['confidence_bin'] = confidence_bin
data['accuracy_bin'] = accuracy_bin
data['bin_sum'] = bin_sum
data.to_excel('data1.xlsx')