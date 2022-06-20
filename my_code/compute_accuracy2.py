import numpy as np
import json
import math
import pandas as pd
import matplotlib.pyplot as plt

# 读取json文件内容,返回字典格式
with open('result.json', 'r', encoding='utf8')as fp:
    result = json.load(fp)

# 记录confidence与accuracy的列表
low = 0.99
high = 1
bin_num = 20
confidence_list = [[] for i in range(bin_num)]
accuracy_list = [[] for i in range(bin_num)]
tick_list = [round(low + i * (high - low) / bin_num, 4) for i in range(bin_num)]
confidence_bin_border = pd.DataFrame(tick_list)
print(confidence_bin_border)

# 按对话循环,用索引实现
result_index = 0
while result_index < len(result):
    turn_index = 1
    while turn_index <= result[result_index]['turn_num']:
        bspn = result[result_index + turn_index]['bspn']
        bspn_tokens = result[result_index + turn_index]['bspn_tokens']
        bspn_prob = result[result_index + turn_index]['bspn_prob']
        bspn_tokens = eval(bspn_tokens)
        bspn_prob = eval(bspn_prob)
        # 消除\u字符
        for i in range(len(bspn_tokens)):
            if "\u0120" in bspn_tokens[i]:
                bspn_tokens[i] = bspn_tokens[i][1:]
        # 将第一个token及其概率去掉
        bspn_tokens = bspn_tokens[1:-1]
        bspn_prob = bspn_prob[1:-1]
        # print(bspn)
        # print(bspn_tokens)
        # print(bspn_prob)
        # 得到confidence所在区间
        for i in range(len(bspn_prob)):
            temp_list = confidence_bin_border < bspn_prob[i]
            confidence_bin_position = (temp_list.sum() - 1)[0]
            confidence_list[confidence_bin_position].append(bspn_prob[i])
            positive_negative_flag = (bspn_tokens[i] in bspn)
            accuracy_list[confidence_bin_position].append(positive_negative_flag)
        turn_index += 1
    result_index += turn_index
##
# 计算ECE
accuracy_array = np.array([sum(accuracy_list[i]) / len(accuracy_list[i]) if len(accuracy_list[i]) != 0 else 0 for i in
                           range(bin_num)]).squeeze()
accuracy_num_array = np.array([len(accuracy_list[i]) for i in range(bin_num)]).squeeze()
confidence_array = np.array(
    [sum(confidence_list[i]) / len(confidence_list[i]) if len(confidence_list[i]) != 0 else 0 for i in
     range(bin_num)]).squeeze()
print(accuracy_array)
print(confidence_array)
ECE = sum(abs(confidence_array - accuracy_array) * accuracy_num_array) / sum(accuracy_num_array)
print("ECE:", ECE)
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

##
print("accuracy_num_array =", accuracy_num_array)
##


'''
    动态规划——字符串的编辑距离
    s1 = "abc", s2 = "def"
    计算公式：
             | 0                                           i = 0, j = 0
             | j                                           i = 0, j > 0
    d[i,j] = | i                                           i > 0, j = 0
             | min(d[i,j-1]+1, d[i-1,j]+1, d[i-1,j-1])    s1(i) = s2(j)
             | min(d[i,j-1]+1, d[i-1,j]+1, d[i-1,j-1]+1)  s1(i) ≠ s2(j)
    定义二维数组[4][4]：
        d e f            d e f
    |x|x|x|x|        |0|1|2|3|
    a |x|x|x|x|  =>  a |1|1|2|3|  => 编辑距离d = [4][4] = 3
    b |x|x|x|x|      b |2|2|2|3|
    c |x|x|x|x|      c |3|3|3|3|
'''


def levenshtein(s1, s2):
    i = 0  # s1字符串中的字符下标
    j = 0  # s2字符串中的字符下标
    s1i = ""  # s1字符串第i个字符
    s2j = ""  # s2字符串第j个字符
    m = len(s1)  # s1字符串长度
    n = len(s2)  # s2字符串长度
    # 返回一个表示s2对应位置是否正确的向量
    s2_right_array = np.zeros(n) + 1
    if m == 0:
        return n  # s1字符串长度为0，此时的编辑距离就是s2字符串长度
    if n == 0:
        return m  # s2字符串长度为0，此时的编辑距离就是s1字符串长度
    solutionMatrix = [[0 for col in range(n + 1)] for row in range(m + 1)]  # 长为m+1，宽为n+1的矩阵
    '''
             d e f
          |0|x|x|x|
        a |1|x|x|x|
        b |2|x|x|x|
        c |3|x|x|x|
    '''
    for i in range(m + 1):
        solutionMatrix[i][0] = i
    '''
             d e f
          |0|1|2|3|
        a |x|x|x|x|
        b |x|x|x|x|
        c |x|x|x|x|

    '''
    for j in range(n + 1):
        solutionMatrix[0][j] = j
    '''
        上面两个操作后，求解矩阵变为
             d e f
          |0|1|2|3|
        a |1|x|x|x|
        b |2|x|x|x|
        c |3|x|x|x|
        接下来就是填充剩余表格
    '''
    for x in range(1, m + 1):
        s1i = s1[x - 1]
        for y in range(1, n + 1):
            s2j = s2[y - 1]
            flag = 0 if s1i == s2j else 1
            solutionMatrix[x][y], operation = min(solutionMatrix[x - 1][y] + 1, solutionMatrix[x][y - 1] + 1,
                                                  solutionMatrix[x - 1][y - 1] + flag)
            if operation == 'delete' or operation == 'edit':
                s2_right_array[y] = 0

    return solutionMatrix[m][n], s2_right_array


def min(insert, delete, edit):
    if insert < delete:
        tmp, operation = insert, 'insert'
    elif delete < edit:
        tmp, operation = delete, 'delete'
    else:
        tmp, operation = edit, 'edit'
    return tmp, operation


s1 = "abc"
s2 = "def"
distance = levenshtein(s1, s2)
print(distance)
