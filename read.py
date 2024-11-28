import pandas as pd
import re

# 自定义分隔符函数
def custom_delimiter(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        # 使用正则表达式分隔数据
        split_line = re.split(r'  +| (?=-)', line.strip())
        data.append(split_line)
    return pd.DataFrame(data)

# 读取数据集
data = custom_delimiter('X_train.txt')

# 打印数据集的前几行
for i in range(15):
    print(data.iloc[0,i])
