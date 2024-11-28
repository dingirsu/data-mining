import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, shapiro
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from numpy.random import uniform
import missingno as msno

def preprocess(file_path):
    data = custom_delimiter(file_path)
    data = data.apply(pd.to_numeric, errors='coerce')
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    return data_scaled.to_numpy()

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

if __name__ == "__main__":
    # 读取数据集
    data = custom_delimiter('data.txt')

    # 使用 df.info 函数分析数据
    print(data.info())

    # 标准化
    # 将所有数据列转换为数值类型
    data = data.apply(pd.to_numeric, errors='coerce')

    # 使用 StandardScaler 进行标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 将标准化后的数据转换为 DataFrame
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)


    # 分布分析
    # 计算每一列数据的偏度
    skewness = data_scaled.apply(lambda x: skew(x.dropna()))
    print("Skewness of each column after normalization:")
    print(skewness)

    # 对每一列数据进行 Shapiro-Wilk 检验
    shapiro_results = data_scaled.apply(lambda x: shapiro(x.dropna())[1])
    print("Shapiro-Wilk test p-values for each column:")
    print(shapiro_results)


    # 相关性分析
    # 计算每一列变量间的相关系数矩阵（皮尔逊相关系数）
    correlation_matrix = data_scaled.corr(method='pearson')
    print("Pearson correlation matrix:")
    print(correlation_matrix)

    # 找出皮尔逊相关系数矩阵中大于0.9的项对应的两列
    high_corr_pairs = [(correlation_matrix.columns[i], correlation_matrix.columns[j]) 
                    for i in range(len(correlation_matrix.columns)) 
                    for j in range(i+1, len(correlation_matrix.columns)) 
                    if abs(correlation_matrix.iloc[i, j]) > 0.9]

    print("Pairs of columns with Pearson correlation coefficient > 0.9:")
    print(high_corr_pairs)


    # 主成分分析
    pca = PCA(n_components=2)  # 选择前两个主成分
    principal_components = pca.fit_transform(data_scaled)

    # 将主成分转换为 DataFrame
    principal_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])

    # 打印主成分分析结果的前几行
    print(principal_df.head())

    # 打印主成分的解释方差
    explained_variance = pca.explained_variance_ratio_
    print("Explained variance of each principal component:")
    print(explained_variance)

    # 绘制主成分分析结果的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(principal_df['Principal Component 1'], principal_df['Principal Component 2'])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Result')
    plt.show()


    # 计算 Hopkins 统计量
    def hopkins(X):
        d = X.shape[1]
        n = len(X)
        m = int(0.1 * n)  # heuristic from the article [1]
        nbrs = NearestNeighbors(n_neighbors=1).fit(X)
        rand_X = uniform(X.min(axis=0), X.max(axis=0), (m, d))
        ujd = []
        wjd = []
        for j in range(0, m):
            u_dist, _ = nbrs.kneighbors(rand_X[j].reshape(1, -1), 2, return_distance=True)
            ujd.append(u_dist[0][1])
            random_index = np.random.randint(0, n)
            w_dist, _ = nbrs.kneighbors(X.iloc[random_index].values.reshape(1, -1), 2, return_distance=True)
            wjd.append(w_dist[0][1])
        H = sum(ujd) / (sum(ujd) + sum(wjd))
        return H

    hopkins_stat = hopkins(data_scaled)
    print("Hopkins statistic:", hopkins_stat)
