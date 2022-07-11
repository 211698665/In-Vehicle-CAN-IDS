import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
import numpy as np


# 获取初始数据集
def ori_data(path):
    allData = pd.read_csv(path)
    print(allData['Lable'].value_counts())
    numeric_features = ['Timestamp', 'Arbitration_ID', 'Data0', 'Data1', 'Data2',
                        'Data3', 'Data4', 'Data5', 'Data6', 'Data7']
    # 归一化
    allData[numeric_features] = allData[numeric_features].apply(
        lambda xi: (xi - xi.min()) / (xi.max() - xi.min()))
    # 删除空数据
    allData = allData.dropna(axis=0)
    y = allData['Lable']
    x = allData.drop(columns='Lable')
    # 划分初始训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    test_data = pd.concat([pd.DataFrame(x_test), pd.DataFrame(y_test)], axis=1)
    test_data.columns = ['Timestamp', 'Arbitration_ID', 'DLC', 'Data0', "Data1", "Data2", "Data3", "Data4", "Data5",
                         "Data6", "Data7", "Lable"]
    test_data.to_csv("temp_test.csv",index=False)
    print("训练", pd.DataFrame(y_train).value_counts())
    print("测试", pd.DataFrame(y_test).value_counts())
    return x_train, x_test, y_train, y_test


# 数据处理
def kmeans_res(x_train, y_train):
    df = pd.concat([pd.DataFrame(x_train), pd.DataFrame(y_train)], axis=1)
    # 平均数
    meanNum = y_train.shape[0] // 4
    # 少数类,SMOTE生成
    df_minor_temp = df[(df['Lable'] == 1) | (df['Lable'] == 2) | (df['Lable'] == 3)]  # 少数类,生成
    x_minor, y_minor = SMOTE(random_state=0, sampling_strategy={1: meanNum, 2: meanNum, 3: meanNum}).fit_resample(
        df_minor_temp.drop(['Lable'], axis=1), np.ravel(df_minor_temp.iloc[:, -1].values.reshape(-1, 1)))
    df_minor = pd.concat([pd.DataFrame(x_minor), pd.DataFrame(y_minor)], axis=1)
    df_minor.columns = ['Timestamp', 'Arbitration_ID', 'DLC', 'Data0', "Data1", "Data2", "Data3", "Data4", "Data5",
                        "Data6", "Data7", "Lable"]
    # 多数类,聚类采样
    df_major = df.drop(df_minor_temp.index)
    # 使用kemans聚类进行数据筛选
    x_major = df_major.drop(['Lable'], axis=1)
    y_major = df_major.iloc[:, -1].values.reshape(-1, 1)
    y_major = np.ravel(y_major)
    # 批量聚类
    kmeans = MiniBatchKMeans(n_clusters=meanNum, random_state=0).fit(x_major)
    # 获取聚类中心
    temp_major = kmeans.cluster_centers_
    temp_major = pd.DataFrame(temp_major)
    temp_major['Lable'] = 0
    temp_major.columns = ['Timestamp', 'Arbitration_ID', 'DLC', 'Data0', "Data1", "Data2", "Data3", "Data4", "Data5",
                          "Data6", "Data7", "Lable"]
    df_res = pd.concat([df_minor, temp_major], axis=0)
    df_res = df_res.dropna(axis=0)
    # 数据清洗,可设置清洗参数
    x_res, y_res = TomekLinks(sampling_strategy="all").fit_resample(df_res.drop(['Lable'], axis=1),
                                                                    np.ravel(df_res.iloc[:, -1].values.reshape(-1, 1)))
    df_res = pd.concat([pd.DataFrame(x_res), pd.DataFrame(y_res)], axis=1)
    df_res.columns = ['Timestamp', 'Arbitration_ID', 'DLC', 'Data0', "Data1", "Data2", "Data3", "Data4", "Data5",
                      "Data6", "Data7", "Lable"]
    df_res.to_csv("temp_train.csv",index=False)


if __name__ == "__main__":
    cwd = os.getcwd()
    myData = cwd + "/data/all.csv"
    x_train, x_test, y_train, y_test = ori_data(myData)  # 原始数据
    kmeans_res(x_test, y_test)  # 平衡数据
