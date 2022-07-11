import os
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

def feature_importance_cal(path):
    allData = pd.read_csv(path)
    y = allData['Lable']
    x = allData.drop(columns='Lable')
    grd = GradientBoostingClassifier(n_estimators=30)
    grd.fit(x, y)
    l = []
    x = 0
    columns = ['Timestamp', 'Arbitration_ID', 'DLC', 'Data0', "Data1", "Data2", "Data3", "Data4", "Data5",
               "Data6", "Data7", "Lable"]
    for i in grd.feature_importances_:
        # 评分较低的特征
        if i < 0.005:
            l.append(columns[x])
        x += 1
    return l


def feature_select(l, train_data, test_data):
    train_data = pd.read_csv(train_data)
    test_data = pd.read_csv(test_data)
    train_data = train_data.drop(l, axis=1)
    test_data = test_data.drop(l, axis=1)
    train_data.to_csv("train_data.csv",index=False)
    test_data.to_csv("test_data.csv",index=False)


if __name__ == "__main__":
    cwd = os.getcwd()
    my_data = cwd + "/temp_train.csv"
    l = feature_importance_cal(my_data)
    test_data = cwd + "/temp_test.csv"
    train_data = cwd + "/temp_train.csv"
    feature_select(l, train_data, test_data)
