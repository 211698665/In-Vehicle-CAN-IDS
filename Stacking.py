#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from mlxtend.classifier import StackingClassifier
from xgboost import XGBClassifier as xgbc
from catboost import CatBoostClassifier
import pandas as pd


def classification(train_path, test_path):
    train_data = pd.read_csv(train_path)
    y_train = train_data['Lable']
    x_train = train_data.drop(columns='Lable')
    test_data = pd.read_csv(test_path)
    y_test = test_data['Lable']
    x_test = test_data.drop(columns='Lable')
    print(pd.DataFrame(y_train).value_counts())
    print(pd.DataFrame(y_test).value_counts())
    cat = CatBoostClassifier(iterations=800, depth=6, learning_rate=0.05)
    lgbm = LGBMClassifier(n_estimators=160, max_depth=5, learning_rate=0.08)
    xgb = xgbc(n_estimators=230, max_depth=4, learning_rate=0.1)
    mlp = MLPClassifier(hidden_layer_sizes=(200,), learning_rate=0.05)
    stacking = StackingClassifier(classifiers=[xgb, lgbm, cat],
                                  use_probas=True,
                                  average_probas=False,
                                  meta_classifier=mlp)
    stacking.fit(x_train, y_train)
    y_pred = stacking.predict(x_test)
    print("stacking", classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    cwd = os.getcwd()
    train = cwd + "/train_data.csv"
    test = cwd + "/test_data.csv"
    classification(train, test)
