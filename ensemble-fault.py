from __future__ import absolute_import
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import math
import glob
from dataset import cut_CWRU_data,read_CWRU_data
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost
import os
# 指定使用0,1,2三块卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = 'data/cwru'
save_path = 'data'

# 该函数综合和CRWU数据的读取源数据，建立标签，切割数据
def read_and_save_CWRU_data(data_path, save_path, origin_index, first_index, second_index, cut_size=1024,
                            whether_use_normal=1):
    # 四级文件夹
    origin_level_document = ['/12K_Drive_End', '/48K_Drive_End', '/Fan_End']  # 该驱动端和风扇表示轴承的故障位置
    first_level_document = ['/1797', '/1772', '/1750', '/1730']
    second_level_document = ['/7', '/14', '/21', '/28']
    third_level_document = ['/normal/', '/ball/', '/inner_ring/', '/outer_ring_3/', '/outer_ring_6/', '/outer_ring_12/']
    if whether_use_normal == 0:
        third_level_document = ['/ball/', '/inner_ring/', '/outer_ring_3/', '/outer_ring_6/', '/outer_ring_12/']

    # 建立文件夹路径，循环调用-必须在函数输入中提供各级文件夹的索引序号，origin_index=0, first_index=0, second_index=0,
    read_category = data_path + origin_level_document[origin_index] + first_level_document[first_index] + \
                    second_level_document[second_index]

    # 准备读取数据
    dataset = []
    label = []
    for i in range(len(third_level_document)):  # 遍历最后一级文件夹
        read_path = read_category + third_level_document[i]  # 路径拼接
        fnames = glob.glob(read_path + '*.mat')  # 路径下的所有*.mat文件名
        for file_name in fnames:  # 遍历文件名字

            # 给定文件名，按列标签读取
            if whether_use_normal == 0:
                read_data = read_CWRU_data(file_name, channel=3)
            else:
                read_data = read_CWRU_data(file_name, channel=2)

            # 分割
            read_data, read_label = cut_CWRU_data(read_data, i, cut_size=cut_size)

            # 记录
            if dataset == []:
                dataset = read_data
                label.extend(read_label)
                continue
            dataset = np.concatenate((dataset, read_data), axis=0)  # 按样本维度扩展
            label.extend(read_label)  # 标签文件继续扩展

    # 保存
    dataset_save_name = save_path + '/CWRU_dataset_' + str(cut_size) + '.npy'
    label_save_name = save_path + '/CWRU_label_' + str(cut_size) + '.npy'
    return dataset,label

dataset = np.load('data/CWRU_dataset_1024.npy')
label = np.load('data/CWRU_label_1024.npy')
dataset = dataset[:,:,0]  # DE_time
label = label.reshape(-1)


X_train, X_test, y_train, y_test = train_test_split(dataset, label, random_state=0, test_size=0.3)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)



tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))   # 1.000/0.397

# 生成500个决策树，详细的参数建议参考官方文档
bag = BaggingClassifier(base_estimator=tree, n_estimators=200, max_samples=1.0, max_features=1.0, bootstrap=True,
                        bootstrap_features=False, n_jobs=1, random_state=1)

bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print('Bagging train/test accuracies %.3f/%.3f' % (bag_train, bag_test))   # 1.00/0.7386

rf = RandomForestClassifier(n_estimators=1000, max_features='sqrt', max_depth=None, min_samples_split=2, bootstrap=True,
                            n_jobs=2, random_state=1)

# 度量随机森林的准确性
rf = rf.fit(X_train, y_train)
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Random Forest train/test accuracies %.3f/%.3f' % (tree_train, tree_test))   #  1.00/0.799

ada = AdaBoostClassifier(n_estimators=1000, learning_rate=0.1, random_state=0)
ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print('AdaBoost train/test accuracies %.3f/%.3f' % (ada_train, ada_test))   # 0.535/0.229

gbdt = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, random_state=0)
gbdt = gbdt.fit(X_train, y_train)
y_train_pred = gbdt.predict(X_train)
y_test_pred = gbdt.predict(X_test)
gbdt_train = accuracy_score(y_train, y_train_pred)
gbdt_test = accuracy_score(y_test, y_test_pred)
print('GBDT train/test accuracies %.3f/%.3f' % (gbdt_train, gbdt_test))


xgb = xgboost.XGBClassifier(n_estimators=1000, learning_rate=0.1,tree_method='gpu_hist')
xgb = xgb.fit(X_train, y_train)
y_train_pred = xgb.predict(X_train)
y_test_pred = xgb.predict(X_test)
xgb_train = accuracy_score(y_train, y_train_pred)
xgb_test = accuracy_score(y_test, y_test_pred)
print('XGBoost train/test accuracies %.3f/%.3f' % (xgb_train, xgb_test))  # 1.000/0.581

