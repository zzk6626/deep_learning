import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost
from sklearn.metrics import accuracy_score

# 读入乳腺癌数据集
data = load_breast_cancer()
x = pd.DataFrame(data.data)
y = data.target
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3)
print(y_test)
# 度量单个决策树的准确性
tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))
# Decision tree train/test accuracies 1.000/0.942

# Boosting分类器准确性
ada = AdaBoostClassifier(n_estimators=1000, learning_rate=0.1, random_state=0)
ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print('AdaBoost train/test accuracies %.3f/%.3f' % (ada_train, ada_test))
# AdaBoost train/test accuracies 1.000/0.977

gbdt = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, random_state=0)
gbdt = gbdt.fit(X_train, y_train)
y_train_pred = gbdt.predict(X_train)
y_test_pred = gbdt.predict(X_test)
gbdt_train = accuracy_score(y_train, y_train_pred)
gbdt_test = accuracy_score(y_test, y_test_pred)
print('GBDT train/test accuracies %.3f/%.3f' % (gbdt_train, gbdt_test))
# GBDT train/test accuracies 1.000/0.982

xgb = xgboost.XGBClassifier(n_estimators=1000, learning_rate=0.1)
xgb = xgb.fit(X_train, y_train)
y_train_pred = xgb.predict(X_train)
y_test_pred = xgb.predict(X_test)
xgb_train = accuracy_score(y_train, y_train_pred)
xgb_test = accuracy_score(y_test, y_test_pred)
print('XGBoost train/test accuracies %.3f/%.3f' % (xgb_train, xgb_test))
# XGBoost train/test accuracies 1.000/0.982