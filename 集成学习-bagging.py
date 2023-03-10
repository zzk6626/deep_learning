import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 读入乳腺癌数据集
data = load_breast_cancer()
x = pd.DataFrame(data.data)
y = data.target
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3)

print(x.shape)
# 度量单个决策树的准确性
tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))
# Decision tree train/test accuracies 1.000/0.942

# 度量bagging分类器的准确性，必须基于单个决策树做出bagging的设置方法
# 生成500个决策树，详细的参数建议参考官方文档
bag = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True,
                        bootstrap_features=False, n_jobs=1, random_state=1)
bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print('Bagging train/test accuracies %.3f/%.3f' % (bag_train, bag_test))
# Bagging train/test accuracies 1.000/0.959
#
# # 随机森林，bagging思想的拓展变体
rf = RandomForestClassifier(n_estimators=1000, max_features='sqrt', max_depth=None, min_samples_split=2, bootstrap=True,
                            n_jobs=1, random_state=1)
# 度量随机森林的准确性
rf = rf.fit(X_train, y_train)
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Random Forest train/test accuracies %.3f/%.3f' % (tree_train, tree_test))
# RandomFores train/test accuracies 1.000/0.971