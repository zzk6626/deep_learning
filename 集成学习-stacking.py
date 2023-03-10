from __future__ import division
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    np.random.seed(0)  # seed to shuffle the train set
    n_folds = 10
    verbose = True
    shuffle = False
    data = load_breast_cancer()
    X = data.data
    y = data.target
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

    if shuffle:
        idx = np.random.permutation(y_train.size)
        X_train = X_train[idx]
        y_train = y_train[idx]

    skf = StratifiedKFold(y_train, n_folds)

    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=100)]

    print("Creating train and test sets for stacking.")

    dataset_blend_train = np.zeros((X_train.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_test.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print(j, clf)
        dataset_blend_test_j = np.zeros((X_test.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print("Fold", i)
            X_train_b = X_train[train]
            y_train_b = y_train[train]
            X_test_b = X_train[test]
            y_test_b = y_train[test]
            clf.fit(X_train_b, y_train_b)
            y_submission = clf.predict_proba(X_test_b)[:, 1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_test)[:, 1]
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    print("Stacking.")
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y_train)
    print("Stacking Accuracy %0.6f:" % accuracy_score(y_test, clf.predict(dataset_blend_test)))
    n = 1
    for model in clfs:
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        print("模型%d,Accuracy %0.6f:" % (n, accuracy_score(y_test, y_test_pred)))
        n = n + 1