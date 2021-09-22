# Author: Narth Chin
# Update Date: 2021/09/22
# Coding: UTF-8

from sklearn.datasets import load_breast_cancer
from sklearn.svm import LinearSVC
dataX, dataY = load_breast_cancer(return_X_y=True)
from sklearn.model_selection import train_test_split


def split_data_classfication():
    # 分层采样拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4
    return train_test_split(dataX, dataY, test_size=0.1,
                            random_state=0, stratify=dataY)


# 支持向量机线性分类LinearSVC模型
def Score(*data, tol, verbose, iter, c):
    X_train, X_test, y_train, y_test = data
    lsvc = LinearSVC(tol=tol, C=c, verbose=verbose, max_iter=iter)
    lsvc.fit(X_train, y_train)
    print('Score: %.2f' % lsvc.score(X_test, y_test))


# 生成用于分类的数据集
X_train, X_test, y_train, y_test = split_data_classfication()
# 调用 LinearSVC
Score(X_train, X_test, y_train, y_test, tol=1e-4,
      verbose= 1, c=1e-4, iter=10000)