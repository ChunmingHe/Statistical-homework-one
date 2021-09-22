# Author: Narth Chin
# Update Date: 2021/09/22
# Coding: UTF-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from matplotlib.colors import ListedColormap
import seaborn as sns

plt.rcParams['savefig.dpi'] = 1200  # 图片像素
plt.rcParams['figure.dpi'] = 600  # 分辨率


def get_fakenews():
    try:
        fake_news = pd.read_csv('fake.csv', usecols=[4], dtype='string')
        fake_news = fake_news.dropna()  # 滤除缺失数据
        fake_news = fake_news.sample(1298, ignore_index=True)  # 随机提取指定数量样本
        fake_news = fake_news['title'].tolist()
        # print(fake_news)
        return fake_news
    except IOError:
        print('oops!')


def get_realnews():
    try:
        real_news = pd.read_csv('abcnews-date-text.csv', usecols=[1], dtype='string')
        real_news = real_news.dropna()  # 滤除缺失数据
        real_news = real_news.sample(1968, ignore_index=True)  # 随机提取指定数量样本
        real_news = real_news['headline_text'].tolist()
        # print(real_news)
        return real_news
    except IOError:
        print('oops!')


def make_label():  # 为样本指派标签
    label = range(3266)
    label = list(label)
    for i in range(1968):
        label[i] = 1
    for i in range(1298):
        label[i + 1968] = 0
    # print(label)
    return label


def get_class_name(plist):
    for i in range(len(plist)):
        if plist[i] == 1:
            plist[i] = 'Real news'
        else:
            plist[i] = 'Fake news'


def main():
    fake_news = get_fakenews()
    real_news = get_realnews()
    all_news = real_news + fake_news
    # print(len(all_news))

    cv = CountVectorizer()  # 生成词频矩阵
    cv_fit = cv.fit_transform(all_news).toarray()
    cv_np = np.array(cv_fit, dtype='float32')
    # print(cv_np)

    label = make_label()
    svd = TruncatedSVD(2)  # 截断奇异值分解降维，可增强算法鲁棒性
    cv_np = svd.fit_transform(cv_np)

    X_train, X_test, y_train, y_test = train_test_split(cv_np,
                                                        label,
                                                        test_size=0.10,
                                                        random_state=42,
                                                        stratify=label)

    acc_val = []
    x_val = range(2, 33)
    for k in range(2, 33):  # 重复实验获取表现最优的K值
        neigh = KNeighborsClassifier(n_neighbors=k)
        crval = cross_val_score(neigh, X_train, y_train, cv=10)
        crvalmean = np.mean(crval)
        print('The mean cross validation accuracy of %s neighbors: %.4f' % (k, crvalmean))
        acc_val.append(crvalmean)

    plt.plot(x_val, acc_val, '.-')
    plt.xlabel('K-Numbers')
    plt.ylabel('Score')
    plt.title('Validation Score')
    plt.savefig('plot_nice.pdf')
    plt.show()

    # 决策分界线的打印
    bestk = acc_val.index(max(acc_val))
    print('Best K-value: %.0f' % bestk)
    neigh = KNeighborsClassifier(n_neighbors=bestk)
    neigh.fit(X_train, y_train)
    print('Best k-Score: %.4f' % neigh.score(X_test, y_test))
    cmap_light = ListedColormap(['orange', 'cyan'])
    cmap_bold = ['darkorange', 'c']
    h = .02
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = neigh.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    get_class_name(y_train)

    # 在决策面上打印出训练集点位
    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train[:],
                    palette=cmap_bold, alpha=1.0, edgecolor="black")
    plt.xlim(0.35, 1)
    plt.ylim(-1.2, -0.5)
    plt.title("2-Class classification (Partial example)")
    plt.savefig('scatter_nice.pdf')
    plt.show()


if __name__ == "__main__":
    main()
