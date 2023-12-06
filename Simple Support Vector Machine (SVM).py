# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:48:06 2023

@author: Jose Cruz TP
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100)

print(len(digits.data))


x,y = digits.data[:-10], digits.target[:-10]
clf.fit(x,y)

print('prediction:',clf.predict(digits.data[-6]))

plt.imshow(digits.images[-1],cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
