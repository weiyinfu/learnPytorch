from collections import Counter

import matplotlib.pyplot as plt

import mnist

"""
MNIST 训练数据60000，测试数据10000
10个类别，32*32*1的图片
"""
ds = mnist.Mnist.train

train_data = mnist.Mnist.train
test_data = mnist.Mnist.test
print("train_x_size", train_data.data.size(),
      '\ntrain_y_size', train_data.targets.size(),
      "\ntest_x_size", test_data.data.size(),
      "\ntest_y_size", test_data.targets.size(),
      "\ntype_train_x", type(train_data.data),
      "\ntype_train_y", type(train_data.targets),
      "\ntype_test_x", type(test_data.data),
      "\ntype_test_y", type(test_data.targets),
      )
