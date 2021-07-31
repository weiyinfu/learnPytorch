from collections import Counter

import torch as t
from sklearn.datasets import load_iris


def get_data():
    x, y = load_iris(True)
    print(Counter(y))
    print(x.shape, y.shape)
    x = t.from_numpy(x)
    out = t.empty(len(y), 3)  # 3个类别
    yy = t.LongTensor(y).view(-1, 1)
    out.scatter_(dim=1, index=yy, value=1)
    # print(out)
    return x, t.from_numpy(y)


def train():
    w = t.rand(4, 3, dtype=t.float64)  # 4个属性
    b = t.rand(3, dtype=t.float64)  # 3个类别
    w.requires_grad = True
    b.requires_grad = True

    def model(x: t.Tensor):
        yy = t.sigmoid(t.matmul(x, w) + b)
        y = t.softmax(yy, dim=0)
        # 此处如果不进行softmax则结果不对，为啥
        return y

    optimizer = t.optim.Adam([w, b], lr=1e-4)
    x, y = get_data()
    loss_f = t.nn.CrossEntropyLoss()
    for i in range(100000):
        yy = model(x)
        l = loss_f(yy, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        if i % 1001 == 0:
            y_ = t.argmax(yy, dim=1)
            prec = t.count_nonzero(y_ == y) / len(y)
            print(f"epoch={i} loss={l} prec={prec}")


train()
"""
epoch=96096 loss=1.096520944682032 prec=0.9933333396911621
epoch=97097 loss=1.096520113935385 prec=0.9933333396911621
epoch=98098 loss=1.096519303479361 prec=0.9933333396911621
epoch=99099 loss=1.096518512590337 prec=0.9933333396911621
"""
