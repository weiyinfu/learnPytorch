import matplotlib.pyplot as plt
import torch as t

"""
使用optimizer进行优化
"""


def generate():
    x = t.linspace(-3, 3, 100)
    a, b, c = t.rand(3)
    y = a * x ** 2 + b * x + c
    return a, b, c, x, y


def solve(x, y):
    arg = t.rand(3)
    arg.requires_grad = True
    optimizer = t.optim.Adam([arg], lr=1e-3)
    loss_f = t.nn.MSELoss()
    for i in range(10000):
        output = arg[0] * x ** 2 + arg[1] * x + arg[2]
        loss = loss_f(output, y)
        if loss < 1e-3:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 101 == 0:
            print(f"epoch={i} loss={loss}")
    return arg.data


def gel(x, y):
    # 基于最小二乘法求解一元二次方程参数
    A = t.column_stack([x ** 2, x, t.ones(len(x))])
    ans, _ = t.lstsq(y, A)
    return ans[:3]


def main():
    a, b, c, x, y = generate()
    aa, bb, cc = solve(x, y)
    aaa, bbb, ccc = gel(x, y)
    print("标准答案", a, b, c)
    print("迭代方法", aa, bb, cc)
    print("最小二乘法", aaa, bbb, ccc)
    plt.plot(x, y)
    plt.plot(x, aa * x * x + bb * x + cc)
    # plt.show()


main()
