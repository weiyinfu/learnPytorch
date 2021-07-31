import matplotlib.pyplot as plt
import torch as t
from tqdm import tqdm


def generate():
    x = t.linspace(-3, 3, 100)
    a, b, c = t.rand(3)
    y = a * x * x + b * x + c
    return a, b, c, x, y


def solve(x, y):
    arg = t.rand(3)
    arg.requires_grad = True
    for i in tqdm(range(10000)):
        dis = t.linalg.norm(arg[0] * x * x + arg[1] * x + arg[2] - y) / len(y)
        if dis < 1e-4:
            break
        dis.backward(retain_graph=True)
        arg.data.sub_(arg.grad * 1e-3)
        # arg.sub_(arg.grad*1e-3)
        arg.grad = None
        if i % 101 == 0:
            print(f"epoch={i} dis={dis} arg={arg.data}")
    return arg.data


def main():
    a, b, c, x, y = generate()
    aa, bb, cc = solve(x, y)
    print(a, b, c)
    print(aa, bb, cc)
    plt.plot(x, y)
    plt.plot(x, aa * x * x + bb * x + cc)
    plt.show()


main()
