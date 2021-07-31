import torch as t

"""
保存模型的两种方法：一种是使用state_dict
一种是使用裸的save和load

这两种方法都需要torch.save()和torch.load()
"""
class X(t.nn.Module):
    def __init__(self):
        super().__init__()


x = X()


def one():
    t.save(x, 'haha.t')
    t.load('haha.t')


def two():
    global x
    t.save(x.state_dict(), 'haha.t')
    x = X()
    x.load_state_dict(t.load('haha.t'))
