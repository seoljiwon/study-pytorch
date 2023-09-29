# implement computation graph and back propagation

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x * w


def minus(y_pred, y):
    return y_pred - y


def square(s):
    return s * s


def local_gradient_0(s):
    return 2 * s


def local_gradient_1():
    return 1


def local_gradient_2(x):
    return x


print("Prediction (before training)", 4, forward(4))

for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        s = minus(y_pred_val, y_val)
        l = square(s)
        g0 = local_gradient_0(s)
        g1 = local_gradient_1()
        g2 = local_gradient_2(x_val)
        grad = g0 * g1 * g2
        w = w - 0.01 * grad
        print("\t", x_val, y_val, "grad: ", grad, "loss: ", l)

    print("progress:", epoch, "w=", w)

print("Prediction (after training)", "4", forward(4))

# compute gradients using PyTorch
import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = Variable(torch.Tensor([1.0]), requires_grad=True)
w2 = Variable(torch.Tensor([1.0]), requires_grad=True)
b = Variable(torch.Tensor([1.0]), requires_grad=True)


def forward(x):
    return x * x * w2 + x * w1 + b


def loss(y_pred, y):
    return (y_pred - y) * (y_pred - y)


print("Prediction (before training)", 4, forward(4).data[0])

for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        l = loss(y_pred_val, y_val)
        l.backward()
        print(
            "\tgrad: ",
            x_val,
            y_val,
            "w1: ",
            w1.grad.data[0],
            "w2: ",
            w2.grad.data[0],
            "b: ",
            b.grad.data[0],
        )
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data

        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()

    print("progress: ", epoch, l.data)

print("Prediction (after training)", "4", forward(4).data[0])
