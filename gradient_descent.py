x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x * w


def loss(y_pred, y):
    return (y_pred - y) * (y_pred - y)


def gradient(x, y):
    return 2 * x * (x * w - y)


print("Prediction (before training)", 4, forward(4))

for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(forward(x_val), y_val)
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad
        print("\t", x_val, y_val, "grad: ", grad, "loss: ", l)

    print("progress:", epoch, "w=", w)

print("Prediction (after training)", "4", forward(4))
