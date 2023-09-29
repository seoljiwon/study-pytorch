x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = 1.0
w2 = 1.0
b = 0.0


def forward(x):
    return x * x * w2 + x * w1 + b


def loss(y_pred, y):
    return (y_pred - y) * (y_pred - y)


def gradient_w1(x, y):
    return 2 * (x) * (x * x * w2 + x * w1 + b - y)


def gradient_w2(x, y):
    return 2 * (x * x) * (x * x * w2 + x * w1 + b - y)


print("Prediction (before training)", 4, forward(4))

for epoch in range(1000):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(forward(x_val), y_val)
        grad_w1 = gradient_w1(x_val, y_val)
        grad_w2 = gradient_w2(x_val, y_val)
        w1 = w1 - 0.01 * grad_w1
        w2 = w2 - 0.01 * grad_w2
        print(
            "\t", x_val, y_val, "grad_w1: ", grad_w1, "grad_w2: ", grad_w2, "loss: ", l
        )

    if epoch % 10 == 0:
        print("progress: ", epoch, "w1=", w1, "w2=", w2)

print("Prediction (after training)", "4", forward(4))
