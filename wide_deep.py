import numpy as np
import torch
from torch.autograd import Variable

xy = np.loadtxt("data/diabetes.csv", delimiter=",", dtype=np.float32)

x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
y_data = Variable(torch.from_numpy(xy[:, [-1]]))

print(x_data.data.shape)
print(y_data.data.shape)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 4)
        self.l4 = torch.nn.Linear(4, 3)
        self.l5 = torch.nn.Linear(3, 4)
        self.l6 = torch.nn.Linear(4, 6)
        self.l7 = torch.nn.Linear(6, 2)
        self.l8 = torch.nn.Linear(2, 4)
        self.l9 = torch.nn.Linear(4, 1)
        self.l10 = torch.nn.Linear(1, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        out3 = self.sigmoid(self.l3(out2))
        out4 = self.sigmoid(self.l4(out3))
        out5 = self.sigmoid(self.l5(out4))
        out6 = self.sigmoid(self.l6(out5))
        out7 = self.sigmoid(self.l7(out6))
        out8 = self.sigmoid(self.l8(out7))
        out9 = self.sigmoid(self.l9(out8))
        y_pred = self.sigmoid(self.l10(out9))
        return y_pred


model = Model()

criterion = torch.nn.BCELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)
    if epoch % 10 == 9:
        print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
