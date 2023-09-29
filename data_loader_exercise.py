import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


class TitanicDataset(Dataset):
    def __init__(self):
        df_train = pd.read_csv("data/titanic_train.csv")

        # drop useless columns
        df_train.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

        # convert categorical columns to one-hot encoding
        sex = pd.get_dummies(df_train["Sex"], drop_first=True)
        embark = pd.get_dummies(df_train["Embarked"], drop_first=True)

        # drop categorical columns
        df_train = pd.concat([df_train, sex, embark], axis=1)
        df_train.drop(["Sex", "Embarked"], axis=1, inplace=True)

        # fill missing values
        df_train.fillna(df_train.mean(), inplace=True)
        xy = df_train.values.astype(np.float32)

        # split data into x and y
        self.len = df_train.shape[0]
        self.x_data = torch.from_numpy(xy[:, 2:])
        self.y_data = torch.from_numpy(xy[:, [1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = TitanicDataset()
train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8, 512)
        self.l2 = torch.nn.Linear(512, 512)
        self.l3 = torch.nn.Linear(512, 1)

        self.lelu = torch.nn.ReLU()

    def forward(self, x):
        out1 = self.lelu(self.l1(x))
        out2 = self.lelu(self.l2(out1))
        y_pred = self.lelu(self.l3(out2))
        return y_pred


model = Model()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        y_pred = model(inputs)

        loss = criterion(y_pred, labels)
        print(f"Epoch: {i} | Loss {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
