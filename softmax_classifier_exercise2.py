import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

batch_size = 64

data = pd.read_csv(
    "data/otto-group-product-classification-challenge/train.csv",
    index_col="id",
)

# label encoding
le = LabelEncoder()
le = le.fit(data["target"])
data["target"] = le.transform(data["target"])

# split data into train and test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


class OttoGroupProductDataset(Dataset):
    def __init__(self, df):
        features = df.drop(["target"], axis=1)
        label = df["target"]

        self.len = df.shape[0]
        self.x_data = features.values.astype(np.float32)
        self.y_data = label.values.astype(np.int64)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


train_dataset = OttoGroupProductDataset(train_data)
test_dataset = OttoGroupProductDataset(test_data)


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(93, 512)
        self.l2 = torch.nn.Linear(512, 512)
        self.l3 = torch.nn.Linear(512, 512)
        self.l4 = torch.nn.Linear(512, 512)
        self.l5 = torch.nn.Linear(512, 9)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.relu(self.l4(x))
        return self.l5(x)  # No need activation


model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} ({100.0*batch_idx/len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)

        output = model(data)
        test_loss += criterion(output, target).item()

        pred = torch.max(output.data, 1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0*correct/len(test_loader.dataset):.0f}%)\n"
    )


for epoch in range(1, 10):
    train(epoch)
    test()
