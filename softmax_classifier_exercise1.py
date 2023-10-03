import torch
from torch.autograd import Variable

# CrossEntropyLoss (LogSoftmax + NLLLoss)
loss = torch.nn.CrossEntropyLoss()

Y = Variable(torch.LongTensor([0]), requires_grad=False)

Y_pred1 = Variable(torch.Tensor([[2.0, 1.0, 0.1]]))
Y_pred2 = Variable(torch.Tensor([[0.5, 2.0, 0.3]]))

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print("Loss1=", l1.data, "\nLoss2=", l2.data)

# Batch loss
Y = Variable(torch.LongTensor([2, 0, 1]), requires_grad=False)

Y_pred1 = Variable(torch.Tensor([[0.1, 0.2, 0.9], [1.1, 0.1, 0.2], [0.2, 2.1, 0.1]]))
Y_pred2 = Variable(torch.Tensor([[0.8, 0.2, 0.3], [0.2, 0.3, 0.5], [0.2, 0.2, 0.5]]))

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print("Batch Loss1=", l1.data, "\nBatch Loss2=", l2.data)

# LogSoftmax + NLLLoss
loss = torch.nn.NLLLoss()

Y = Variable(torch.LongTensor([0]), requires_grad=False)

log_soft_max = torch.nn.LogSoftmax(dim=1)
Y_pred1 = Variable(torch.Tensor([[2.0, 1.0, 0.1]]))
Y_pred2 = Variable(torch.Tensor([[0.5, 2.0, 0.3]]))
Y_pred1 = log_soft_max(Y_pred1)
Y_pred2 = log_soft_max(Y_pred2)

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print("Batch Loss1=", l1.data, "\nBatch Loss2=", l2.data)
