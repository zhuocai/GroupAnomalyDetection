import torch
from torch import nn, Tensor
import numpy as np
import matplotlib.pyplot as plt

device = 'cpu'


class MyModel(nn.Module):

    def __init__(self, input_shape, hidden_dim, lambda1=1, lambda2=1):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.full1 = nn.Linear(self.input_shape + self.hidden_dim, self.input_shape + self.hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.full2 = nn.Linear(self.input_shape + self.hidden_dim, self.input_shape)

        self.h = torch.nn.Parameter()

    def forward(self, input: Tensor):
        N, p = input.shape
        assert p == self.input_shape

        f_x_h = torch.cat((input, self.h), dim=1)
        f_x_h = torch.relu(self.full1(f_x_h))
        f_x_h = self.dropout1(f_x_h)
        f_x_h = torch.relu(self.full2(f_x_h))

        mu = torch.mean(self.h, dim=0)
        S = torch.mean(torch.stack([torch.ger(self.h[i], self.h[i]) for i in range(N - 1)])) - torch.ger(mu, mu)

        loss1 = torch.norm(f_x_h[:-1] - (input[1:] - input[:-1])) / (N - 1) \
                + self.lambda1 * torch.norm(self.h[1:] - self.h[:-1]) / (N - 1)
        loss = torch.norm(f_x_h[:-1] - (input[1:] - input[:-1])) / (N - 1) \
               + self.lambda1 * torch.norm(self.h[1:] - self.h[:-1]) / (N - 1) \
               + 0.5 * self.lambda2 * (mu.dot(mu) + S.trace()-S.shape[0] - torch.logdet(S + (1e-8) * torch.eye(S.shape[0])))

        return f_x_h, loss1, loss


def train(model: MyModel, X: Tensor, epochs=100, lr=0.03):
    print("training...")
    model.h.data = torch.randn([X.shape[0], model.hidden_dim], device=device, requires_grad=True)
    model.h.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=0.92, patience=10, cooldown=10,
                                                           min_lr=1e-5)
    param_num = 0
    for param in model.parameters():
        param_num += 1
        print("param", param_num, param.shape)
    model.h.requires_grad = True
    for epoch in range(epochs):
        optimizer.zero_grad()
        f_x_h, loss1, loss = model(X)
        print("h.std =", torch.std(model.h.data),"loss:", loss.item())
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        f_x_h, loss1, loss = model(X)
        loss.backward()
        scheduler.step(loss)
        # print("h.std =", torch.std(model.h.data))


def evaluate(model: MyModel, X: Tensor, epochs=100, lr=0.1):
    print("evaluating...")
    model.h.data = torch.randn([X.shape[0], model.hidden_dim], device=device, requires_grad=True)
    # model.h.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=0.92, patience=10, cooldown=10,
                                                           min_lr=1e-5)

    for param in model.parameters():
        param.requires_grad = False
    model.h.requires_grad = True
    for epoch in range(epochs):
        optimizer.zero_grad()
        f_x_h, loss1, loss = model(X)
        loss1.backward()
        optimizer.step()

        optimizer.zero_grad()
        f_x_h, loss1, loss = model(X)
        loss1.backward()
        scheduler.step(loss1)

    np.save("h.npy", model.h.data.detach().numpy())
    np.save("f_x_h.npy", f_x_h.detach().numpy())
    for param in model.parameters():
        param.requires_grad = True


if __name__ == "__main__":

    train_X = Tensor(np.load("train_x.npy"))
    test_X = Tensor(np.load("test_x.npy"))
    N, p = train_X.shape

    model = MyModel(input_shape=p, hidden_dim=1, lambda1=0.1, lambda2=0.1).to(device)
    # optimizer = torch.optim.Adam(model.parameters())
    param_num = 0
    for param in model.parameters():
        param_num += 1
        print("param", param_num, param.shape)

    train(model, train_X, epochs=40, lr=0.1)

    evaluate(model, test_X, epochs=100, lr=0.1)
