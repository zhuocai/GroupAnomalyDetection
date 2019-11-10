import torch
from torch import nn, Tensor
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda:0'
batch_size = 12345


class MyModel(nn.Module):

    def __init__(self, input_shape, hidden_dim, lambda1=1, lambda2=1):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.full1 = nn.Linear(self.input_shape + self.hidden_dim, 200)
        self.dropout1 = nn.Dropout(0.2)
        self.full12 = nn.Linear(200, 100)
        self.dropout12 = nn.Dropout(0.2)
        self.full13 = nn.Linear(100, 100)
        self.dropout13 = nn.Dropout(0.2)

        self.full2 = nn.Linear(100, self.input_shape)

        self.h = torch.nn.Parameter()

    def forward(self, input: Tensor, idx0=0, idx1=batch_size):
        N, p = input.shape
        assert p == self.input_shape

        f_x_h = torch.cat((input, self.h[idx0:idx1]), dim=1)
        f_x_h = torch.relu(self.full1(f_x_h))
        f_x_h = self.dropout1(f_x_h)
        f_x_h = torch.relu(self.full12(f_x_h))
        f_x_h = self.dropout12(f_x_h)
        f_x_h = torch.relu(self.full13(f_x_h))
        f_x_h = self.dropout13(f_x_h)
        f_x_h = self.full2(f_x_h)

        mu = torch.mean(self.h[idx0:idx1], dim=0).to(device)
        S = torch.mean(torch.stack([torch.ger(self.h[idx0:idx1][i], self.h[idx0:idx1][i]) for i in range(N - 1)]),
                       dim=0) - torch.ger(mu, mu)
        S = S.to(device)
        #print('mu', mu, 'S', S)
        part1 = torch.norm(f_x_h[:-1] - input[1:]) / (N - 1)
        part2 = self.lambda1 * torch.norm(self.h[idx0:idx1][1:] - self.h[idx0:idx1][:-1]) / (N - 1)
        part3 = 0.5 * self.lambda2 * (
                mu.dot(mu) + S.trace() - S.shape[0] - torch.logdet(S + (1e-8) * torch.eye(S.shape[0]).to(device)))

        loss1 = part1 + part2
        loss = part1 + part2 + part3

        return f_x_h, loss1, loss, (part1, part2, part3)


def train(model: MyModel, X: Tensor, epochs=100, lr=0.03):
    print("training...")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=0.92, patience=2, cooldown=2,
                                                           min_lr=1e-5)
    model.h.data = torch.randn([X.shape[0], model.hidden_dim], device=device, requires_grad=True) * 0.05
    model.h.requires_grad = True
    param_num = 0
    for param in model.parameters():
        param_num += 1
        print("param", param_num, param.shape)
    model.h.requires_grad = True

    for epoch in range(epochs):
        print('{}/{}'.format(epoch, epochs))
        for i in np.random.randint(0, X.shape[0] - batch_size, 1):
            print('i = ', i)
            optimizer.zero_grad()
            f_x_h, loss1, loss, losses = model(X[i:i + batch_size], idx0=i, idx1=i + batch_size)
            print('loss1', loss1.item(), 'loss', loss.item(), (losses[0].item(), losses[1].item(), losses[2].item()))
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            f_x_h, loss1, loss, losses = model(X[i:i + batch_size], idx0=i, idx1=i + batch_size)
            loss.backward()
            scheduler.step(loss)

        # print("h.std =", torch.std(model.h.data))


def evaluate(model: MyModel, X: Tensor, epochs=100, lr=0.1):
    print("evaluating...")

    # model.h.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=0.92, patience=10, cooldown=10,
                                                           min_lr=1e-5)
    model.h = torch.nn.Parameter()
    model.h.data = torch.randn(X.shape[0], model.hidden_dim, device=device, requires_grad=True) * 0.05
    for param in model.parameters():
        # print('param.shape',param.shape)
        param.requires_grad = False
    model.h.requires_grad = True
    h_data = np.zeros((X.shape[0], model.hidden_dim))
    # print('model.h', model.h.shape)

    idxs = list(np.arange(0, X.shape[0], batch_size))
    idxs.append(X.shape[0])
    for i in range(len(idxs) - 1):
        print('{}-{}/{}'.format(idxs[i], idxs[i + 1], idxs[-1]))
        for epoch in range(epochs):
            optimizer.zero_grad()
            f_x_h, loss1, loss, losses = model(X[idxs[i]:idxs[i + 1]], idx0=idxs[i], idx1=idxs[i + 1])
            losses[0].backward()
            optimizer.step()
            print('loss1', loss1.item(), 'loss', loss.item(), (losses[0].item(), losses[1].item(), losses[2].item()))

            optimizer.zero_grad()
            f_x_h, loss1, loss, losses = model(X[idxs[i]:idxs[i + 1]], idx0=idxs[i], idx1=idxs[i + 1])
            losses[0].backward()
            scheduler.step(losses[0])

    np.save("h.npy", model.h.data.detach().numpy())
    # np.save("f_x_h.npy", f_x_h.detach().numpy())

    for param in model.parameters():
        param.requires_grad = True


if __name__ == "__main__":
    # train_X = Tensor(np.load("x_normal_11.npy"))
    # test_X = Tensor(np.load("x_anomaly_11.npy"))
    train_X = Tensor(np.load('../../data/wadi/normal_np4.npy')).to(device)
    test_X = Tensor(np.load('../../data/wadi/anomaly_np4.npy')[:, :-1]).to(device)
    print('train_x.shape', train_X.shape, 'test_x.shape', test_X.shape)
    N, p = train_X.shape

    model = MyModel(input_shape=p, hidden_dim=10, lambda1=0, lambda2=0.0).to(device)
    # optimizer = torch.optim.Adam(model.parameters())
    param_num = 0
    for param in model.parameters():
        param_num += 1
    print("param", param_num, param.shape)

    train(model, train_X, epochs=500, lr=0.03)
    torch.save(model.state_dict(), 'model.model')
    evaluate(model, test_X, epochs=10, lr=0.1)
