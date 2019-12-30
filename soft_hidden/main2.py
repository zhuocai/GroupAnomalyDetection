import torch
from torch import nn, Tensor
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda:0'
batch_size = 1000


class MyModel(nn.Module):

    def __init__(self, input_shape, hidden_dim, lambda0=1, lambda1=1, lambda2=1, lambda3=1):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

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
        S = torch.mean(torch.Tensor([torch.matmul(self.h[i], self.h[i]) for i in range(idx0, idx1)]))
        S = S.to(device)

        part0 = torch.norm(f_x_h[:-1] - input[1:]) / (N - 1)
        part1 = self.lambda1 * torch.norm(mu, p=1) / self.hidden_dim
        part2 = self.lambda1 * torch.abs(S / self.hidden_dim - 1)
        part3 = self.lambda3 * torch.mean(
            torch.norm(self.h[idx0 + 1:idx1] - self.h[idx0:idx1 - 1], p=1, dim=1)) / self.hidden_dim

        return f_x_h, part0, part1, part2, part3


def train(model: MyModel, X: Tensor, epochs=100, lr=0.03, steps_per_sample=5):
    print("training...")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=0.92, patience=2, cooldown=2,
                                                           min_lr=1e-5)
    model.h.data = torch.randn([X.shape[0], model.hidden_dim], device=device, requires_grad=True)
    model.h.requires_grad = True
    param_num = 0
    for param in model.parameters():
        param_num += 1
        print("param", param_num, param.shape)
    model.h.requires_grad = True

    for epoch in range(epochs):
        print('{}/{}'.format(epoch, epochs))
        for i in np.random.randint(0, X.shape[0] - batch_size, 1):
            for step in range(steps_per_sample):
                print('i = ', i)
                optimizer.zero_grad()
                f_x_h, part0, part1, part2, part3 = model(X[i:i + batch_size], idx0=i, idx1=i + batch_size)
                print('part0:', part0.item(), 'part1:', part1.item(), 'part2:', part2.item(), 'part3:', part3.item())
                loss = part0 + part1 + part2 + part3
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()
            f_x_h, part0, part1, part2, part3 = model(X[i:i + batch_size], idx0=i, idx1=i + batch_size)
            print('part0:', part0.item(), 'part1:', part1.item(), 'part2:', part2.item(), 'part3:', part3.item())
            loss = part0 + part1 + part2 + part3
            loss.backward()
            scheduler.step(loss)

        # print("h.std =", torch.std(model.h.data))


def evaluate(model: MyModel, X: Tensor, epochs=100, lr=0.1):
    print("evaluating...")
    model.h = torch.nn.Parameter()
    model.h.data = torch.randn(X.shape[0], model.hidden_dim, device=device, requires_grad=True)
    for param in model.parameters():
        # print('param.shape',param.shape)
        param.requires_grad = False
    model.h.requires_grad = True

    params = torch.nn.ParameterList([model.h])

    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=0.92, patience=10, cooldown=10,
                                                           min_lr=1e-5)

    idxs = list(np.arange(0, X.shape[0], batch_size))
    idxs.append(X.shape[0])
    for i in range(len(idxs) - 1):
        print('{}-{}/{}'.format(idxs[i], idxs[i + 1], idxs[-1]))
        for epoch in range(epochs):
            optimizer.zero_grad()
            f_x_h, part0, part1, part2, part3 = model(X[idxs[i]:idxs[i + 1]], idx0=idxs[i], idx1=idxs[i + 1])
            print('part0:', part0.item(), 'part1:', part1.item(), 'part2:', part2.item(), 'part3:', part3.item())
            loss = part0 + part1 + part2 + part3
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            f_x_h, part0, part1, part2, part3 = model(X[idxs[i]:idxs[i + 1]], idx0=idxs[i], idx1=idxs[i + 1])
            loss = part0 + part1 + part2 + part3
            loss.backward()
            scheduler.step(loss)

    np.save("h.npy", model.h.data.detach().cpu().numpy())
    # np.save("f_x_h.npy", f_x_h.detach().numpy())

    for param in model.parameters():
        param.requires_grad = True


if __name__ == "__main__":
    # train_X = Tensor(np.load("x_normal_11.npy"))
    # test_X = Tensor(np.load("x_anomaly_11.npy"))
    train_X = Tensor(np.load('../../data/wadi/normal_valid.npy')).to(device)
    test_X = Tensor(np.load('../../data/wadi/anomaly_valid.npy')).to(device)
    print('train_x.shape', train_X.shape, 'test_x.shape', test_X.shape)
    N, p = train_X.shape

    model = MyModel(input_shape=p, hidden_dim=10, lambda0=5, lambda1=1, lambda2=1, lambda3=0.2).to(device)

    train(model, train_X, epochs=100, lr=0.03, steps_per_sample=10)
    torch.save(model.state_dict(), 'model.model')
    evaluate(model, test_X, epochs=10, lr=0.1)
