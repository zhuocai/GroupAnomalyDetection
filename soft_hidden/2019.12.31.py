import torch
from torch import nn, Tensor
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda:0'
batch_size = 3600


class MyModel(nn.Module):

    def __init__(self, input_shape, hidden_dim):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim

        self.full1 = nn.Linear(self.input_shape + self.hidden_dim, 100)
        self.dropout1 = nn.Dropout(0.2)
        self.full12 = nn.Linear(100, 50)
        self.dropout12 = nn.Dropout(0.2)
        self.full13 = nn.Linear(50 + self.hidden_dim, 30)
        self.dropout13 = nn.Dropout(0.2)
        self.full14 = nn.Linear(30, 20)
        self.dropout14 = nn.Dropout(0.2)
        self.full15 = nn.Linear(20 + self.hidden_dim, 40)
        self.dropout15 = nn.Dropout(0.2)

        self.full2 = nn.Linear(40, self.input_shape)

        self.h = torch.nn.Parameter()

    def forward(self, input: Tensor, idx0=0, idx1=batch_size):
        N, p = input.shape
        assert p == self.input_shape

        f_x_h = torch.cat((input, self.h[idx0:idx1]), dim=1)
        f_x_h = torch.relu(self.full1(f_x_h))
        f_x_h = self.dropout1(f_x_h)
        # 100
        f_x_h = torch.relu(self.full12(f_x_h))
        f_x_h = self.dropout12(f_x_h)
        # 50
        f_x_h = torch.cat((f_x_h, self.h[idx0:idx1]), dim=1)
        f_x_h = torch.relu(self.full13(f_x_h))
        f_x_h = self.dropout13(f_x_h)
        # 50
        f_x_h = torch.relu(self.full14(f_x_h))
        f_x_h = self.dropout14(f_x_h)
        # 50
        f_x_h = torch.cat((f_x_h, self.h[idx0:idx1]), dim=1)
        f_x_h = torch.relu(self.full15(f_x_h))
        f_x_h = self.dropout15(f_x_h)
        # 50

        f_x_h = self.full2(f_x_h)

        mu = torch.mean(self.h[idx0:idx1], dim=0).to(device)
        S = torch.mean(torch.Tensor([torch.matmul(self.h[i], self.h[i]) for i in range(idx0, idx1)]))
        S = S.to(device)

        part0 = torch.norm(f_x_h[:-1] - input[1:]) / (N - 1)
        part1 = torch.norm(mu, p=1) / self.hidden_dim
        part2 = torch.abs(S / self.hidden_dim - 1)
        part3 = torch.mean(
            torch.norm(self.h[idx0 + 1:idx1] - self.h[idx0:idx1 - 1], p=1, dim=1)) / self.hidden_dim

        return f_x_h, part0, part1, part2, part3


def train(model: MyModel, X: Tensor, epochs=100, lr=0.03, h_lr=0.1, steps_per_sample=5, f_per_sample=1,
          lambda0=1, lambda1=1, lambda2=1, lambda3=1):
    print("training...")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=0.95, patience=5, cooldown=5,
                                                           min_lr=1e-5)

    optimizer_h = torch.optim.Adam(torch.nn.ParameterList([model.h]), lr=h_lr)
    scheduler_h = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_h,
                                                             factor=0.9, patience=5, cooldown=5,
                                                             min_lr=1e-5)
    h_init_weight = torch.randn([X.shape[1], model.hidden_dim], device=device)/np.sqrt(X.shape[1])
    model.h.data = torch.matmul(X, h_init_weight)
    model.h.requires_grad = True
    param_num = 0
    for param in model.parameters():
        param_num += 1
        print("param", param_num, param.shape)
    model.h.requires_grad = True

    for epoch in range(epochs):
        print('{}/{}'.format(epoch, epochs))
        for i in np.random.randint(0, X.shape[0] - batch_size, 1):
            update_f_steps = np.random.choice(steps_per_sample, f_per_sample, replace=False)
            for step in range(steps_per_sample):
                print('i = ', i, ' update f?:', (step in update_f_steps))

                if step in [steps_per_sample-1]:#update_f_steps:
                    optimizer.zero_grad()
                    f_x_h, part0, part1, part2, part3 = model(X[i:i + batch_size], idx0=i, idx1=i + batch_size)
                    print('part0:', part0.item(), 'part1:', part1.item(), 'part2:', part2.item(), 'part3:',
                          part3.item())
                    loss = lambda0 * part0 + lambda1 * part1 + lambda2 * part2 + lambda2 * part3
                    loss.backward()
                    optimizer.step()

                else:
                    optimizer_h.zero_grad()
                    f_x_h, part0, part1, part2, part3 = model(X[i:i + batch_size], idx0=i, idx1=i + batch_size)
                    print('part0:', part0.item(), 'part1:', part1.item(), 'part2:', part2.item(), 'part3:',
                          part3.item())
                    loss = lambda0 * part0 + lambda1 * part1 + lambda2 * part2 + lambda2 * part3
                    loss.backward()
                    optimizer_h.step()

            if np.random.random() < f_per_sample / steps_per_sample:
                optimizer.zero_grad()
                f_x_h, part0, part1, part2, part3 = model(X[i:i + batch_size], idx0=i, idx1=i + batch_size)
                print('part0:', part0.item(), 'part1:', part1.item(), 'part2:', part2.item(), 'part3:', part3.item())
                loss = lambda0 * part0 + lambda1 * part1 + lambda2 * part2 + lambda2 * part3
                loss.backward()
                scheduler.step(loss)
            else:
                # optimizer_h.zero_grad()
                # f_x_h, part0, part1, part2, part3 = model(X[i:i + batch_size], idx0=i, idx1=i + batch_size)
                # print('part0:', part0.item(), 'part1:', part1.item(), 'part2:', part2.item(), 'part3:', part3.item())
                # loss = lambda0 * part0 + lambda1 * part1 + lambda2 * part2 + lambda2 * part3
                # loss.backward()
                # scheduler_h.step(loss)
                pass
        # print("h.std =", torch.std(model.h.data))


def evaluate(model: MyModel, X: Tensor, epochs=100, lr=0.1, lambda0=1, lambda1=1, lambda2=1, lambda3=1):
    print("evaluating...")
    model.h = torch.nn.Parameter()
    h_init_weight = torch.randn([X.shape[1], model.hidden_dim], device=device)/np.sqrt(X.shape[1])
    model.h.data = torch.matmul(X, h_init_weight)
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
            loss = lambda0 * part0 + lambda1 * part1 + lambda2 * part2 + lambda2 * part3
            loss.backward()
            optimizer.step()

            # optimizer.zero_grad()
            # f_x_h, part0, part1, part2, part3 = model(X[idxs[i]:idxs[i + 1]], idx0=idxs[i], idx1=idxs[i + 1])
            # loss = lambda0 * part0 + lambda1 * part1 + lambda2 * part2 + lambda2 * part3
            # loss.backward()
            # scheduler.step(loss)

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

    model = MyModel(input_shape=p, hidden_dim=15).to(device)

    train(model, train_X, epochs=500, lr=0.03, h_lr=0.1, steps_per_sample=10, f_per_sample=2, lambda0=10, lambda1=1,
          lambda2=1, lambda3=0.05)
    torch.save(model.state_dict(), 'model.model')
    evaluate(model, test_X, epochs=20, lr=0.3, lambda0=10, lambda1=1, lambda2=1, lambda3=0.01)
