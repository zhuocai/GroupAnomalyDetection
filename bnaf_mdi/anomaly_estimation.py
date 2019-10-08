import os
import json
import argparse
import pprint
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data
import bnaf
from optim import adam
from optim import lr_scheduler
import tqdm


def create_model(args, verbose: bool = False):
    flows = []
    for f in range(args.flows):
        layers = []
        for _ in range(args.layers - 1):
            layers.append(bnaf.MaskedWeight(args.n_dim * args.hidden_dim,
                                            args.n_dim * args.hidden_dim, dim=args.n_dim))
            layers.append(bnaf.Tanh())

        flows.append(
            bnaf.BNAF(*([bnaf.MaskedWeight(args.n_dim, args.n_dim * args.hidden_dim, dim=args.n_dim), bnaf.Tanh()] +
                        layers +
                        [bnaf.MaskedWeight(args.n_dim * args.hidden_dim, args.n_dim, dim=args.n_dim)]),
                      res='gated' if f < args.flows - 1 else False
                      )
        )

        if f < args.flows - 1:
            flows.append(bnaf.Permutation(args.n_dim, 'flip'))

    model = bnaf.Sequential(*flows).to(args.device)
    params = sum(
        (p != 0).sum() if len(p.shape) > 1 else torch.tensor(p.shape).item() for p in model.parameters()).item()

    if verbose:
        print('{}'.format(model))
        print('Parameters={}, n_dims={}'.format(sum((p != 0).sum()
                                                    if len(p.shape) > 1 else torch.tensor(p.shape).item()
                                                    for p in model.parameters()), args.n_dim))

    return model


def compute_log_g_x(model: bnaf.Sequential, x_mb: torch.Tensor):
    y_mb, log_diag_j_mb = model(x_mb)
    log_p_y_mb = torch.distributions.Normal(torch.zeros_like(y_mb),
                                            torch.ones_like(y_mb)).log_prob(y_mb).sum(-1)
    return log_p_y_mb + log_diag_j_mb


def train(dataset_loader: tuple, model: bnaf.Sequential, optimizer: adam.Adam,
          scheduler: lr_scheduler.ReduceLROnPlateau,
          args):
    dataloader_train, dataloader_test, dataloader_val = dataset_loader
    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(os.path.join(args.tensorboard, args.load or args.path))

    epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        t = tqdm.tqdm(dataloader_train, smoothing=0, ncols=80)
        train_loss = []

        for x_mb, in t:
            loss = - compute_log_g_x(model, x_mb).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)

            optimizer.step()
            optimizer.zero_grad()

            t.set_postfix(loss='{:.2f}'.format(loss.item()), refresh=False)
            train_loss.append(loss)

        train_loss = torch.stack(train_loss).mean()
        optimizer.swap()
        validation_loss = - torch.stack([compute_log_g_x(model, x_mb).mean().detach()
                                         for x_mb, in dataloader_val], -1).mean()
        optimizer.swap()

        print('Epoch {:3}/{:3} -- train-loss: {:4.3f} --validation_loss: {:4.3f}'.format(
            epoch + 1, args.start_epoch + args.epochs, train_loss.item(), validation_loss.item()
        ))

        stop = scheduler.step(validation_loss,
                              callback_best=save(model, optimizer, epoch + 1, args),
                              callback_reduce=load(model, optimizer, args))

        if args.tensorboard:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)
            writer.add_scalar('loss/validation', validation_loss.item(), epoch + 1)
            writer.add_scalar('loss/train', train_loss.item(), epoch + 1)

        if (stop):
            break

    load(model, optimizer, args)
    optimizer.swap()
    validation_loss = -torch.stack([compute_log_g_x(model, x_mb).mean().detach()
                                    for x_mb, in dataloader_val], -1).mean()
    test_loss = -torch.stack([compute_log_g_x(model, x_mb).mean().detach()
                              for x_mb, in dataloader_test], -1).mean()

    print('###### Stop training after {} epochs!'.format(epoch + 1))
    print('validation loss {:4.3f}'.format(validation_loss.item()))
    print('Test loss:      {:4.3f}'.format(test_loss.item()))

    if args.save:
        with open(os.path.join(args.load or args.path, 'results.txt'), 'a') as f:
            print('##### Stop training after {} epochs!'.format(epoch + 1), file=f)
            print('Validation loss: {:4.3f}'.format(validation_loss.item()), file=f)
            print('Test loss: {:4.3f}'.format(test_loss.item()), file=f)


def load_np(args):
    data_dir = "/home/caizhuo/research/anomaly/data"
    if args.dataset == "wadi":
        if args.dataset_type == "normal":
            print("read dataset: wadi*normal")
            return np.load(os.path.join(data_dir, "wadi/normal"+args.dataset_filename))
        elif args.dataset_type == "anomaly":
            if args.is_training:
                return np.load(os.path.join(data_dir, "wadi/anomaly"+args.dataset_filename))[:, :-1]
            else:
                return np.load(os.path.join(data_dir, "wadi/anomaly"+args.dataset_filename))
        elif args.dataset_type == "all":
            data_normal = np.load(os.path.join(data_dir, "wadi/normal"+args.dataset_filename))
            data_anomaly = np.load(os.path.join(data_dir, "wadi/anomaly"+args.dataset_filename))[:, :-1]
            if args.is_training:
                return np.concatenate((data_normal, data_anomaly), axis=0)
            else:
                return data_normal, data_anomaly
    elif args.dataset == "swat":
        if args.dataset_type == "normal":
            return np.load(os.path.join(data_dir, "swat/normal_pc.npy"))
        elif args.dataset_type == "anomaly":
            if args.is_training:
                return np.load(os.path.join(data_dir, "swat/anomaly_pc.npy"))[:, :-1]
            else:
                return np.load(os.path.join(data_dir, "swat/anomaly_pc.npy"))
        elif args.dataset_type == "all":
            data_normal = np.load(os.path.join(data_dir, "swat/normal_pc.npy"))
            data_anomaly = np.load(os.path.join(data_dir, "swat/anomaly_pc.npy"))[:, :-1]
            if args.is_training:
                return np.concatenate((data_normal, data_anomaly), axis=0)
            else:
                return data_normal, data_anomaly
    else:
        print("args.dataset load error")
        return


def load_dataset(args):
    dataset_np_temp = load_np(args)
    args.n_dim = dataset_np_temp.shape[1]
    if args.is_training:
        train_np, test_np = train_test_split(dataset_np_temp, test_size=0.25, random_state=40)
        test_np, val_np = train_test_split(test_np, test_size=0.25, random_state=40)

        dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(train_np).float().to(args.device))
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_dim, shuffle=True)

        dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(test_np).float().to(args.device))
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_dim, shuffle=True)

        dataset_val = torch.utils.data.TensorDataset(torch.from_numpy(val_np).float().to(args.device))
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_dim, shuffle=True)

        return dataloader_train, dataloader_test, dataloader_val
    else:
        _dataset = torch.utils.data.TensorDataset(torch.from_numpy(dataset_np_temp).float().to(args.device))
        _dataloader = torch.utils.data.DataLoader(_dataset)
        return _dataloader


def load(model: bnaf.Sequential, optimizer: adam, args, load_start_epoch=False):
    def f():
        if args.load:
            print('Loading model..')
            checkpoint = torch.load(os.path.join(args.load or args.path, 'checkpoint.pt'))
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            if load_start_epoch:
                args.start_epoch = checkpoint['epoch']

    return f


def save(model: bnaf.Sequential, optimizer: adam, epoch, args):
    def f():
        if args.save:
            print("Saving model..")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, os.path.join(args.load or args.path, 'checkpoint.pt'))

            with open("checkpoint_path.txt", "w") as file:
                file.write(args.load or args.path)

    return f


def train_main(args):
    if not os.path.isdir("cai"):
        os.mkdir("cai")

    if not os.path.isdir("cai/checkpoint"):
        os.mkdir("cai/checkpoint")

    args.path = os.path.join('cai/checkpoint', '{}_layers{}_h{}_flows{}_{}'.format(
        args.dataset, args.layers, args.hidden_dim, args.flows,
        str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

    print('loading datasets')
    dataset_loader = load_dataset(args)

    if args.save and not args.load:
        print('Creating directory experiment..')
        os.mkdir(args.path)
        with open(os.path.join(args.path, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=4, sort_keys=True)

    print("creating BNAF model..")
    model = create_model(args, verbose=True)

    print("creating optimizer..")
    optimizer = adam.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    print("creating scheduler..")
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.decay,
                                               patience=args.patience, cooldown=args.cooldown,
                                               min_lr=args.min_lr, verbose=True,
                                               early_stopping=args.early_stopping,
                                               threshold_mode='abs')

    args.start_epoch = 0
    if args.load:
        load(model, optimizer, args, load_start_epoch=True)()

    print("training..")
    train(dataset_loader, model, optimizer, scheduler, args)

    return


def convert_data_np(model, data_np):
    model.eval()
    sample_size = data_np.shape[0]
    convert_batch_size = 2000
    start_i = 0
    y_mds = np.zeros(data_np.shape)
    attack_level = np.zeros(sample_size)
    while start_i < sample_size:
        print("start i:", start_i)
        end_i = min(sample_size, start_i + convert_batch_size)
        x_mb = torch.from_numpy(data_np[start_i:end_i, :]).float().to(args.device)
        #y_md_tensor,  = model()
        loss = - compute_log_g_x(model, x_mb).mean()
        y_mb, log_diag_j_mb = model(x_mb)
        log_p_y_mb = torch.distributions.Normal(torch.zeros_like(y_mb),
                                                torch.ones_like(y_mb)).log_prob(y_mb).sum(-1)
        loss = log_p_y_mb + log_diag_j_mb
        print("loss.shape:", loss.shape)
        attack_level[start_i:end_i] = loss.detach().cpu().numpy()
        y_md_np = y_mb.detach().cpu().numpy()
        print("y_md_np.shape", y_md_np.shape)
        y_mds[start_i:end_i, :] = y_md_np
        start_i += convert_batch_size
    np.save(os.path.join(args.dataset_path, args.dataset+"/loss.npy"), attack_level)
    model.train()
    return y_mds


def convert_data(args):
    print('loading datasets')
    dataset_np = load_np(args)
    if args.dataset_type == 'normal':
        args.n_dim = dataset_np.shape[1]
    elif args.dataset_type == 'anomaly':
        args.n_dim = dataset_np.shape[1] - 1
    else:
        args.n_dim = dataset_np[0].shape[1]

    print("creating BNAF model..")
    model = create_model(args, verbose=True)

    print("creating optimizer..")
    optimizer = adam.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    args.start_epoch = 0
    if args.load:
        load(model, optimizer, args, load_start_epoch=True)()
    else:
        print("convert flow model args.load can't be empty")
        return

    if args.dataset_type == 'normal':
        np.save(os.path.join(args.dataset_path, args.dataset + "/" + args.dataset_type + "_1.npy"),
                convert_data_np(model, dataset_np))
    elif args.dataset_type == 'anomaly':
        np.save(os.path.join(args.dataset_path, args.dataset + "/" + args.dataset_type + "_1.npy"),
                np.concatenate((convert_data_np(model, dataset_np[:, :-1]), dataset_np[:, -1].reshape(dataset_np.shape[0], 1)),
                               axis=1))
    elif args.dataset_type == 'all':
        np.save(os.path.join(args.dataset_path, args.dataset + "/normal_1.npy"),
                convert_data_np(model, dataset_np[0]))
        np.save(os.path.join(args.dataset_path, args.dataset + "/anomaly_1.npy"),
                np.concatenate(
                    (convert_data_np(model, dataset_np[1][:, :-1]), dataset_np[1][:, -1].reshape(dataset_np[1].shape[0], 1)),
                    axis=1))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='wadi',
                        choices=['wadi', 'swat'])
    parser.add_argument('--dataset_type', type=str, default="normal",
                        choices=['normal', 'anomaly', 'all'])
    parser.add_argument('--dataset_path', type=str, default='../../data/')
    parser.add_argument('--is_training', type=int, default=1,
                        choices=[0, 1])
    parser.add_argument('--dataset_filename',type=str, default="_np2.npy")

    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--batch_dim', type=int, default=200)
    parser.add_argument('--clip_norm', type=float, default=.1)
    parser.add_argument('--epochs', type=int, default=1000)

    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--cooldown', type=int, default=10)
    parser.add_argument('--early_stopping', type=int, default=100)
    parser.add_argument('--decay', type=float, default=0.5)
    parser.add_argument('--min_lr', type=float, default=5e-4)
    parser.add_argument('--polyak', type=float, default=0.998)

    parser.add_argument('--flows', type=int, default=5)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=10)
    parser.add_argument('--residual', type=str, default='gated',
                        choices=[None, 'normal', 'gated'])

    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--tensorboard', type=str, default='tensorboard')

    args = parser.parse_args()

    args.is_training = bool(args.is_training)

    print('Arguments:')
    pprint.pprint(args.__dict__)

    if args.is_training:
        train_main(args)
    else:
        convert_data(args)
