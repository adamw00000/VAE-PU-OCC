# %%
import torch
import os
import numpy as np
import numba as nb
from datasets import load_dataset
from torchvision.datasets import MNIST, CIFAR10, STL10
from data_loading.small_dataset_wrapper import DATASET_NAMES, get_small_dataset


@nb.jit(parallel=True)
def is_in_set_pnb(a, b):
    shape = a.shape
    a = a.ravel()
    n = len(a)
    result = np.full(n, False)
    for i in nb.prange(n):
        if a[i] in b:
            result[i] = True
    return result.reshape(shape)


def __get_pu_mnist(included_classes, scar, positive_classes, label_frequency,
        data_dir='./data/', val_size=0.15, test_size=0.15):
    included_classes = torch.tensor(included_classes)
    positive_classes = torch.tensor(positive_classes)

    # Prepare datasets
    train_base = MNIST(root=data_dir, train=True, download=True)
    test_base = MNIST(root=data_dir, train=False, download=True)

    X = torch.cat([
        train_base.data.float().view(-1,784) / 255.,
        test_base.data.float().view(-1,784) / 255.
    ], 0)
    y = torch.cat([train_base.targets, test_base.targets], 0)

    indices_all = torch.where(torch.isin(y, included_classes))
    X, y = X[indices_all], y[indices_all]
    X = 2 * X - 1

    if not scar:
        o = torch.zeros_like(y)
        for pos_class in positive_classes:
            # Get positive distribution
            p_indices = torch.where(y == pos_class)[0]
            X_pos = X[p_indices]

            # Sample positive distribution
            num_labeled = int(len(X_pos) * label_frequency)
            bold_sort = torch.argsort(X_pos.sum(axis=1), descending=True)
            most_bold_pos_indices = bold_sort[0:num_labeled]
            l_indices = p_indices[most_bold_pos_indices]
            
            o[l_indices] = 1
    
    y = torch.where(torch.isin(y, positive_classes), 1, -1)

    if scar:
        o = torch.where(
            y == 1, 
            torch.where(
                torch.rand_like(y, dtype=float) < label_frequency, 
                1,
                0
            ),
            0
        ) # label without bias
    
    # Split datasets
    n = X.shape[0]
    n_val = int(val_size * n)
    n_test = int(test_size * n)
    n_train = n - n_val - n_test

    shuffled_indices = torch.randperm(X.shape[0])
    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train:(n_train+n_val)]
    test_indices = shuffled_indices[(n_train+n_val):]

    x_train, y_train, o_train = X[train_indices], y[train_indices], o[train_indices]
    x_val, y_val, o_val = X[val_indices], y[val_indices], o[val_indices]
    x_test, y_test, o_test = X[test_indices], y[test_indices], o[test_indices]

    train = x_train.numpy(), y_train.numpy(), o_train.numpy()
    val = x_val.numpy(), y_val.numpy(), o_val.numpy()
    test = x_test.numpy(), y_test.numpy(), o_test.numpy()

    pi_p = (y == 1).float().mean().numpy()
    label_frequency = (o == 1).float().mean().numpy() / pi_p
    n_input = X.shape[1]

    return train, val, test, label_frequency, pi_p, n_input

def __get_pu_cifar10_precomputed(included_classes, scar, positive_classes, label_frequency,
        data_dir='./data/', val_size=0.15, test_size=0.15):
    included_classes = torch.tensor(included_classes)
    positive_classes = torch.tensor(positive_classes)

    X_preprocessed = torch.load(os.path.join(data_dir, 'cifar10.pt'), map_location='cpu')

    # Prepare datasets
    train_base = CIFAR10(root=data_dir, train=True, download=True)
    test_base = CIFAR10(root=data_dir, train=False, download=True)

    X = torch.cat([
        torch.tensor(train_base.data).float().reshape(-1,32,32,3) / 255.,
        torch.tensor(test_base.data).float().reshape(-1,32,32,3) / 255.
    ], 0)
    y = torch.cat([torch.tensor(train_base.targets), torch.tensor(test_base.targets)], 0)

    indices_all = torch.where(torch.isin(y, included_classes))
    X, y, X_preprocessed = X[indices_all], y[indices_all], X_preprocessed[indices_all]

    if not scar:
        o = torch.zeros_like(y)
        for pos_class in positive_classes:
            # Get positive distribution
            p_indices = torch.where(y == pos_class)[0]
            X_pos = X[p_indices]

            # Sample positive distribution
            num_labeled = int(len(X_pos) * label_frequency)

            redness = ((X_pos[:, :, :, 0] - X_pos[:, :, :, 1]) + (X_pos[:, :, :, 0] - X_pos[:, :, :, 1])) / 2
            red_sort = torch.argsort(redness.reshape(-1, 32 * 32).sum(axis=1), descending=True)
            most_red_pos_indices = red_sort[0:num_labeled]
            l_indices = p_indices[most_red_pos_indices]
            
            o[l_indices] = 1
    
    y = torch.where(torch.isin(y, positive_classes), 1, -1)

    if scar:
        o = torch.where(
            y == 1, 
            torch.where(
                torch.rand_like(y, dtype=float) < label_frequency, 
                1,
                0
            ),
            0
        ) # label without bias

    X = X_preprocessed
    
    # Split datasets
    n = X.shape[0]
    n_val = int(val_size * n)
    n_test = int(test_size * n)
    n_train = n - n_val - n_test

    shuffled_indices = torch.randperm(X.shape[0])
    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train:(n_train+n_val)]
    test_indices = shuffled_indices[(n_train+n_val):]

    x_train, y_train, o_train = X[train_indices], y[train_indices], o[train_indices]
    x_val, y_val, o_val = X[val_indices], y[val_indices], o[val_indices]
    x_test, y_test, o_test = X[test_indices], y[test_indices], o[test_indices]

    train = x_train.numpy(), y_train.numpy(), o_train.numpy()
    val = x_val.numpy(), y_val.numpy(), o_val.numpy()
    test = x_test.numpy(), y_test.numpy(), o_test.numpy()

    pi_p = (y == 1).float().mean().numpy()
    label_frequency = (o == 1).float().mean().numpy() / pi_p
    n_input = X.shape[1]

    return train, val, test, label_frequency, pi_p, n_input

def __get_pu_stl_precomputed(included_classes, scar, positive_classes, label_frequency,
        data_dir='./data/', val_size=0.15, test_size=0.15):
    included_classes = torch.tensor(included_classes)
    positive_classes = torch.tensor(positive_classes)

    X_preprocessed = torch.load(os.path.join(data_dir, 'stl10.pt'), map_location='cpu')

    # Prepare datasets
    train_base = STL10(root=data_dir, split='train', download=True)
    test_base = STL10(root=data_dir, split='test', download=True)

    X = torch.cat([
        torch.tensor(train_base.data).float().reshape(-1,32,32,3) / 255.,
        torch.tensor(test_base.data).float().reshape(-1,32,32,3) / 255.
    ], 0)
    y = torch.cat([torch.tensor(train_base.labels), torch.tensor(test_base.labels)], 0)

    indices_all = torch.where(torch.isin(y, included_classes))
    X, y, X_preprocessed = X[indices_all], y[indices_all], X_preprocessed[indices_all]

    if not scar:
        o = torch.zeros_like(y)
        for pos_class in positive_classes:
            # Get positive distribution
            p_indices = torch.where(y == pos_class)[0]
            X_pos = X[p_indices]

            # Sample positive distribution
            num_labeled = int(len(X_pos) * label_frequency)

            redness = ((X_pos[:, :, :, 0] - X_pos[:, :, :, 1]) + (X_pos[:, :, :, 0] - X_pos[:, :, :, 1])) / 2
            red_sort = torch.argsort(redness.reshape(-1, 32 * 32).sum(axis=1), descending=True)
            most_red_pos_indices = red_sort[0:num_labeled]
            l_indices = p_indices[most_red_pos_indices]
            
            o[l_indices] = 1
    
    y = torch.where(torch.isin(y, positive_classes), 1, -1)

    if scar:
        o = torch.where(
            y == 1, 
            torch.where(
                torch.rand_like(y, dtype=float) < label_frequency, 
                1,
                0
            ),
            0
        ) # label without bias

    X = X_preprocessed
    
    # Split datasets
    n = X.shape[0]
    n_val = int(val_size * n)
    n_test = int(test_size * n)
    n_train = n - n_val - n_test

    shuffled_indices = torch.randperm(X.shape[0])
    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train:(n_train+n_val)]
    test_indices = shuffled_indices[(n_train+n_val):]

    x_train, y_train, o_train = X[train_indices], y[train_indices], o[train_indices]
    x_val, y_val, o_val = X[val_indices], y[val_indices], o[val_indices]
    x_test, y_test, o_test = X[test_indices], y[test_indices], o[test_indices]

    train = x_train.numpy(), y_train.numpy(), o_train.numpy()
    val = x_val.numpy(), y_val.numpy(), o_val.numpy()
    test = x_test.numpy(), y_test.numpy(), o_test.numpy()

    pi_p = (y == 1).float().mean().numpy()
    label_frequency = (o == 1).float().mean().numpy() / pi_p
    n_input = X.shape[1]

    return train, val, test, label_frequency, pi_p, n_input

def get_dataset(name, device, label_frequency,
        data_dir='./data/',
        val_size=0.15, test_size=0.15, 
        use_scar_labeling=False):
    if 'MNIST' in name:
        if '3v5' in name:
            included_classes = [3, 5]
            positive_classes = [3]
        elif 'OvE' in name:
            included_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            positive_classes = [1, 3, 5, 7, 9]
        else:
            raise Exception('Dataset not supported')
        
        return __get_pu_mnist(
            included_classes = included_classes,
            positive_classes = positive_classes,
            scar=use_scar_labeling,
            label_frequency=label_frequency,
            data_dir=data_dir,
            val_size=val_size,
            test_size=test_size
        )
    elif 'CIFAR' in name:
        if 'CarTruck' in name:
            included_classes = [1, 9]
            positive_classes = [1]
        elif 'CarVsRest' in name:
            included_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            positive_classes = [1]
        elif 'MachineAnimal' in name:
            included_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            positive_classes = [0, 1, 8, 9]
        else:
            raise Exception('Dataset not supported')
        
        return __get_pu_cifar10_precomputed(
            included_classes = included_classes,
            positive_classes = positive_classes,
            scar=use_scar_labeling,
            label_frequency=label_frequency,
            data_dir=data_dir,
            val_size=val_size,
            test_size=test_size
        )
    elif 'STL' in name:
        if 'CarTruck' in name:
            included_classes = [1, 9]
            positive_classes = [1]
        elif 'CarVsRest' in name:
            included_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            positive_classes = [1]
        elif 'MachineAnimal' in name:
            included_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            positive_classes = [0, 1, 8, 9]
        else:
            raise Exception('Dataset not supported')

        return __get_pu_stl_precomputed(
            included_classes = included_classes,
            positive_classes = positive_classes,
            scar=use_scar_labeling,
            label_frequency=label_frequency,
            data_dir=data_dir,
            val_size=val_size,
            test_size=test_size
        )
    elif name in DATASET_NAMES:
        return get_small_dataset(
            name=name,
            device=device,
            scar=use_scar_labeling,
            label_frequency=label_frequency,
            val_size=val_size,
            test_size=test_size
        )
    else:
        raise Exception('Dataset not supported')


def create_vae_pu_adapter(train, val, test, device='cuda'):
    x_train, y_train, s_train = train
    x_val, y_val, s_val = val
    x_test, y_test, s_test = test

    x_train, y_train, s_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float(), torch.from_numpy(s_train).float()
    x_val, y_val, s_val = torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float(), torch.from_numpy(s_val).float()
    x_test, y_test, s_test = torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float(), torch.from_numpy(s_test).float()

    y_train = torch.where(y_train == 1, 1, -1)
    y_val = torch.where(y_val == 1, 1, -1)
    y_test = torch.where(y_test == 1, 1, -1)

    l_indices = torch.where(s_train == 1)
    u_indices = torch.where(s_train == 0)

    x_tr_l, y_tr_l = x_train[l_indices], y_train[l_indices]
    x_tr_u, y_tr_u = x_train[u_indices], y_train[u_indices]

    return x_tr_l.to(device), y_tr_l.to(device), x_tr_u.to(device), y_tr_u.to(device), \
        x_val.to(device), y_val.to(device), \
        x_test.to(device), y_test.to(device)
        