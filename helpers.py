import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve, f1_score, \
    mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

# Define the function to print and write log
def writelog(file, line):
    file.write(line + '\n')
    print(line)

# Get current learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Change View via Transpose
# 3D
def change_view_3D(x, args, transpose=False):
    x_ = torch.flatten(x.contiguous().view(-1,
                                           args.som_map_size[0],
                                           args.som_map_size[1],
                                           args.som_map_size[2]).transpose(3, 2), start_dim=1, end_dim=2)
    if transpose:
        x_ = x.t()
    return x_

# 2D
def change_view(x, args, transpose=False):
    x_ = torch.flatten(x.contiguous().view(-1,
                                           args.som_map_size[1],
                                           args.som_map_size[2]).transpose(2, 1), start_dim=1, end_dim=2)
    if transpose:
        x_ = x.t()
    return x_

# Define loeader with quartet sample
def get_quartet_loader(args, mris, demographic, labels, index, seed=0):

    # Tensor Seed
    seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Make quartet with boosting
    idxs = {}
    for i, l in enumerate(np.unique(labels[index])):
        lidx = index[labels[index]==l]
        idxs[i] = np.array(lidx[np.random.permutation(lidx.shape[0])]).flatten()
    idxs_n = np.array([idxs[l].shape[0] for l in np.unique(labels[index])])
    idxs_boosted = idxs.copy()
    for i, n in enumerate([idxs_n.max() - i_n for i_n in idxs_n]):
        idxs_boosted[i] = np.hstack([idxs_boosted[i], idxs[i][np.random.randint(idxs_n[i], size=n)]])
    if args.spectrum_dim == 3:
        quartetindex = np.array([[idxs_boosted[0][i],
                                  idxs_boosted[1][i],
                                  idxs_boosted[2][i]] for i in range(idxs_n.max())]).flatten()
    else:
        quartetindex = np.array([[idxs_boosted[0][i],
                                  idxs_boosted[1][i],
                                  idxs_boosted[2][i],
                                  idxs_boosted[3][i]] for i in range(idxs_n.max())]).flatten()

    if args.load_images:
        records = [{'data': torch.from_numpy(np.array(mris[i])),
                    'label': torch.from_numpy(np.array(labels[i])),
                    'demographic': torch.from_numpy(np.array(demographic[i])),
                    'index': torch.from_numpy(np.array(i)),} for i in quartetindex]
    else:
        records = [{'label': torch.from_numpy(np.array(labels[i])),
                    'demographic': torch.from_numpy(np.array(demographic[i])),
                    'index': torch.from_numpy(np.array(i)),} for i in quartetindex]

    # Define loader
    loader = DataLoader(records,
                        batch_size=args.batch_size,
                        num_workers=1,
                        shuffle=False,
                        drop_last=False)
    return loader

# Define loeader without quartet sample
def get_loader(args, image_ids, mris, demographic, labels, index, shuffle=False, train_boost=False):

    # Tensor Seed
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if train_boost:
        idxs = {}
        for i, l in enumerate(np.unique(labels[index])):
            lidx = index[labels[index]==l]
            idxs[i] = np.array(lidx[np.random.permutation(lidx.shape[0])]).flatten()
        idxs_n = np.array([idxs[l].shape[0] for l in np.unique(labels[index])])
        idxs_boosted = idxs.copy()
        for i, n in enumerate([idxs_n.max() - i_n for i_n in idxs_n]):
            idxs_boosted[i] = np.hstack([idxs_boosted[i], idxs[i][np.random.randint(idxs_n[i], size=n)]])
        if idxs_n.shape[0] == 2:
            index = np.array([[idxs_boosted[0][i], idxs_boosted[1][i]] for i in range(idxs_n.max())]).flatten()
        else:
            index = np.array([[idxs_boosted[0][i],
                               idxs_boosted[1][i],
                               idxs_boosted[2][i],
                               idxs_boosted[3][i]
                               ] for i in range(idxs_n.max())]).flatten()


    if args.load_images:
        records = [{'data': torch.from_numpy(np.array(mris[i])),
                    'label': torch.from_numpy(np.array(labels[i])),
                    'demographic': torch.from_numpy(np.array(demographic[i])),
                    'image_id': torch.from_numpy(np.array(image_ids[i])),
                    'index': torch.from_numpy(np.array(i)),} for i in index]
    else:
        records = [{'label': torch.from_numpy(np.array(labels[i])),
                    'demographic': torch.from_numpy(np.array(demographic[i])),
                    'image_id': torch.from_numpy(np.array(image_ids[i])),
                    'index': torch.from_numpy(np.array(i)),} for i in index]

    # Define loader
    loader = DataLoader(records,
                        batch_size=args.batch_size,
                        num_workers=1,
                        shuffle=shuffle,
                        drop_last=False)

    return loader

# Define sample loader CN/SMCI/PMCI/AD
def smri_batchloader(args, is_quartet=False, is_train_boost=False, seed=0):

    # Tensor Seed
    seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    mris = np.memmap(filename='/data/MRIs.npy', mode="r", shape=(1540, 3, 96, 114, 96), dtype=np.float32)
    labels = np.load('/data/labels.npy', allow_pickle=True).astype(np.int32) # Dimensionality: 1540
    image_ids = np.load('/data/image_ids.npy', allow_pickle=True) # Dimensionality: 1540
    demographics = np.load('/data/new_clinicals.npy', allow_pickle=True) # Dimensionality: 1540 X 7

    # Label Class Correction
    labels[labels == 2] = 9
    labels[labels == 1] = 2
    labels[labels == 9] = 1

    # Define Task
    mytask = []
    is_MCI = False
    # One-hot encoding for the disease label
    labels_ohe = np.eye(4)[labels]
    for t in args.task:
        if t == 'CN':
            mytask.append(0)

        if t == 'SMCI':
            mytask.append(1)

        if t == 'PMCI':
            mytask.append(2)

        if t == 'AD':
            mytask.append(3)

        if t == 'MCI':
            mytask.append(1)
            mytask.append(2)
            is_MCI = True
    mytask = np.array(mytask)

    # Extract appropriate indicies for the given task
    task_idx = np.zeros(shape=labels.shape)

    labels_ = labels.copy()
    for t in range(len(mytask)):
        task_idx += np.array(labels == mytask[t])
        labels_[np.array(labels == mytask[t])] = t
    task_idx = task_idx.astype(bool)

    image_ids_ = image_ids[task_idx]
    mris_ = mris[task_idx]
    demographics_ = demographics[task_idx]
    labels_ = labels_[task_idx]
    labels_ohe = labels_ohe[task_idx]

    # Folding
    np.random.seed(0)
    tr_idx = np.empty([], dtype=np.int)
    vd_idx = np.empty([], dtype=np.int)
    ts_idx = np.empty([], dtype=np.int)
    for i, l in enumerate(np.unique(labels_)):
        idx_l = np.squeeze(np.argwhere(labels_ == l))
        n_l = int(len(idx_l) / 5)
        idx_vd_ = idx_l[(args.fold-1) * n_l:args.fold * n_l]
        idx_tr_ = np.setdiff1d(idx_l, idx_vd_)
        idx_ts_ = idx_vd_[:int(len(idx_vd_) / 2)]
        idx_vd_ = np.setdiff1d(idx_vd_, idx_ts_)

        if i == 0:
            tr_idx = idx_tr_
            vd_idx = idx_vd_
            ts_idx = idx_ts_
        else:
            tr_idx = np.hstack([tr_idx, idx_tr_])
            vd_idx = np.hstack([vd_idx, idx_vd_])
            ts_idx = np.hstack([ts_idx, idx_ts_])

    # Label correction
    if is_MCI :
        if ('/').join(args.task) == 'CN/MCI/AD':
            labels_[labels_ == 2] = 1
            labels_[labels_ == 3] = 2
        if ('/').join(args.task) == 'MCI/AD':
            labels_[labels_ == 1] = 0
            labels_[labels_ == 2] = 1
        if ('/').join(args.task) == 'CN/MCI':
            labels_[labels_ == 2] = 1

    # Demographic information concatenation
    demographics_ = np.concatenate((labels_ohe, demographics_[:, 3:]), -1) # Dimensionality: 1540 x 8

    # Demographics Gaussian Normalization (MMSE, Age, Edu)
    means = np.array([np.mean(demographics_[tr_idx, 4]),
                      np.mean(demographics_[tr_idx, 5]),
                      np.mean(demographics_[tr_idx, 7])])
    stds = np.array([np.std(demographics_[tr_idx, 4]),
                     np.std(demographics_[tr_idx, 5]),
                     np.std(demographics_[tr_idx, 7])])
    mins = np.array([np.min(demographics_[tr_idx, 4]),
                     np.min(demographics_[tr_idx, 5]),
                     np.min(demographics_[tr_idx, 7])])
    maxs = np.array([np.max(demographics_[tr_idx, 4]),
                     np.max(demographics_[tr_idx, 5]),
                     np.max(demographics_[tr_idx, 7])])

    demographics_[tr_idx, 4] = (demographics_[tr_idx, 4] - means[0]) / stds[0]
    demographics_[tr_idx, 5] = (demographics_[tr_idx, 5] - means[1]) / stds[1]
    demographics_[tr_idx, 7] = (demographics_[tr_idx, 7] - means[2]) / stds[2]

    demographics_[vd_idx, 4] = (demographics_[vd_idx, 4] - means[0]) / stds[0]
    demographics_[vd_idx, 5] = (demographics_[vd_idx, 5] - means[1]) / stds[1]
    demographics_[vd_idx, 7] = (demographics_[vd_idx, 7] - means[2]) / stds[2]

    demographics_[ts_idx, 4] = (demographics_[ts_idx, 4] - means[0]) / stds[0]
    demographics_[ts_idx, 5] = (demographics_[ts_idx, 5] - means[1]) / stds[1]
    demographics_[ts_idx, 7] = (demographics_[ts_idx, 7] - means[2]) / stds[2]

    if is_quartet: # CN/SMCI/PMCI/AD
        train_loader = get_quartet_loader(args, mris_, demographics_, labels_, tr_idx, seed=seed)
        valid_loader = get_quartet_loader(args, mris_, demographics_, labels_, vd_idx)
        test_loader = get_quartet_loader(args, mris_, demographics_, labels_, ts_idx)
    else:
        train_loader = get_loader(args, image_ids_, mris_, demographics_, labels_, tr_idx, shuffle=True, train_boost=is_train_boost)
        valid_loader = get_loader(args, image_ids_, mris_, demographics_, labels_, vd_idx)
        test_loader = get_loader(args, image_ids_, mris_, demographics_, labels_, ts_idx)

    dataloaders = {'train': train_loader,
                   'valid': valid_loader,
                   'test': test_loader}

    return dataloaders, means, stds, mins, maxs

# Define function to normalize the data using Gaussian with augmentation
def normalize_gauss_aug_on(device, x_):
    x = torch.empty((x_.shape[0], 1, 96, 114, 96)).to(device)
    for i in range(x.shape[0]):
        x[i] = x_[i][0].unsqueeze(0)

        pad = nn.ConstantPad3d(5, 0)
        with tf.device('/cpu:0'):
            x[i] = torch.from_numpy(tf.image.random_crop(pad(x[i]).detach().to('cpu').numpy(), x[i].shape).numpy()).to(device)

        # Quantile normalization
        Q1, Q3 = torch.quantile(x[i], 0.1), torch.quantile(x[i], 0.9)
        x[i] = torch.where(x[i] < Q1, Q1, x[i])
        x[i] = torch.where(x[i] > Q3, Q3, x[i])

        # Gaussian normalization
        m, std = x[i].mean(), x[i].std()
        x[i] = (x[i] - m) / std

    x = x.float().to(device)

    return x

# Define function to normalize the data using Gaussian without augmentation
def normalize_gauss(device, x_):
    x = torch.empty((x_.shape[0], 1, 96, 114, 96)).to(device)
    for i in range(x.shape[0]):
        x[i] = x_[i][0].unsqueeze(0)

        # Quantile normalization
        Q1, Q3 = torch.quantile(x[i], 0.1), torch.quantile(x[i], 0.9)
        x[i] = torch.where(x[i] < Q1, Q1, x[i])
        x[i] = torch.where(x[i] > Q3, Q3, x[i])

        # Gaussian normalization
        m, std = x[i].mean(), x[i].std()
        x[i] = (x[i] - m) / std

    x = x.float().to(device)

    return x

# Calculate classification scores for multi-class case
def get_metric_multi(y, y_hat, y_prob):
    auprc = 0
    acc = accuracy_score(y, y_hat) * 100
    a, b, c, \
    d, e, f, \
    g, h, i = confusion_matrix(y, y_hat, labels=np.arange(3)).ravel()
    tn, fp, fn, tp = (a + c + g + i), (b + h), (d + f), (e)
    if tp == 0 and fn == 0:
        sen = 0.0
        recall = 0.0
    else:
        sen = tp / (tp + fn)
        recall = tp / (tp + fn)
        p, r, t = 0, 0, 0
    spec = np.nan_to_num(tn / (tn + fp))
    balacc = ((spec + sen) / 2) * 100
    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = np.nan_to_num(tp / (tp + fp))

    try:
        labels = np.arange(3)
        auc = roc_auc_score(y, y_prob, multi_class='ovr')
    except ValueError:
        auc = 0
    f1 = f1_score(y, y_hat, average='weighted')

    return auc, auprc, acc, balacc, sen, spec, prec, recall, f1

# Calculate classification scores for binary case
def get_metric_binary(y, y_hat, y_prob):
    acc = accuracy_score(y, y_hat) * 100
    tn, fp, fn, tp = confusion_matrix(y, y_hat, labels=[0, 1]).ravel()
    if tp == 0 and fn == 0:
        sen = 0.0
        recall = 0.0
        auprc = 0.0
    else:
        sen = tp / (tp + fn)
        recall = tp / (tp + fn)
        p, r, t = precision_recall_curve(y, y_hat)
        auprc = np.nan_to_num(metrics.auc(r, p))
    spec = np.nan_to_num(tn / (tn + fp))
    balacc = ((spec + sen) / 2) * 100
    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = np.nan_to_num(tp / (tp + fp))

    try:
        auc = roc_auc_score(y, y_prob)
    except ValueError:
        auc = 0
    f1 = f1_score(y, y_hat)

    return auc, auprc, acc, balacc, sen, spec, prec, recall, f1

# Define function to measure the evaluation metric for classification
def calculate_performance(y, y_hat, y_prob, args):
    metric = {}

    if ('/').join(args.task) == 'CN/SMCI/PMCI/AD' or ('/').join(args.task) == 'CN/MCI/AD':
        metric[('/').join(args.task)] = get_metric_multi(y, y_hat, y_prob)
    else:
        metric[('/').join(args.task)] = get_metric_binary(y, y_hat, y_prob)

    return metric

# Calculate regression scores
def get_metric_reg(y, y_pred):
    mse = mean_squared_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, rmse, mae, r2

# Define function to measure the evaluation metric for regression
def calculate_performance_reg(y, y_score, args):
    metric = {}
    metric[('/').join(args.task)] = get_metric_reg(y, y_score)
    return metric
