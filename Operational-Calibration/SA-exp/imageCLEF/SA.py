import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.autograd import Variable

from torch.nn import functional as F
from scipy.stats import gaussian_kde

from tqdm import tqdm

import numpy as np


def _get_kdes(x_train, train_pred, class_matrix):
    """Kernel density estimation

    Args:
        train_ats (list): List of activation traces in training set.
        train_pred (list): List of prediction of train set.
        class_matrix (list): List of index of classes.
        args: Keyboard args.

    Returns:
        kdes (list): List of kdes per label if classification task.
        removed_cols (list): List of removed columns by variance threshold.
    """

    removed_cols = []
    for label in range(12):
        col_vectors = np.transpose(x_train[class_matrix[label]])
        for i in range(col_vectors.shape[0]):
            if (
                np.var(col_vectors[i]) < 1e-1
                and i not in removed_cols
            ):
                removed_cols.append(i)

    kdes = {}
    for label in tqdm(range(12), desc="kde"):
        refined_ats = np.transpose(x_train[class_matrix[label]])
        refined_ats = np.delete(refined_ats, removed_cols, axis=0)

        if refined_ats.shape[0] == 0:
            print(
                warn("ats were removed by threshold {}".format(
                    args.var_threshold))
            )
            break
        kdes[label] = gaussian_kde(refined_ats)

    return kdes, removed_cols


def _get_lsa(kde, at, removed_cols):
    refined_at = np.delete(at, removed_cols, axis=0)
    return np.asscalar(-kde.logpdf(np.transpose(refined_at)))


def fetch_lsa(model, x_train, x_target):
    """Likelihood-based SA

    Args:
        model (keras model): Subject model.
        x_train (list): Set of training inputs.
        x_target (list): Set of target (test or[] adversarial) inputs.
        target_name (str): Name of target set.
        layer_names (list): List of selected layer names.
        args: Keyboard args.

    Returns:
        lsa (list): List of lsa for each target input.
    """
    softmaxes = F.softmax(model.classifier(x_train.cuda()), dim=1)
    _, train_pred = torch.max(softmaxes, 1)

    softmaxes = F.softmax(model.classifier(x_target.cuda()), dim=1)
    _, target_pred = torch.max(softmaxes, 1)

    x_train = x_train.cpu().detach().numpy()
    x_target = x_target.cpu().detach().numpy()
    train_pred = train_pred.cpu().detach().numpy()
    target_pred = target_pred.cpu().detach().numpy()

    class_matrix = {}
    for i, label in enumerate(train_pred):
        if label not in class_matrix:
            class_matrix[label] = []
        class_matrix[label].append(i)

    kdes, removed_cols = _get_kdes(x_train, train_pred, class_matrix)

    lsa = []
    for i, at in enumerate(tqdm(x_target)):
        label = target_pred[i]
        kde = kdes[label]
        lsa.append(_get_lsa(kde, at, removed_cols))

    return lsa
