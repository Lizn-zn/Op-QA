import torch
import numpy as np


def kernel_matrix(x_op, center_, clf):
    fc = x_op.detach().numpy()
    from sklearn.metrics.pairwise import pairwise_distances
    dist = pairwise_distances(center_, fc, metric='euclidean')
    cluster_label = np.argmin(dist, axis=0)
    Cov = []
    for k in range(center_.shape[0]):
        kernel = clf[k].kernel
        index = np.where(cluster_label == k)
        Cov.append(kernel(fc[index], fc[index]))

    return Cov
