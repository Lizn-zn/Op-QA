import numpy as np
import torch
from torch.nn import functional as F
from scipy.stats import norm

alpha = 1e-1
length_scale = 10
sigma_0 = 0.1


def conf_build(model, x, center_):
    '''
    build GP for c
    :param model: the DNN model
    :param x: the data in representation space 
    :param center_: the cluster center in representation space
    : return gp: the GP model
    '''

    # get them cluster label
    from sklearn.metrics.pairwise import pairwise_distances
    fc_output = x.detach().numpy()
    dist = pairwise_distances(center_, fc_output, metric='euclidean')
    cluster_label = np.argmin(dist, axis=0)

    # get the confidences for x
    softmaxes = F.softmax(model.classifier(x.cuda()), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    predictions = predictions.cpu().detach().numpy()
    confidences = confidences.cpu().detach().numpy()
    confidences = np.clip(confidences, 1e-3, 1 - 1e-3)
    # confidences = np.log(confidences)

    # build GP for each cluster
    clf = []
    for k in range(center_.shape[0]):
        cluster_index = np.where(cluster_label == k)

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct
        import Optim
        opt = Optim.adam
        kernel = ConstantKernel(
            1) * RBF(length_scale=length_scale, length_scale_bounds=(1e-5, 1e5))

        gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=alpha, optimizer=opt)

        temp_y = confidences[cluster_index]
        temp_x = x[cluster_index].detach().numpy()
        gp.fit(temp_x, temp_y)
        clf.append(gp)
    return clf


def ratio_build(model, x, x_select, y_select, select_index, center_):
    '''
    build GP for r
    :param model: the DNN model
    :param x: the data in representation space 
    :param x_select: the selected data in representation space
    :param y_select: the selected data's label
    :param select_index: the selected index
    : return gp: the GP model
    '''
    # get selected inputs label
    select = x_select.detach().numpy()
    from sklearn.metrics.pairwise import pairwise_distances
    dist = pairwise_distances(center_, select, metric='euclidean')
    cluster_label = np.argmin(dist, axis=0)

    # get conf ratio
    softmaxes = F.softmax(model.classifier(x_select.cuda()), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    predictions = predictions.cpu().detach().numpy()
    confidences = confidences.cpu().detach().numpy()
    confidences = np.clip(confidences, 1e-3, 1 - 1e-3)
    temp = np.clip((predictions == y_select.detach().numpy()), 1e-3, 1 - 1e-3)
    # regress = np.log(temp) - np.log(confidences)
    regress = temp - confidences

    clf = []
    for k in range(center_.shape[0]):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct
        import Optim
        opt = Optim.adam
        kernel = ConstantKernel(
            1) * RBF(length_scale=length_scale, length_scale_bounds=(1e-5, 1e5))
        # kernel = ConstantKernel(1) * RBF(length_scale=np.ones((84,)))
        gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=alpha, optimizer=opt)

        temp_index = np.where(cluster_label == k)
        temp_x = x_select[temp_index].detach().numpy()
        temp_y = regress[temp_index]
        # temp_kernel = gp.kernel_
        # gp.operimizer = None
        gp.fit(temp_x, temp_y)
        clf.append(gp)
    return clf

def truncated_mean(mean, std, lower=-1, upper=1):
    alpha = (lower - mean) / std
    beta = (upper - mean) / std
    Z = (norm.cdf(beta) - norm.cdf(alpha)) + 1e-10
    mu = mean - (norm.pdf(beta) - norm.pdf(alpha)) / Z * std
    var = np.square(std) * (1 + (alpha * norm.pdf(alpha) - beta * norm.pdf(beta)
                                 ) / Z - np.square((norm.pdf(alpha) - norm.pdf(beta)) / Z))
    return mu, np.sqrt(var)

def opc_predict(model, clf, x_new, center_):
    '''
    build GP for operational confidence
    :param model: the DNN model
    :param clf1: the GP for c
    :param clf2: the GP for ratio
    :param x_new: the predictive x_new 
    : return prediction and std
    '''

    # get cluster_label

    softmaxes = F.softmax(model.classifier(x_new.cuda()), dim=1)
    confidences, _ = torch.max(softmaxes, 1)
    confidences = confidences.cpu().detach().numpy()

    x_new = x_new.cpu().detach().numpy()

    from sklearn.metrics.pairwise import pairwise_distances
    dist = pairwise_distances(center_, x_new, metric='euclidean')
    cluster_label = np.argmin(dist, axis=0)

    pred = []
    std = []
    for k in range(x_new.shape[0]):
        i = cluster_label[k]
        temp = x_new[k].reshape(1, -1)
        gp = clf[i]
        r, std1 = gp.predict(temp, return_std=True)
        c = confidences[k]
        r, s = gp.predict(temp, return_std=True)
        r, s = truncated_mean(r, s, lower=0 - c, upper=1 - c)
        # r, s = truncated_mean(r, s, lower=-1, upper=1)
        pred.append(c + r)
        std.append(s)


    pred = np.array(pred).reshape(-1)
    std = np.array(std).reshape(-1)
    return pred, std
