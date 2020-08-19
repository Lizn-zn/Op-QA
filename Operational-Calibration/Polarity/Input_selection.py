import numpy as np


def input_selection(x, x_new, conf, std, lamda, select_size, rand_select=True):
    '''
    select input for update GP for ration
    :param x_new: the new input for selection
    :param select_size: the size of selection
    : return selected input and its index
    '''
    if rand_select == True:
        ind = np.random.permutation(x_new.shape[0])
        temp = ind[-select_size:]
    else:
        score = np.square(std) / np.square(lamda - conf)
        # score = np.square(std)
        temp = np.argsort(score)[-select_size:]
    return x_new[temp], temp

    # error = np.square(pred - conf)[noselected]
    # error = np.square(pred - 0.5)[noselected]

    # index1 = np.where(pred[noselected] > lamda)[0]
    # index2 = np.where(pred[noselected] <= lamda)[0]
    # t = np.random.rand(1)
    # if t < 0.9:
    #     deter = int(select_size * 1.0)
    # else:
    #     deter = int(select_size * 0.0)
    # size1 = np.minimum(index1.shape[0], deter)
    # size2 = select_size - size1
    # ind = np.argsort(score[index1])
    # ind1 = ind[-size1:]
    # if size2 <= 0:
    #     temp = index1[ind1].reshape(-1)
    #     return x_new[temp], temp
    # ind = np.argsort(score[index2])
    # ind2 = ind[-size2:]
    # if size1 <= 0:
    #     temp = index2[ind2].reshape(-1)
    #     return x_new[temp], temp
    # temp = np.append(index1[ind1], index2[ind2]).reshape(-1)
