import numpy as np


def input_selection(x_new, pred, std, lamda, select_size, rand_select=True):
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
        # l = np.abs(pred - lamda)
        deter = int(select_size * 1.0)
        index1 = np.where(pred < lamda)
        index2 = np.where(pred >= lamda)
        size1 = np.minimum(index2[0].shape[0], deter)
        size2 = select_size - size1
        ind = np.argsort(std[index1])
        ind1 = ind[-size1:]
        if size2 == 0:
            temp = ind1.reshape(-1)
            return x_new[temp], temp
        ind = np.argsort(std[index2])
        ind2 = ind[-size2:]
        temp = np.append(ind1, ind2).reshape(-1)
    return x_new[temp], temp
