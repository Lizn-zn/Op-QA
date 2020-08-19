import numpy as np

def input_selection(x_new, select_size, rand_select=True):
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
        return
    return x_new[temp], temp