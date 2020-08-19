import numpy as np


def input_initiation(x_op, y_op, M, C, init_size):
    '''
    evaluate the model for profit
    :param x_op: operational data in representation space
    :param y_op: operational label
    :param init_size: size for initial examples
    :param iteration: iterative size
    : return x_select: the selected data
    : return y_select: the select data's label
    : return select_index: the selected index
    '''
    fc_output = x_op.detach().numpy()
    if init_size <= M.shape[0]:
        select_index = M[0:init_size]
        x_select = x_op[select_index]
        y_select = y_op[select_index]
    else:
        # # random select
        select_index = []
        for k in range(init_size):
            ind = C[k % M.shape[0]]
            random = np.random.permutation(ind.shape[0])
            select_index.append(ind[random[-1]])
        select_index = np.array(select_index).reshape(-1)
        x_select = x_op[select_index]
        y_select = y_op[select_index]
    return x_select, y_select, select_index
