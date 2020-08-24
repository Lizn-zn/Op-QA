import numpy as np


def input_select(cluster, x_op, ws, init_size, rand_select=True):
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
    if rand_select == True:
        # # random select
        select_index = []
        random = np.random.permutation(x_op.shape[0])
        select_index.append(random[0:init_size])
        select_index = np.array(select_index).reshape(-1)
    else:
        labels = cluster.predict(x_op.cpu().detach().numpy())
        select_index = np.array([]).astype('int')
        # ========# ========# ========# ========# ========
        for i in range(ws.shape[0]):
            if ws[i] < 0.1:
                continue
            else:
                ind = np.where(labels == i)[0]
                w = int(ws[i]*init_size) - 1
                random = np.random.permutation(ind.shape[0])
                select_index = np.append(select_index, ind[random[0:w]])
        # ========# ========# ========# ========# ========
        if select_index.shape[0] < init_size:
            num = int(init_size / np.sum(ws > 0))
            for i in range(ws.shape[0]):
                if ws[i] == 0:
                    continue
                else:
                    ind = np.where(labels == i)[0]
                    random = np.random.permutation(ind.shape[0])
                    num = np.minimum(num, ind.shape[0])
                    select_index = np.append(select_index, ind[random[0:num]])      
        #   
        select_index = np.array(select_index).reshape(-1)
        if select_index.shape[0] < init_size:
            diff = init_size - select_index.shape[0]
            tmp = input_select(cluster, x_op, ws, diff, rand_select=True)
            select_index = np.append(select_index, tmp)
    return select_index
