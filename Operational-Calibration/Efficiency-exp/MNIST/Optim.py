import numpy as np


def nag(obj_func, initial_theta, bounds):
      # Nesterov accerlate gradient descent
    l0 = 1e10
    x = initial_theta
    y0 = x
    max_iter = 1000
    step_size = 1e-2
    mu, t = 0.9, 0
    while(t < max_iter):
        t += 1
        l1, grad = obj_func(x, eval_gradient=True)
        if(np.abs(l0 - l1) < 1e-8):
            break
        elif l1 >= l0:
            x = orig_x
            l1 = l0
            break
        else:
            orig_x = x
            y1 = x - step_size * grad
            x = y1 + mu * (y1 - y0)
            x = np.clip(x, bounds[:, 0], bounds[:, 1])
            l0 = l1
    return x, l1

def adam(obj_func, initial_theta, bounds):
    # # adam optimizer
    x = initial_theta
    l0 = 1e10
    step_size = 0.1
    max_iter = 100
    alpha, beta1, beta2, epsilon = step_size, 0.9, 0.999, 1e-8
    m, v, t = 0, 0, 0
    while(t < max_iter):
        t += 1
        l1, grad = obj_func(x, eval_gradient=True)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * np.square(grad)
        temp_m = m / (1 - (beta1 ** t))
        temp_v = v / (1 - (beta2 ** t))
        delta = temp_m / (np.sqrt(temp_v) + epsilon)
        new_x = x - alpha * delta.reshape(x.shape)
        new_x = np.clip(new_x, bounds[:, 0], bounds[:, 1])
        if(np.linalg.norm((new_x - x).reshape(-1), 2) < 1e-3):
            break
        elif l1 >= l0:
            break
        else:
            l0 = l1
            x = new_x
    return x, l1
