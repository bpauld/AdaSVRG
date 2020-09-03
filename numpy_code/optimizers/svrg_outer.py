import math
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import pickle
from sklearn import datasets, metrics

import urllib
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file

def svrg_armijo_outer_end(closure,
                          batch_size,
                          D,
                          labels,
                          n,
                          d,
                          c,
                          beta,
                          max_iter_armijo,
                          max_step_size,
                          reset_step_size=False,
                          max_epoch=100,
                          m=0,
                          x0=None,
                          verbose=True):
    """
        SVRG with Armijo step size at the end of the outer loop for solving finite-sum problems
        Closure: a PyTorch-style closure returning the objective value and it's gradient.
        batch_size: the size of minibatches to use.
        D: the set of input vectors (usually X).
        labels: the labels corresponding to the inputs D.
        init_step_size: initial step size
        n, d: size of the problem
        max_iter_armijo: maximum number of Armijo iterations before falling back to small determined stepsize
        reset_step_size: whether or not the stepsize should be reset to max_step_size at the next iteration
        beta: multiplicative factor in Armijo line search (0 < beta < 1).
    """
    if not isinstance(m, int) or m <= 0:
        m = n
        if verbose:
            print('Info: set m=n by default')

    if x0 is None:
        x = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d,):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    start_step_size = max_step_size
    LOSS = np.zeros((max_epoch))
    GRAD_NORM = np.zeros((max_epoch))
    NB_ARMIJO_STEPS = np.zeros((max_epoch))

    for k in range(max_epoch):
        loss, full_grad = closure(x, D, labels)
        x_tilde = x.copy()

        last_full_grad = full_grad
        last_x_tilde = x_tilde
        norm_full_grad = np.linalg.norm(full_grad)

        if verbose:
            output = 'Epoch.: %d, Step size: %.2e, Grad. norm: %.2e' % \
                     (k, start_step_size, norm_full_grad)
            output += ', Func. value: %e' % loss
            print(output)

        LOSS[k] = loss
        GRAD_NORM[k] = norm_full_grad
        step_size = start_step_size

        for j in range(max_iter_armijo):
            # Create Minibatches:
            minibatches = make_minibatches(n, m, batch_size)
            for i in range(m):
                # get the minibatch for this iteration
                indices = minibatches[i]
                Di, labels_i = D[indices, :], labels[indices]

                # compute the gradients:
                f_x, x_grad = closure(x, Di, labels_i)
                x_tilde_grad = closure(x_tilde, Di, labels_i)[1]
                g = x_grad - x_tilde_grad + full_grad
                x -= step_size * g

            if closure(x, D, labels)[0] > loss - c * step_size * norm_full_grad:
                step_size = beta * step_size
                x = x_tilde.copy()
                NB_ARMIJO_STEPS[k] += 1
            else:
                found = True
                break
        AVERAGE_ARMIJO_STEPS[k] += j
        if found:
            if not reset_step_size:
                start_step_size = step_size
            x -= step_size * g
        if not found:
            x -= 1e-6 * full_grad

        output = 'Epoch.: %d, Nb of Armijo steps: %.2e' % \
                 (k, NB_ARMIJO_STEPS[k])
        print(output)

    return x, LOSS, GRAD_NORM