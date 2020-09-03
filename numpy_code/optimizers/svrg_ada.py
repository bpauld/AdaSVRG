from dependencies import *

from utils import *
from datasets import *
from plotting import *
from objectives import *


def compute_statistic(term1, term2, t, x, g, eta, eps = 1e-4):

    term1 = 1.0/t * ((t - 1) * term1 + eta * np.dot(x,g))
    term2 = 1.0/t * ((t - 1) * term2 + (eta**2/2) * np.dot(g, g))

    return term1, term2


def armijo_ls(closure, D, labels, x, loss, grad, init_step_size, c, beta):

    temp_step_size = init_step_size
    armijo_iter = 1
    while closure(x - temp_step_size * grad, D, labels,
                  backwards=False) > loss - c * temp_step_size * np.linalg.norm(grad) ** 2:

        temp_step_size *= beta
        armijo_iter += 1
        if armijo_iter == 50:
            temp_step_size = 1e-6
            break

    step_size = temp_step_size

    return step_size, armijo_iter


def svrg_ada(closure, batch_size, D, labels, init_step_size, n, d, max_epoch=100, m=0, x0=None, verbose=True,
             linesearch_option = 0, max_sgd_warmup_epochs= 0,  adaptive_termination = False):
    """
        SVRG with fixed step size for solving finite-sum problems
        Closure: a PyTorch-style closure returning the objective value and it's gradient.
        batch_size: the size of minibatches to use.
        D: the set of input vectors (usually X).
        labels: the labels corresponding to the inputs D.
        init_step_size: step-size to use
        n, d: size of the problem
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

    LOSS = []
    GRAD_NORM = []
    GRAD_EVALS = []
    num_grad_evals = 0

    term1 = 0
    term2 = 0
    t = 0

    step_size = init_step_size

    # SGD (Adagrad) passes
    for k in range(2 * max_sgd_warmup_epochs):

        if k % 2 == 0:
            loss, full_grad = closure(x, D, labels)
            GRAD_EVALS.append(num_grad_evals)
            LOSS.append(loss)
            GRAD_NORM.append(np.linalg.norm(full_grad))

            if verbose:
                output = 'Epoch.: %d, Grad. norm: %.2e' % \
                         (k, np.linalg.norm(full_grad))
                output += ', Func. value: %e' % loss
                output += ', Num gradient evaluations: %d' % num_grad_evals
                print(output)

        Gk2 = 0
        # Create Minibatches:
        minibatches = make_minibatches(n, m, batch_size)
        for i in range(m):
            # get the minibatch for this iteration
            indices = minibatches[i]
            Di, labels_i = D[indices, :], labels[indices]
            # compute the gradients:
            loss_temp, x_grad = closure(x, Di, labels_i)
            num_grad_evals = num_grad_evals + batch_size
            Gk2 = Gk2 + (np.linalg.norm(x_grad) ** 2)

            if linesearch_option == 2:
                reset_stepsize = step_size * ((2)**(batch_size/n))
                c = 0.5
                beta = 0.9
                step_size, armijo_iter = armijo_ls(closure, Di, labels_i, x, loss_temp, x_grad, reset_stepsize, c, beta)
                num_grad_evals = num_grad_evals + batch_size * armijo_iter

            else:
                step_size = init_step_size

            # computing statistic to decide termination
            # t = t + 1
            # term1, term2 = compute_statistic(term1, term2, t, x, x_grad, step_size)
            # print(np.abs(term1 - term2))

            x -= (step_size / np.sqrt(Gk2)) * x_grad

    for k in range(max_epoch - max_sgd_warmup_epochs):

        loss, full_grad = closure(x, D, labels)
        x_tilde = x.copy()

        # initialize running sum of gradient norms
        Gk2 = 0

        term1 = 0
        term2 = 0

        if linesearch_option == 1 and k > 0:
            # c = 1e-4
            # beta = 0.9
            # reset_step_size = init_step_size
            # step_size, armijo_iter = armijo_ls(closure, D, labels, x, loss, full_grad, reset_step_size, c, beta)
            # num_grad_evals = num_grad_evals + n * armijo_iter

            s = x_tilde - last_x_tilde
            y = full_grad - last_full_grad
            step_size = np.linalg.norm(s) ** 2 / np.dot(s, y) / m

        else:
            step_size = init_step_size

        last_full_grad = full_grad
        last_x_tilde = x_tilde

        if verbose:
            output = 'Epoch.: %d, Grad. norm: %.2e' % \
                     (k, np.linalg.norm(full_grad))
            output += ', Func. value: %e' % loss
            output += ', Num gradient evaluations: %d' % num_grad_evals
            print(output)

        if np.linalg.norm(full_grad) <= 1e-10:
            GRAD_EVALS[k + 1:max_epoch] = [0] * (max_epoch - k)
            LOSS[k + 1:max_epoch] = [0] * (max_epoch - k)
            GRAD_NORM[k + 1:max_epoch] = [0] * (max_epoch - k)
            return x, GRAD_EVALS, LOSS, GRAD_NORM

        num_grad_evals = num_grad_evals + n

        GRAD_EVALS.append(num_grad_evals)
        LOSS.append(loss)
        GRAD_NORM.append(np.linalg.norm(full_grad))

        # Create Minibatches:
        minibatches = make_minibatches(n, m, batch_size)
        for i in range(m):
            # get the minibatch for this iteration
            indices = minibatches[i]
            Di, labels_i = D[indices, :], labels[indices]
            num_grad_evals = num_grad_evals + batch_size

            # compute the gradients:
            loss_temp, x_grad = closure(x, Di, labels_i)
            x_tilde_grad = closure(x_tilde, Di, labels_i)[1]
            gk = x_grad - x_tilde_grad + full_grad
            Gk2 = Gk2 + (np.linalg.norm(gk) ** 2)

            if linesearch_option == 2:
                reset_step_size = step_size
                c = 0.5
                beta = 0.9
                step_size, armijo_iter = armijo_ls(closure, Di, labels_i, x, loss_temp, x_grad, reset_step_size, c, beta)
                num_grad_evals = num_grad_evals + batch_size * armijo_iter

            if adaptive_termination == True:
                if (i+1) == int(m/2.):
                    term1, term2 = compute_statistic(term1, term2, (i+1), x, gk, step_size)
                    if (np.abs(term1 - term2)) < 1e-8:
                        print('Test passed. Breaking out of inner loop')
                        break

            x -= (step_size / np.sqrt(Gk2)) * gk

    GRAD_EVALS = np.asarray(GRAD_EVALS)
    LOSS = np.asarray(LOSS)
    GRAD_NORM = np.asarray(GRAD_NORM)

    return x, GRAD_EVALS, LOSS, GRAD_NORM


# def svrg_ada(closure, batch_size, D, labels, init_step_size, n, d, max_epoch=100, m=0, x0=None, verbose=True,
#              adaptive_termination=False):
#     """
#         SVRG with fixed step size for solving finite-sum problems
#         Closure: a PyTorch-style closure returning the objective value and it's gradient.
#         batch_size: the size of minibatches to use.
#         D: the set of input vectors (usually X).
#         labels: the labels corresponding to the inputs D.
#         init_step_size: step-size to use
#         n, d: size of the problem
#     """
#     if not isinstance(m, int) or m <= 0:
#         m = n
#         if verbose:
#             print('Info: set m=n by default')
#
#     if x0 is None:
#         x = np.zeros(d)
#     elif isinstance(x0, np.ndarray) and x0.shape == (d,):
#         x = x0.copy()
#     else:
#         raise ValueError('x0 must be a numpy array of size (d, )')
#
#     LOSS = []
#     GRAD_NORM = []
#     GRAD_EVALS = []
#
#     snapshot_num = 0
#     num_grad_evals = 0
#
#     save_granularity = 2 * n
#
#     k = -1
#
#     budget = 2 * n * max_epoch # total budget on gradient evaluations.
#
#     while(1):
#
#         k = k + 1
#
#         if num_grad_evals >= budget:
#             print('End of grad evals budget. Exit')
#             break
#
#         loss, full_grad = closure(x, D, labels)
#         x_tilde = x.copy()
#
#         last_full_grad = full_grad
#         last_x_tilde = x_tilde
#
#         # initialize running sum of gradient norms
#         Gk2 = 0
#         step_size = init_step_size
#
#         if verbose:
#             output = 'Epoch.: %d, Grad. norm: %.2e' % \
#                      (k, np.linalg.norm(full_grad))
#             output += ', Func. value: %e' % loss
#             output += ', Num gradient evaluations: %d' % num_grad_evals
#             print(output)
#
#         if np.linalg.norm(full_grad) <= 1e-14:
#             return x, LOSS, GRAD_NORM
#
#         num_grad_evals = num_grad_evals + n
#
#         if num_grad_evals > save_granularity * snapshot_num:
#
#             if budget - num_grad_evals <= 0:
#                 break
#
#             snapshot_num = snapshot_num + 1
#             print('Budget remaining: ', budget - num_grad_evals)
#
#             GRAD_EVALS.append(num_grad_evals)
#             LOSS.append(loss)
#             GRAD_NORM.append(np.linalg.norm(full_grad))
#
#         # Create Minibatches:
#         minibatches = make_minibatches(n, m, batch_size)
#         for i in range(m):
#             # get the minibatch for this iteration
#             indices = minibatches[i]
#             Di, labels_i = D[indices, :], labels[indices]
#
#             # compute the gradients:
#             x_grad = closure(x, Di, labels_i)[1]
#             x_tilde_grad = closure(x_tilde, Di, labels_i)[1]
#
#             gk = x_grad - x_tilde_grad + full_grad
#
#             Gk2 = Gk2 + (np.linalg.norm(gk) ** 2)
#
#             x -= (step_size / np.sqrt(Gk2)) * (gk)
#
#             num_grad_evals = num_grad_evals + batch_size
#
#             if num_grad_evals > save_granularity * snapshot_num:
#
#                 if budget - num_grad_evals <= 0:
#                     break
#
#                 snapshot_num = snapshot_num + 1
#                 print('Budget remaining: ', budget - num_grad_evals)
#
#                 loss, full_grad = closure(x, D, labels)
#                 GRAD_EVALS.append(num_grad_evals)
#                 LOSS.append(loss)
#                 GRAD_NORM.append(np.linalg.norm(full_grad))
#
#             if num_grad_evals >= budget:
#                 print('End of grad evals budget. Exit')
#                 break
#
#             if adaptive_termination:
#
#                 if i == 0:
#                     org = (step_size / np.sqrt(Gk2))
#
#                 else:
#                     temp = (step_size / np.sqrt(Gk2))
#                     # print(temp / org)
#                     # check condition to terminate inner loop
#                     if (temp / org) < 1e-2:
#
#                         print('Breaking from inner loop')
#                         break
#
#     GRAD_EVALS = np.asarray(GRAD_EVALS)
#     LOSS = np.asarray(LOSS)
#     GRAD_NORM = np.asarray(GRAD_NORM)
#     return x, GRAD_EVALS, LOSS, GRAD_NORM