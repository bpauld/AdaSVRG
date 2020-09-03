from dependencies import *

from optimizers.svrg import *
from optimizers.svrg_bb import *
from optimizers.svrg_ada import *

from utils import *

from datasets import *
from plotting import *
from objectives import *


def run_experiment(method_name,
                   closure,
                   X,
                   y,
                   X_test,
                   y_test,
                   batch_size=10,
                   max_epochs=50,
                   m=0,
                   num_restarts=10,
                   verbose=False,
                   seed=9513451,
                   **kwargs):
    '''Run an experiment with multiple restarts and compute basic statistics from the runs.'''
    # set the experiment seed
    print("Running Experiment:")
    np.random.seed(seed)
    loss_results = []
    gradnorm_results = []
    gradevals_results = []

    arg_dict = kwargs

    n, d = X.shape
    n_test = X_test.shape[0]
    x_sum = np.zeros(d)

    # do the restarts
    if method_name == "svrg":
        init_step_size = arg_dict["init_step_size"]
        adaptive_termination = arg_dict["adaptive_termination"]
        x, grad_evals_record, loss_record, gradnorm_record = svrg(closure=closure,
                                               batch_size=batch_size,
                                               D=X,
                                               labels=y,
                                               init_step_size=init_step_size,
                                               n=n,
                                               d=d,
                                               max_epoch=max_epochs,
                                               m=m,
                                               verbose=verbose,
                                               adaptive_termination = adaptive_termination)

    elif method_name == "svrg_bb":
        init_step_size = arg_dict["init_step_size"]
        adaptive_termination = arg_dict["adaptive_termination"]
        x, grad_evals_record, loss_record, gradnorm_record = svrg_bb(closure=closure,
                                                  batch_size=batch_size,
                                                  D=X,
                                                  labels=y,
                                                  init_step_size=init_step_size,
                                                  n=n,
                                                  d=d,
                                                  max_epoch=max_epochs,
                                                  m=m,
                                                  verbose=verbose,
                                                  adaptive_termination=adaptive_termination)

    elif method_name == "svrg_ada":
        init_step_size = arg_dict["init_step_size"]
        max_sgd_warmup_epochs = arg_dict["max_sgd_warmup_epochs"]
        linesearch_option = arg_dict["linesearch_option"]
        adaptive_termination = arg_dict["adaptive_termination"]

        x, grad_evals_record, loss_record, gradnorm_record = svrg_ada(closure=closure,
                                                   batch_size=batch_size,
                                                   D=X,
                                                   labels=y,
                                                   init_step_size=init_step_size,
                                                   n=n,
                                                   d=d,
                                                   max_epoch=max_epochs,
                                                   m=m,
                                                   verbose=verbose,
                                                   linesearch_option = linesearch_option,
                                                   max_sgd_warmup_epochs = max_sgd_warmup_epochs,
                                                   adaptive_termination = adaptive_termination)
    else:
        print('Method does not exist')
    for i in range(num_restarts):

        x_sum += x
        gradevals_results.append(grad_evals_record)
        loss_results.append(loss_record)
        gradnorm_results.append(gradnorm_record)

        if verbose:
            y_predict = np.sign(np.dot(X_test, x))
            print('Restart %d, Test accuracy: %f' % (i, (np.count_nonzero(y_test == y_predict) * 1.0 / n_test)))

    # compute basic statistics from the runs
    x_mean = x_sum / num_restarts

    loss_results = np.stack(loss_results)
    loss_std = loss_results.std(axis=0)
    loss_mean = loss_results.mean(axis=0)

    gradnorm_results = np.stack(gradnorm_results)
    gradnorm_std = gradnorm_results.std(axis=0)
    gradnorm_mean = gradnorm_results.mean(axis=0)

    gradevals_results = np.stack(gradevals_results)
    gradevals_std = gradevals_results.std(axis=0)
    gradevals_mean = gradevals_results.mean(axis=0)

    return x_mean, gradevals_mean, loss_mean, loss_std, gradnorm_mean, gradnorm_std