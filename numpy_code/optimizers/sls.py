from dependencies import *

from utils import *
from datasets import *
from objectives import *
import time

def armijo_ls(closure, D, labels, x, loss, grad, p, init_step_size, c, beta):

    temp_step_size = init_step_size
    armijo_iter = 1
    while closure(x - temp_step_size * grad, D, labels,
                  backwards=False) > loss - c * temp_step_size * np.dot(grad, p) :

        temp_step_size *= beta
        if armijo_iter == 50:
            temp_step_size = 1e-6
            break
        armijo_iter += 1

    step_size = temp_step_size

    return step_size, armijo_iter

def sls(score_list, closure, batch_size, D, labels, max_epoch=100, init_step_size =  None,
         x0=None, verbose=True, adaptive_termination = 0,  D_test=None, labels_test=None):
    """
        SVRG with fixed step size for solving finite-sum problems
        Closure: a PyTorch-style closure returning the objective value and it's gradient.
        batch_size: the size of minibatches to use.
        D: the set of input vectors (usually X).
        labels: the labels corresponding to the inputs D.
        init_step_size: step-size to use
        n, d: size of the problem
    """
    n = D.shape[0]
    d = D.shape[1]
 
    m = n
    
    if init_step_size is None:
        # hardcoding for now. 
        step_size = 1
        c = 0.5
        beta = 0.9        
    else:
        step_size = init_step_size

    if x0 is None:
        x = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d,):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    num_grad_evals = 0

    #check adaptive termination every "interval_size" iterations
    interval_size = 10 #hardcoding this for now
    if adaptive_termination == 2:
        # hardcoding these parameters for now.
        threshold  = 0.6
        num_checkpoints = int((m - (min(n,m)/2))/interval_size)
        start = int(min(n,m)/2)
        q = 1.5
            
        check_iter_indices = list(map(int,  list(np.linspace(start, m, num_checkpoints))))
        save_iter_indices = list(map(int, np.linspace(start, m, num_checkpoints) / q))        
                    
    for k in range(max_epoch):

        if num_grad_evals >= 2 * n * max_epoch:
            # exceeds the number of standard SVRG gradient evaluations (only for batch-size = 1)
            print('End of budget for gradient evaluations')
            break
        t_start = time.time()

        if adaptive_termination == 2:
            save_dist = np.zeros((num_checkpoints)) 
            checkpoint_num = 0

        loss, full_grad = closure(x, D, labels)
        
        term1 = 0
        term2 = 0

        if verbose:
            output = 'Epoch.: %d, Grad. norm: %.2e' % \
                     (k, np.linalg.norm(full_grad))
            output += ', Func. value: %e' % loss
            output += ', Step size: %e' % step_size
            output += ', Num gradient evaluations: %d' % num_grad_evals
            print(output) 

        if np.linalg.norm(full_grad) <= 1e-12:
            break

        score_dict = {"epoch": k}
        score_dict["n_grad_evals"] = num_grad_evals
        score_dict["train_loss"] = loss
        score_dict["grad_norm"] = np.linalg.norm(full_grad)
        score_dict["train_accuracy"] = accuracy(x, D, labels)
        score_dict["train_loss_log"] = np.log(loss)
        score_dict["grad_norm_log"] = np.log(score_dict["grad_norm"])
        # score_dict["train_accuracy_log"] = np.log(score_dict["train_accuracy"])
        if D_test is not None:
            test_loss = closure(x, D_test, labels_test, backwards=False)
            score_dict["test_loss"] = test_loss
            score_dict["test_accuracy"] = accuracy(x, D_test, labels_test)
            score_dict["test_loss_log"] = np.log(test_loss)
            # score_dict["test_accuracy_log"] = np.log(score_dict["test_accuracy"])

        score_list += [score_dict]
            
        # Create Minibatches:
        minibatches = make_minibatches(n, m, batch_size)
        for i in range(m):
            # get the minibatch for this iteration
            indices = minibatches[i]
            Di, labels_i = D[indices, :], labels[indices]

            # compute the loss, gradients
            loss, x_grad = closure(x, Di, labels_i)        
            gk  = x_grad
            num_grad_evals = num_grad_evals + batch_size

            if adaptive_termination == 1:
                if (i+1) >= int(min(n,m)/2): # evaluate the statistic halfway through the inner loop
                    term1, term2 = compute_pflug_statistic(term1, term2, (i+1), x, gk, step_size)
                    if (i+1) % interval_size == 0:
                        if (np.abs(term1 - term2)) < 1e-10:
                            print('Test passed. Breaking out of inner loop at iteration ', (i+1))                            
                            break                                       
                        
            if adaptive_termination == 2:
                if (i+1) in (save_iter_indices):                    
                    save_dist[checkpoint_num] = np.linalg.norm(x - x0)
                    checkpoint_num += 1
            
                elif ((i+1) in check_iter_indices):
               
                    t1 = i+1
                                
                    ind = check_iter_indices.index(i+1)                    
                    x_prev_dist = save_dist[ind]
                    x_dist = np.linalg.norm(x - x0)
                    t2 = save_iter_indices[ind]
                    S = (np.log(x_dist**2) - np.log(x_prev_dist**2)) / (np.log(t1) - np.log(t2))
                    # print('S = ', S)
                    if S < threshold:
                        x -= step_size * gk
                        print('Test passed. Breaking out of inner loop at iteration ', (i+1))
                        break
            
            if init_step_size is None:
                reset_step_size = step_size 
                step_size, armijo_iter = armijo_ls(closure, Di, labels_i, x, loss, 
                                            x_grad, x_grad, reset_step_size, c, beta)            

                num_grad_evals = num_grad_evals + (armijo_iter - 1)  * batch_size       

            x -= step_size * gk

        t_end = time.time()
        time_epoch = t_end - t_start
        score_list[len(score_list) - 1]["time"] = time_epoch    

    return score_list