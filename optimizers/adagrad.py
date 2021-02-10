from dependencies import *

from utils import *
from datasets import *
from objectives import *
import time


def armijo_ls(closure, D, labels, x, loss, grad, p, init_step_size, c, beta=0.7, precond=1):

    
    temp_step_size = init_step_size
    armijo_iter = 1
    while closure(x - temp_step_size * precond * grad, D, labels,
                  backwards=False) > loss - c * temp_step_size * precond * np.dot(grad, p) :

        temp_step_size *= beta
        if armijo_iter == 100:
            temp_step_size = 1e-6
            break
        armijo_iter += 1

    step_size = temp_step_size

    return step_size, armijo_iter

def AdaGrad(score_list, closure, batch_size, D, labels,
            max_epoch=100, init_step_size=None, linesearch_option=0, 
            adaptive_termination=0, threshold_at=0.5,
            c=0.5, beta=0.7,
         x0=None, verbose=True,  D_test=None, labels_test=None):
    """
        Adagrad with fixed step size for solving finite-sum problems
        Closure: a PyTorch-style closure returning the objective value and it's gradient.
        batch_size: the size of minibatches to use.
        D: the set of input vectors (usually X).
        labels: the labels corresponding to the inputs D.
        init_step_size: step-size to use
        n, d: size of the problem
    """
    n = D.shape[0]
    d = D.shape[1]
 
    m = int(n/batch_size)

    if x0 is None:
        x = np.zeros(d)
        x0 = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d,):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    num_grad_evals = 0
    Gk2 = 0
    
    if linesearch_option in [0]:
        step_size = init_step_size
    if linesearch_option in [1]:
        step_size = init_step_size / 2**(batch_size/n) 
     
    condition_checked = False
    
    if adaptive_termination == 1:
        Gk2_list = np.zeros(max_epoch * m)
        iteration_counter = 0
        warmup_time = int(n / (batch_size))
        ratio_max = 0
        print(threshold_at, warmup_time)

            

    for k in range(max_epoch):      
        # if num_grad_evals >= 2 * n * max_epoch:
        #     # exceeds the number of standard SVRG gradient evaluations (only for batch-size = 1)
        #     print('End of budget for gradient evaluations')
        #     break
        t_start = time.time()



        loss, full_grad = closure(x, D, labels)
        
        if verbose:
            output = 'Epoch.: %d, Grad. norm: %.2e' % \
                     (k, np.linalg.norm(full_grad))
            output += ', Func. value: %e' % loss
            output += ', Step size: %e' % step_size
            output += ', Num gradient evaluations/n: %f' % (num_grad_evals / n) 
            print(output) 

        

        score_dict = {"epoch": k}
        score_dict["optimizer"] = 0
        score_dict["n_grad_evals"] = num_grad_evals
        score_dict["n_grad_evals_normalized"] = num_grad_evals / n
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
        
        if np.linalg.norm(full_grad) <= 1e-10:
            break
        if np.linalg.norm(full_grad) >= 1e10:
            break
        if np.isnan(full_grad).any():
            break
                   
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
            
            Gk2 = Gk2 + (np.linalg.norm(gk) ** 2)
            
            if linesearch_option == 0:
                step_size = init_step_size

              
            elif linesearch_option == 1:
                step_size, armijo_iter = armijo_ls(closure, Di, labels_i, x, loss,
                                                   x_grad, x_grad, 2**(batch_size/n) * step_size, c=c, beta=beta,
                                                  precond = 1)
                num_grad_evals += batch_size * armijo_iter
                
                if "armijo_iter" in score_list[len(score_list) - 1].keys(): 
                    score_list[len(score_list) - 1]["armijo_iter"] += armijo_iter
                else:
                    score_list[len(score_list) - 1]["armijo_iter"] = armijo_iter
                
            if adaptive_termination == 1:  
                if iteration_counter >=  warmup_time:
                    Gk2_list[iteration_counter] = Gk2
                    if iteration_counter % 2 == 0:                      
                        if iteration_counter/2 >= warmup_time:
                            Gk2_last = Gk2_list[int(iteration_counter/2)]                           
                            ratio = (Gk2 - Gk2_last) / Gk2_last
                            #print(ratio)
                            if ratio > ratio_max:
                                ratio_max = ratio
                            if ratio > threshold_at:
                                x -= (step_size / np.sqrt(Gk2)) * gk
                                print('Breaking out of inner loop at iteration', iteration_counter)
                                condition_checked = True
                                break
                iteration_counter += 1
                
                        
            
            x -= (step_size / np.sqrt(Gk2)) * gk
            
            

        
        
        t_end = time.time()
        time_epoch = t_end - t_start
        score_list[len(score_list) - 1]["time"] = time_epoch   
        if condition_checked:
            break
    
    return score_list, x, num_grad_evals, k