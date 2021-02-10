from dependencies import *
from utils import *
from datasets import *
from objectives import *
import time

def AdaSVRG(score_list, closure, batch_size, D, labels, 
            init_step_size = None, max_epoch=100, r=0, x0=None, verbose=True,
            linesearch_option = 1,
            adaptive_termination = 0,threshold_at=0.5,
            D_test=None, labels_test=None):
    """
        AdaSVRG for solving finite-sum problems
        Closure: a PyTorch-style closure returning the objective value and it's gradient.
        batch_size: the size of minibatches to use.
        D: the set of input vectors (usually X).
        labels: the labels corresponding to the inputs D.
        init_step_size: step-size to use
        n, d: size of the problem
    """
    n = D.shape[0]
    d = D.shape[1]
     
        
    if init_step_size is None:
        linesearch_option = 1
    if r <= 0:
        m = n
        if verbose:
            print('Info: set m=n by default')   
    else:
        m = int(r * n)
        print('Info: set m = ', r, ' n')

    if x0 is None:
        x = np.zeros(d)        
    elif isinstance(x0, np.ndarray) and x0.shape == (d,):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')
    

    num_grad_evals = 0    
    step_size = init_step_size
    for k in range(max_epoch):
        t_start = time.time()

        loss, full_grad = closure(x, D, labels)
        x_tilde = x.copy()
        num_grad_evals = num_grad_evals + n

        # initialize running sum of gradient norms
        Gk2 = 0
        
           
        if linesearch_option == 0:
            step_size = init_step_size                                      

        elif linesearch_option == 1:
            if k == 0:
                x_rand = x + np.random.normal(0, 0.01, d)
                loss_rand , full_grad_rand = closure(x_rand, D, labels)
                num_grad_evals = num_grad_evals + n
                L_hat = np.linalg.norm(full_grad - full_grad_rand) / np.linalg.norm(x_tilde - x_rand)
                max_L_hat = L_hat

                # to prevent the step-size from blowing up. 
                if (max_L_hat < 1e-8):
                    step_size = 1e-4
                else:
                    step_size = np.linalg.norm(full_grad) / (L_hat * np.sqrt(2))
                       
            else:
                L_hat = np.linalg.norm(full_grad - last_full_grad) / np.linalg.norm(x_tilde - last_x_tilde)
                if L_hat > max_L_hat:
                    max_L_hat = L_hat

                # to prevent the step-size from blowing up. 
                print("max_L_hat = ", max_L_hat)
                if (max_L_hat < 1e-8):
                    step_size =  1e-4 # some small default step-size
                else:
                    step_size = np.linalg.norm(full_grad) / (max_L_hat * np.sqrt(2))
        else:
            print("Linesearch option " +  str(linesearch_option) + " not supported for AdaSVRG")
            return
                    
        print("m = ", m)
        
        last_full_grad = full_grad
        last_x_tilde = x_tilde
        
        score_dict = {"epoch": k}
        score_dict["optimizer"] = 1
        score_dict["m"] = m
        score_dict["n_grad_evals"] = num_grad_evals
        score_dict["L_hat"] = L_hat
        score_dict["step_size"] = step_size
        score_dict["n_grad_evals_normalized"] = num_grad_evals / n
        score_dict["train_loss"] = loss
        score_dict["grad_norm"] = np.linalg.norm(full_grad)
        score_dict["train_accuracy"] = accuracy(x, D, labels)    
        if D_test is not None:
            test_loss = closure(x, D_test, labels_test, backwards=False)
            score_dict["test_loss"] = test_loss
            score_dict["test_accuracy"] = accuracy(x, D_test, labels_test)

        score_list += [score_dict]
        if verbose:
            output = 'Epoch.: %d, Grad. norm: %.2e' % \
                     (k, np.linalg.norm(full_grad))
            output += ', Func. value: %e' % loss
            output += ', Step size: %e' % step_size
            output += ', Num gradient evaluations/n: %f' % (num_grad_evals / n) 
            output += ', L_hat: %e' % L_hat
            print(output)        
        
        full_grad_norm = np.linalg.norm(full_grad)
        if full_grad_norm <= 1e-12:
            return score_list
        elif full_grad_norm >= 1e10:
            return score_list
        elif np.isnan(full_grad_norm):
            return score_list

                  
            
        if adaptive_termination == 1:
            Gk2_list = np.zeros(m)
            iteration_counter = 0
            warmup_time = int(n / (4*batch_size))
            ratio_max = 0
            print(threshold_at, warmup_time)
        

        # Create Minibatches:
        minibatches = make_minibatches(n, m, batch_size)
        for i in range(m):
            # get the minibatch for this iteration
            indices = minibatches[i]
            Di, labels_i = D[indices, :], labels[indices]
            num_grad_evals = num_grad_evals + 2 * batch_size

            # compute the gradients:
            loss_temp, x_grad = closure(x, Di, labels_i)
            x_tilde_grad = closure(x_tilde, Di, labels_i)[1]
            gk = x_grad - x_tilde_grad + full_grad
            Gk2 = Gk2 + (np.linalg.norm(gk) ** 2)
                        
            if adaptive_termination == 1:  
                if iteration_counter >=  warmup_time:
                    Gk2_list[iteration_counter] = Gk2
                    if iteration_counter % 2 == 0:                      
                        if iteration_counter/2 >= warmup_time:
                            Gk2_last = Gk2_list[int(iteration_counter/2)]                           
                            ratio = (Gk2 - Gk2_last) / Gk2_last
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

    return score_list
            
