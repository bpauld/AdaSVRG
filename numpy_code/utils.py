from dependencies import *


def make_minibatches(n, m, minibatch_size):
    ''' Create m minibatches from the training set by sampling without replacement.
        This function may sample the training set multiple times.
    Parameters:
        n: the number of examples in the dataset
        m: number of minibatches to generate
        batch_size: size of the desired minibatches'''

    k = math.ceil(m * minibatch_size / n)
    batches = []
    for i in range(k):
        batches += minibatch_data(n, minibatch_size)

    return batches


def minibatch_data(n, batch_size):
    '''Splits training set into minibatches by sampling **without** replacement.
    This isn't performant for large datasets (e.g. we should switch to PyTorch's streaming data loader eventually).
    Parameters:
        n: the number of examples in the dataset
        batch_size: size of the desired minibatches'''
    # shuffle training set indices before forming minibatches
    indices = np.arange(n)
    np.random.shuffle(indices)

    batches = []
    num_batches = math.floor(n / batch_size)
    # split the training set into minibatches
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        stop_index = (batch_num + 1) * batch_size

        # create a minibatch
        batches.append(indices[start_index:stop_index])

    # generate a final, smaller batch if the batch_size doesn't divide evenly into n
    if num_batches != math.ceil(n / batch_size):
        batches.append(indices[stop_index:])

    return batches

def reset(model):
    # reset the model
    for param in model.parameters():
        param.data = torch.zeros_like(param)
    loss_results = []
    gradnorm_results = []

    return model