from haven import haven_utils as hu
import itertools
RUNS = [0]

def get_benchmark(benchmark, opt_list):
    if benchmark == 'syn':
        return {"dataset": ["synthetic"],
                "model": ["logistic"],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "acc_func": ["softmax_accuracy"],
                'margin':
                [
                    0.05,
            		0.1,
                    0.5,
                    0.01,
        ],
            "n_samples": [1000],
            "d": 20,
            "batch_size": [100],
            "max_epoch": [200],
            "runs": RUNS}

    elif benchmark == 'kernels':
        return {"dataset": ["mushrooms", "ijcnn", "rcv1"],
                "model": ["logistic"],
                "loss_func": ['softmax_loss'],
                "acc_func": ["softmax_accuracy"],
                "opt": opt_list,
                "batch_size": [100],
                "max_epoch": [100],
                "runs": RUNS}

    elif benchmark == 'mf':
        return {"dataset": ["matrix_fac"],
                "model": ["matrix_fac_1", "matrix_fac_4", "matrix_fac_10", "linear_fac"],
                "loss_func": ["squared_loss"],
                "opt": opt_list,
                "acc_func": ["mse"],
                "batch_size": [100],
                "max_epoch": [100],
                "runs": RUNS}

    elif benchmark == 'mnist':
        return {"dataset": ["mnist"],
                "model": ["mlp"],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "acc_func": ["softmax_accuracy"],
                "batch_size": [128],
                "max_epoch": [100],
                "runs": RUNS}

    elif benchmark == 'cifar10':
        return {"dataset": ["cifar10"],
                "model": [
            "densenet121",

            "resnet34"
        ],
            "loss_func": ["softmax_loss"],
            "opt": opt_list,
            "acc_func": ["softmax_accuracy"],
            "batch_size": [128],
            "max_epoch": [200],
            "runs": RUNS}

    elif benchmark == 'cifar100':
        return {"dataset": ["cifar100"],
                "model": [
            "densenet121_100",
            "resnet34_100"
        ],
            "loss_func": ["softmax_loss"],
            "opt": opt_list,
            "acc_func": ["softmax_accuracy"],
            "batch_size": [128],
            "max_epoch": [200],
            "runs": RUNS}

    elif benchmark == 'cifar10_nobn':
        return {"dataset": ["cifar10"],
                "model": ["resnet34_nobn", "densenet121_nobn"],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "acc_func": ["softmax_accuracy"],
                "batch_size": [128],
                "max_epoch": [200],
                "runs": RUNS}

    elif benchmark == 'cifar100_nobn':
        return {"dataset": ["cifar100"],
                "model": ["resnet34_100_nobn", "densenet121_100_nobn"],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "acc_func": ["softmax_accuracy"],
                "batch_size": [128],
                "max_epoch": [200],
                "runs": RUNS}

    elif benchmark == 'imagenet200':
        return {"dataset": ["tiny_imagenet"],
                "model": ["resnet18"],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "acc_func": ["softmax_accuracy"],
                "batch_size": [128],
                "max_epoch": [200],
                "runs": RUNS}
    elif benchmark == 'imagenet10':
        return {"dataset": ["imagenette2-160", "imagewoof2-160"],
                "model": ["resnet18"],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "acc_func": ["softmax_accuracy"],
                "batch_size": [128],
                "max_epoch": [100],
                "runs": RUNS}


EXP_GROUPS = {}
# -------------- ## -------------- ## -------------- ## -------------- #
# Setting up optimizers
# ------------------ #
opt_list = {'name': 'svrg'}


# -------------- ## -------------- ## -------------- ## -------------- #
# Setting up benchmarks
# ------------------ #
# II. Convex with interpolation
benchmarks_list = ['syn', 'kernels']
# all optimizers for small exps
opt_list = opt_list
           
for benchmark in benchmarks_list:
    EXP_GROUPS['adaptive_II_%s' % benchmark] = hu.cartesian_exp_group(
        get_benchmark(benchmark, opt_list))

# ------------------ #
# III. Easy nonconvex
benchmarks_list = ['mnist', 'mf']
opt_list = opt_list

for benchmark in benchmarks_list:
    EXP_GROUPS['adaptive_III_%s' % benchmark] = hu.cartesian_exp_group(
        get_benchmark(benchmark, opt_list))

