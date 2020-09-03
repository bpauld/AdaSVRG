from dependencies import *

def plotting(results, labels, max_epochs):
    plt.figure()

    offset = 0
    colors = ['r', 'b', 'g', 'k', 'cyan', 'lightgreen']

    x = range(max_epochs)

    for i in range(len(labels)):
        plt.plot(x, (results[i, :]), color=colors[i], label=labels[i])

    plt.xlabel('Number of Effective Passes')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()


def plot_shaded_error_bars(grad_evals, results_mean, results_std, max_epochs, ylabel="Loss", colors=None, labels=None):
    fig = plt.figure()
    offset = 0
    if colors is None:
        colors = ['r', 'b', 'g', 'k', 'cyan', 'lightgreen']
    if labels is None:
        labels = ['SVRG-BB', 'SVRG']

    for i in range(results_mean.shape[0]):
        x = grad_evals[i, :]
        if results_mean[i, :].sum() != 0:
            plt.plot(x, np.log10(results_mean[i, :]), color=colors[i], label=labels[i])
            plt.fill_between(x, np.log10(results_mean[i, :] - results_std[i, :]), np.log10(results_mean[i, :] + results_std[i, :]),
                         color=colors[i], alpha=0.5)

    plt.xlabel('Number of gradient evaluations')
    plt.ylabel(ylabel)
    plt.legend(loc='best')

    return fig