from dependencies import *

LIBSVM_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
LIBSVM_DOWNLOAD_FN = {"rcv1": "rcv1_train.binary.bz2",
                      "mushrooms": "mushrooms",
                      "a1a": "a1a",
                      "a2a": "a2a",
                      "ijcnn": "ijcnn1.tr.bz2",
                      "w8a": "w8a"}


def load_libsvm(name, data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    fn = LIBSVM_DOWNLOAD_FN[name]
    data_path = os.path.join(data_dir, fn)

    if not os.path.exists(data_path):
        url = urllib.parse.urljoin(LIBSVM_URL, fn)
        print("Downloading from %s" % url)
        urllib.request.urlretrieve(url, data_path)
        print("Download complete.")

    X, y = load_svmlight_file(data_path)
    return [X, y]


def data_load(data_dir, dataset_name, n=0, d=0, margin=1e-6, false_ratio=0, is_subsample=0, is_kernelize=0,
              test_prop=0.2, split_seed=9513451):
    
    if (dataset_name != 'synthetic'):

        # real data
        #         data = pickle.load(open(data_dir + data_name +'.pkl', 'rb'), encoding = "latin1")
        data = load_libsvm(dataset_name, data_dir='./')

        # load real dataset
        A = data[0].toarray()

        if dataset_name in ['quantum', 'rcv1', 'protein', 'news']:
            y = data[1].toarray().ravel()
        else:
            y = data[1]

    else:

        A, y, w_true = create_dataset(n, d, margin, false_ratio)

    # subsample
    if is_subsample == 1:
        A = A[:n, :]
        y = y[:n]

    # split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(A, y, test_size=test_prop, random_state=split_seed)

    if is_kernelize == 1:
        # Form kernel
        A_train, A_test = kernelize(X_train, X_test, dataset_name, data_dir=data_dir)
    else:
        A_train = X_train
        A_test = X_test

    print('Loaded ', dataset_name, ' dataset.')

    return A_train, y_train, A_test, y_test


def kernelize(X, X_test, dataset_name, kernel_type=0, data_dir="./Data"):
    n = X.shape[0]

    fname = data_dir + '/Kernel_' + str(n) + '_' + str(dataset_name) + '.p'

    if os.path.isfile(fname):

        print('Reading file ', fname)
        X_kernel, X_test_kernel = pickle.load(open(fname, "rb"))

    else:
        if kernel_type == 0:
            X_kernel = RBF_kernel(X, X)
            X_test_kernel = RBF_kernel(X_test, X)
            print('Formed the kernel matrix')

        pickle.dump((X_kernel, X_test_kernel), open(fname, "wb"))

    return X_kernel, X_test_kernel


def RBF_kernel(A, B, sigma=1.0):
    distance_2 = np.square(metrics.pairwise.pairwise_distances(X=A, Y=B, metric='euclidean'))
    K = np.exp(-1 * np.divide(distance_2, (2 * (sigma ** 2))))

    return K


def create_dataset(n, d, gamma=0, false_ratio=0):
    # create synthetic dataset using the python utility
    # X, y = datasets.make_classification(n_samples=n, n_features=d,n_informative = d, n_redundant = 0, class_sep = 2.0 )
    # convert into -1/+1
    # y = 2 * y - 1

    # create linearly separable dataset with margin gamma
    # w_star = np.random.random((d,1))
    w_star = np.random.normal(0, 1, (d, 1))
    # normalize w_star
    w_star = w_star / np.linalg.norm(w_star)

    num_positive = 0
    num_negative = 0
    count = 0

    X = np.zeros((n, d))
    y = np.zeros((n))

    while (1):

        x = np.random.normal(1, 1, (1, d))
        # normalize x s.t. || x ||_2 = 1
        #         x = x / np.linalg.norm(x)

        temp = np.dot(x, w_star)
        margin = abs(temp)
        sig = np.sign(temp)

        if margin > gamma * np.linalg.norm(w_star):

            if count % 2 == 0:

                # generate positive
                if sig > 0:
                    X[count, :] = x
                else:
                    X[count, :] = -x
                y[count] = + 1

            else:

                # generate negative
                if sig < 0:
                    X[count, :] = x
                else:
                    X[count, :] = -x
                y[count] = - 1

            count = count + 1

        if count == n:
            break

    flip_ind = np.random.choice(n, int(n * false_ratio))
    y[flip_ind] = -y[flip_ind]

    return X, y, w_star
