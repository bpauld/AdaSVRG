# SVRG meets AdaGrad: Painless Variance Reduction

## Experiments

Run the experiments using the command below:

``
python trainval.py -e $exp_{BENCHMARK} -sb ${SAVEDIR_BASE} -r 1
``

with the placeholders defined as follows.

**{BENCHMARK}**

Defines the dataset and regularization constant for the experiments

- `a1a`, `a2a`, `w8a`, `mushrooms`, `ijcnn`, `phishing`, `rcv1` for the experiments comparing AdaSVRG to classical methods (including SVRG).


- `synthetic_interpolation` for the interpolation experiments.

**{SAVEDIR_BASE}**

Defines the absolute path to where the results will be saved.
