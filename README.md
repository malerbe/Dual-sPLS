# Dual-sPLS: Full theory and reimplementation from R to Python

The goal of this repository is to reimplement the paper [] from R to Python. It can be used both as a learning tool to understand the theory behind the algorithm and as a standalone installable library. 

## Usage

To start using the repository, you can clone it:

```
git clone https://github.com/malerbe/Dual-sPLS.git
```

and then install it using pip:

```
cd ./Dual-sPLS
pip install -e . 
```

It is suggested to use the notebook `notebooks/predict_simulated.ipynb` as a "documentation" to understand how to use different features implemented in the library. Reading the docstrings and the commentaries in the code will allow a better understanding of what the arguments correspond to. 

The library also allows the user to generate synthetic data as presented in the paper. To see how to use the generation function, see: `notebooks/simulate.ipynb`

## Learning Path

If your goal is to fully grasp the mechanics behind the algorithms, it is recommended to follow the explanation notebooks in this specific order:

- Fundamentals: `docs/PLS.ipynb`

- Introducing Sparsity: `docs/sPLS.ipynb`

- The Dual Approach: `docs/Dual_sPLS.ipynb`

It is then possible to fully understand the first production implementation `src/dual_spls/lasso.py` easily as it only uses code already explained and implemented the last `docs/Dual_sPLS.ipynb` notebook. 

## Sources and original repository

- Paper: [Dual-sPLS: a family of Dual Sparse Partial Least Squares regressions for feature selection and prediction with tunable sparsity; evaluation on simulated and near-infrared (NIR) data](https://arxiv.org/abs/2301.07206)
- Original repository: [dual.spls](https://github.com/AlsoukiL/dual.spls)
