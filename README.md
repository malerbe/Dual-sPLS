# Dual-sPLS: Full theory and reimplementation from R to Python

> A pedagogical deep dive into Partial Least Squares (PLS) and its sparse variants, leading to a Python implementation of the Dual-SPLS algorithm.

## Learning Path


To fully grasp the mechanics behind the algorithms, it is recommended to follow the notebooks in this specific order:

### 1. Fundamentals: `docs/PLS.ipynb`
*   **Concept:** Standard Partial Least Squares (Vanilla PLS).
*   **Theory:** Covariance maximization, the NIPALS algorithm, and latent structures.
*   **Goal:** Understand how to project data into a lower-dimensional space without sparsity.

### 2. Introducing Sparsity: `docs/sPLS.ipynb`
*   **Concept:** Sparse PLS (sPLS).
*   **Theory:** Introduction of $L_1$ penalties and Soft-Thresholding to perform variable selection.
*   **Goal:** Learn how to select relevant variables while performing dimension reduction.

### 3. The Dual Approach: `docs/Dual_sPLS.ipynb`
*   **Concept:** Dual-SPLS (based on the reference paper).
*   **Theory:** Using dual norms to solve the optimization problem and the generalized NIPALS algorithm.
*   **Goal:** Prototype the logic of the Dual-SPLS algorithm step-by-step.

### 4. First production Implementation: `src/dual_spls/lasso.py`
*   Once the theory is mastered, the file `lasso.py` contains the robust, distinct implementation.
*   It organizes the mathematical prototype into a reusable function `dual_spls_lasso`, handling assertions, deflations, and prediction logic faithful to the original paper.

