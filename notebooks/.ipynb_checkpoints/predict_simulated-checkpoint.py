#!/usr/bin/env python
# coding: utf-8

# # Dummy simulation
# ---

# In[1]:


from dual_spls.simulate import simulate
import numpy as np


# In[2]:


## Configuration:
# Simulation configuration:
coefs = [0, 1, 0, 1, 1, 0, 0, 1, 1] 
n = 300
p = [1000]
nondes = [30]
sigmaondes = [0.40]
sigma_y = 0.05

# Split configuration:
split_mode = "random"

# Dual-sPLS configuration
norm = "pseudo-lasso" # options: "pseudo-lasso", ...


# In[3]:


# generate synthetic data
simulation_results = simulate(n=n, p=p, nondes=nondes , sigmaondes=sigmaondes, sigma_y=sigma_y, coefs=coefs)
X, y = simulation_results["X"], simulation_results["y"]


# In[4]:


# split data
if split_mode == "random":
    # split data randomly
    from sklearn.model_selection import train_test_split

    print("Splitting data using a classic random split.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


# standardize data (after splitting to avoid data leakage)
from sklearn.preprocessing import StandardScaler

X_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)       

y_scaler = StandardScaler()
y_train_scaled= y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()


# In[6]:


# apply Dual-sPLS
from dual_spls.lasso import dual_spls_lasso

results = {}
if norm == "pseudo-lasso":
    print("Applying Dual-sPLS with the \"pseudo-lasso\" norm.")

    for n_comp in range(1, 10):
        result = dual_spls_lasso(X_train_scaled, y_train_scaled, n_components=n_comp, ppnu=0.99)
        beta = result['Bhat'][:, -1]
        intercept = result['intercept'][-1]

        y_pred_scaled = (X_test_scaled @ beta + intercept).flatten()

        rmse = np.sqrt(np.mean((y_test_scaled - y_pred_scaled)**2))
        results[f"rmse_{n_comp}_comp"] = rmse

results


# In[7]:


# apply PLS
from sklearn.cross_decomposition import PLSRegression

results_PLS = {}
for n_comp in range(1, 10):
        PLS = PLSRegression(n_components=n_comp, scale=False, copy=True)
        PLS.fit(X_train_scaled, y_train_scaled)

        y_pred_scaled = PLS.predict(X_test_scaled).flatten()

        rmse = np.sqrt(np.mean((y_test_scaled - y_pred_scaled)**2))
        results_PLS[f"rmse_{n_comp}_comp"] = rmse

results_PLS


# In[8]:


import matplotlib.pyplot as plt

plt.plot(range(1, 10), list(results.values()), color='red',label="$Dual-sPLS_l$")
plt.plot(range(1, 10), list(results.values()), ".", color='red')

plt.plot(range(1, 10), list(results_PLS.values()), color='black',label="$PLS$")
plt.plot(range(1, 10), list(results_PLS.values()), ".", color='black')

plt.xlabel("Number of latent component")
plt.ylabel("RMSE (validation)")
plt.grid()
plt.legend()
plt.show()


# # Paper's data
# ---
# Here, we takes $X$ and $Y$ matrices of Abdi's paper introuced in "A small example" section.

# In[9]:


# Library
import pandas as pd


# In[10]:


data = {
    "X": {
        "columns": ["Price", "Sugar", "Alcohol", "Acidity"],
        "values": [
            [7, 7, 13, 7],
            [4, 3, 14, 7],
            [10, 5, 12, 5],
            [16, 7, 11, 3],
            [13, 3, 10, 3]
        ]
    },
    "Y": {
        "columns": ["Hedonic", "Goes_with_meat", "Goes_with_dessert"],
        "values": [
            [14, 7, 8],
            [10, 7, 6],
            [8, 5, 5],
            [2, 4, 7],
            [6, 2, 4]
        ]
    },
    "source": "Abdi, H. (2003). Partial Least Squares (PLS) Regression."
}

X = pd.DataFrame(
    data["X"]["values"],
    columns=data["X"]["columns"],
)

y = pd.DataFrame(
    data["Y"]["values"],
    columns=data["Y"]["columns"],
)


# In[11]:


X


# In[12]:


y


# In[13]:


# split data
if split_mode == "random":
    # split data randomly
    from sklearn.model_selection import train_test_split

    print("Splitting data using a classic random split.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


# standardize data 
from sklearn.preprocessing import StandardScaler

X_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)       

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values)
y_test_scaled = y_scaler.transform(y_test.values)


# In[15]:


# apply Dual-sPLS
from dual_spls.lasso import dual_spls_lasso

results_dual_spls = {}
if norm == "pseudo-lasso":
    print('Applying Dual-sPLS with the "pseudo-lasso" norm.')

    n_samples, n_targets = y_train_scaled.shape if y_train_scaled.ndim > 1 else (y_train_scaled.shape[0], 1)

    for n_comp in range(1, 10):
        results_per_target = {}

        for j in range(n_targets):
            y_col = y_train_scaled[:, j].flatten() if n_targets > 1 else y_train_scaled
            result = dual_spls_lasso(X_train_scaled, y_col, n_components=n_comp, ppnu=0.99)
            beta = result['Bhat'][:, -1]
            intercept = result['intercept'][-1]

            y_pred_scaled = (X_test_scaled @ beta + intercept).flatten()
            rmse = np.sqrt(np.mean((y_test_scaled[:, j] - y_pred_scaled)**2)) if n_targets > 1 else np.sqrt(np.mean((y_test_scaled - y_pred_scaled)**2))

            results_per_target[f"rmse_target{j}"] = rmse

        results_dual_spls[f"rmse_{n_comp}_comp"] = results_per_target

results_dual_spls


# In[16]:


# apply PLS
from sklearn.cross_decomposition import PLSRegression

results_pls = {}
max_comp = min(X_train_scaled.shape[0], X_train_scaled.shape[1], y_train_scaled.shape[1])

for n_comp in range(1, max_comp + 1):
    pls = PLSRegression(n_components=n_comp, scale=False, copy=True)
    pls.fit(X_train_scaled, y_train_scaled)
    y_pred_scaled = pls.predict(X_test_scaled)

    results_per_target = {}
    for j in range(y_train_scaled.shape[1]):
        rmse = np.sqrt(np.mean((y_test_scaled[:, j] - y_pred_scaled[:, j])**2))
        results_per_target[f"target{j}"] = rmse

    results_pls[f"{n_comp}_comp"] = results_per_target

results_pls


# In[17]:


import matplotlib.pyplot as plt

colors = ["red", "blue", "green"]

for j in range(n_targets):
    # Dual-sPLS
    dual_vals = [results_dual_spls[f"rmse_{n}_comp"][f"rmse_target{j}"] for n in range(1, max_comp + 1)]
    plt.plot(range(1, max_comp + 1), dual_vals, color=colors[j], marker='o', linestyle='-', label=f"Dual-sPLS Target {j+1}")

    # PLS
    pls_vals = [results_pls[f"{n}_comp"][f"target{j}"] for n in range(1, max_comp + 1)]
    plt.plot(range(1, max_comp + 1), pls_vals, color=colors[j], marker='x', linestyle='--', label=f"PLS Target {j+1}")

plt.xlabel("Number of latent components")
plt.ylabel("RMSE (validation)")
plt.grid()
plt.legend()
plt.show()


# In[18]:


pls_vals


# In[19]:


dual_vals


# In[ ]:


get_ipython().system('jupyter nbconvert --to script --output-dir _Prod/report/annexes pred.ipynb')

