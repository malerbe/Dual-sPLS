#!/usr/bin/env python
# coding: utf-8

# Data is comming from the archive http://www.laurent-duval.eu/opus-dual-spls-sparse-pls/Data-dual-sPLS-NIR-near-infrared-spectra-density/matrixXYNir-CSV-dual-sparse-pls.7z (see **Dataset of near-infrared (NIR) spectral data for prediction of organic matter and total carbon in agricultural soil using homemade NIR spectrometer**). Like the paper said, X matrix is tranformed using the Savitzky–Golay filter before use. Here we introduce Dual-sPLS performance using *RMSE* metric for different norms on real data (NIR dataset). Norms $\Omega$ includes :
# 
# * Lasso (*lasso.py*)
# * Ridge (*ridge.py*)
# * GLA (*GLA.py*)
# * GLB (*GLB.py*)
# * GLC (*GLC.py*)
# * $\mathcal{l}_2$ norm (*LS.py*)

# In[1]:


# Importing libraries
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import seaborn as sns 
from sklearn.model_selection import train_test_split

# Dual-sPLS library - norms
from dual_spls.lasso import dual_spls_lasso
from dual_spls.ridge import d_spls_ridge
from dual_spls.GLA   import dual_spls_gla_random
from dual_spls.GLB   import dual_spls_glb
from dual_spls.GLC   import dual_spls_glc
from dual_spls.LS    import d_spls_ls

sns.set_style("darkgrid")


# In[2]:


# Import dataset
X = pd.read_csv(
    "../data/matrixXNirSpectrumDerivative.csv"
)
X.head()


# In[3]:


y = pd.read_csv("../data/matrixYNirPropertyDensityNormalized.csv", header=None)
y.head()


# In[4]:


from sklearn.cluster import AgglomerativeClustering

# ----------- Data (adapted to the lib) ----------- #
X = pd.read_csv(
    "../data/matrixXNirSpectrumDerivative.csv",
     header=None
).T.values
y = pd.read_csv(
    "../data/matrixYNirPropertyDensityNormalized.csv",
    header=None
).values.ravel()

# ----------- Params ----------- #

n_comp = 6
labels = AgglomerativeClustering(n_clusters=25).fit(X.T).labels_ # for GL*

# GLA
indG = labels + 1

nG = indG.max()
ppnu = np.repeat(0.5, nG)

# GLB
nG = indG.max()
gamma = np.repeat(1.0 / nG, nG)

# ----------- Train ----------- #

lasso = dual_spls_lasso(X, y, n_components=n_comp, ppnu=0.8)
ridge = d_spls_ridge(X, y, n_components=n_comp, ppnu=0.75, nu2=0.1)
gla = dual_spls_gla_random(X, y, n_components=n_comp, ppnu=ppnu, indG=indG)
glb = dual_spls_glb(X, y, n_components=n_comp, ppnu=ppnu, indG=indG)
glc = dual_spls_glc(X, y, n_components=n_comp, ppnu=ppnu, indG=indG, gamma=gamma)
ls = d_spls_ls(X, y, n_components=n_comp, ppnu=0.8, verbose=True)


# In[5]:


import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Lasso
rmse_lasso = []
for k in range(lasso["fitted_values"].shape[1]):
    y_hat = lasso["fitted_values"][:, k]
    rmse_lasso.append(np.sqrt(mean_squared_error(y, y_hat)))

# Ridge
rmse_ridge = []
for k in range(ridge["fitted_values"].shape[1]):
    y_hat = ridge["fitted_values"][:, k]
    rmse_ridge.append(np.sqrt(mean_squared_error(y, y_hat)))

# GLA
rmse_gla = []
for k in range(gla["fitted_values"].shape[1]):
    y_hat = gla["fitted_values"][:, k]
    rmse_gla.append(np.sqrt(mean_squared_error(y, y_hat)))

# GLB
rmse_glb = []
for k in range(glb["fitted_values"].shape[1]):
    y_hat = glb["fitted_values"][:, k]
    rmse_glb.append(np.sqrt(mean_squared_error(y, y_hat)))

# GLC
rmse_glc = []
for k in range(glc["fitted_values"].shape[1]):
    y_hat = glc["fitted_values"][:, k]
    rmse_glc.append(np.sqrt(mean_squared_error(y, y_hat)))

# LS
rmse_ls = []
for k in range(ls["fitted_values"].shape[1]):
    y_hat = ls["fitted_values"][:, k]
    rmse_ls.append(np.sqrt(mean_squared_error(y, y_hat)))

plt.plot(range(1, len(rmse_lasso)+1), rmse_lasso, marker="o", color='red', label="$Dual-sPLS \\quad (Lasso)$")
plt.plot(range(1, len(rmse_ridge)+1), rmse_ridge, marker=".", color='black', label="$Dual-sPLS \\quad (Ridge)$")
plt.plot(range(1, len(rmse_gla)+1), rmse_gla, marker=".", color='blue', label="$Dual-sPLS \\quad (GLA)$")
plt.plot(range(1, len(rmse_glb)+1), rmse_glb, marker=".", color='green', label="$Dual-sPLS \\quad (GLB)$")
plt.plot(range(1, len(rmse_glc)+1), rmse_gla, marker=".", color='orange', label="$Dual-sPLS \\quad (GLC)$")
plt.plot(range(1, len(rmse_gla)+1), rmse_ls, marker=".", color='violet', label="$Dual-sPLS \\quad (\\mathcal{l}_2)$")


plt.xlabel("Latent component")
plt.ylabel("RMSE (validation)")
plt.title("RMSE for components values for different norms")
plt.legend()
plt.show()


# In[ ]:




