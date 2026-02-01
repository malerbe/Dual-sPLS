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
import matplotlib.pyplot as plt
import matplotlib.cm     as cm
import matplotlib.colors as colors


from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics         import mean_squared_error
from sklearn.cluster         import AgglomerativeClustering

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


# Initialization

def cv_rmse(model_fun, X, y, **kwargs):
    rmses = []

    for train_idx, test_idx in kf.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        try:
            model = model_fun(Xtr, ytr, n_components=n_comp, **kwargs)

            y_pred = (
                (Xte - model["Xmean"])
                @ model["Bhat"][:, -1]
                + model["intercept"][-1]
            )

            # For divergent models
            if not np.all(np.isfinite(y_pred)):
                return np.inf, None

            rmse = np.sqrt(mean_squared_error(yte, y_pred))
            rmses.append(rmse)

        except Exception:
            return np.inf, None

    return np.mean(rmses), model

# ----------- Init for grid search and K fold (K=5) ----------- #
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_score = np.inf
best_models = {}
best_params = {}

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

n_comp = 50
labels = AgglomerativeClustering(n_clusters=25).fit(X.T).labels_  # for GL*
indG = labels + 1  # 1-based groups vector

# Sparsity params
nG = indG.max()    # groups number. same for all GL*

ppnu_grid = np.linspace(0.01, 1, 2)
nu2_grid = np.logspace(-3, 1, 5)         # For Ridge
gamma_grid = np.linspace(0.01, 1.0, 2)  # For GLC


# In[5]:


# Hyperparameter tunning using grid search and 5 fold cross - validation

####################
#### LASSO #########
####################

best_rmse = np.inf

for ppnu in ppnu_grid:

    print(f"------------- Grid search - ppnu = {ppnu} -------------")
    rmse, model = cv_rmse(
        dual_spls_lasso, X, y, ppnu=ppnu
    )
    if rmse < best_rmse:
        best_rmse = rmse
        best_models["Lasso"] = {
            "model": model,
            "rmse": rmse,
            "params": {"ppnu": ppnu}
        }

####################
#### RIDGE #########
####################

best_rmse = np.inf

for ppnu in ppnu_grid:
    for nu2 in nu2_grid:

        print(f"------------- Grid search - ppnu = {ppnu}, nu = {nu2} -------------")

        rmse, model = cv_rmse(
            d_spls_ridge,
            X, y,
            ppnu=ppnu,
            nu2=nu2
        )

        if rmse < best_rmse:
            best_rmse = rmse
            best_models["Ridge"] = {
                "model": model,
                "rmse": rmse,
                "params": {
                    "ppnu": ppnu,
                    "nu2": nu2
                }
            }


####################
#### LS (L2) #######
####################

best_rmse = np.inf

for ppnu in ppnu_grid:

    print(f"------------- Grid search - ppnu = {ppnu} -------------")
    rmse, model = cv_rmse(
        d_spls_ls, X, y, ppnu=ppnu
    )
    if rmse < best_rmse:
        best_rmse = rmse
        best_models["LS"] = {
            "model": model,
            "rmse": rmse,
            "params": {"ppnu": ppnu}
        }

####################
#### GLA ###########
####################

best_rmse = np.inf

for ppnu in ppnu_grid:

    print(f"------------- Grid search - ppnu = {ppnu} -------------")
    rmse, model = cv_rmse(
        dual_spls_gla_random,
        X, y,
        ppnu=np.repeat(ppnu, nG),
        indG=indG,
        verbose=False
    )
    if rmse < best_rmse:
        best_rmse = rmse
        best_models["GLA"] = {
            "model": model,
            "rmse": rmse,
            "params": {"ppnu": ppnu}
        }

####################
#### GLB ###########
####################

best_rmse = np.inf

for ppnu in ppnu_grid:

    print(f"------------- Grid search - ppnu = {ppnu} -------------")
    rmse, model = cv_rmse(
        dual_spls_glb,
        X, y,
        ppnu=np.repeat(ppnu, nG),
        indG=indG,
        verbose=False
    )
    if rmse < best_rmse:
        best_rmse = rmse
        best_models["GLB"] = {
            "model": model,
            "rmse": rmse,
            "params": {"ppnu": ppnu}
        }

####################
#### GLC ###########
####################

best_rmse = np.inf

for ppnu in ppnu_grid:
    for gamma in gamma_grid:

        print(f"------------- Grid search - ppnu = {ppnu}, gamma = {gamma} -------------")
        gamma_vec = np.repeat(gamma, nG)
        gamma_vec /= gamma_vec.sum()

        rmse, model = cv_rmse(
            dual_spls_glc,
            X, y,
            ppnu=np.repeat(ppnu, nG),
            indG=indG,
            gamma=gamma_vec,
            verbose=False
        )

        if rmse < best_rmse:
            best_rmse = rmse
            best_models["GLC"] = {
                "model": model,
                "rmse": rmse,
                "params": {"ppnu": ppnu, "gamma": gamma}
            }


# In[ ]:


# Best models per norms

for k, v in best_models.items():
    print(f"{k:5s} | RMSE = {v['rmse']:.4f} | params = {v['params']}")


# In[ ]:


# Train Dual-sPLS for different norms using best hyperparameters

""" Hyperparameters from params was obtained with grids :

ppnu_grid = np.linspace(0.01, 1, 100)
nu2_grid = np.logspace(-3, 1, 7)          # For Ridge
gamma_grid = np.linspace(0.01, 1.0, 100)  # For GLC

results : 

Lasso | RMSE = 2.3059 | params = {'ppnu': np.float64(0.9500000000000001)}
Ridge | RMSE = 0.2524 | params = {'ppnu': np.float64(1.0), 'nu2': np.float64(0.001)}
LS    | RMSE = 0.2524 | params = {'ppnu': np.float64(1.0)}
GLA   | RMSE = 0.2524 | params = {'ppnu': np.float64(0.01)}
GLB   | RMSE = 0.6976 | params = {'ppnu': np.float64(0.99)}
GLC   | RMSE = 0.5770 | params = {'ppnu': np.float64(1.0), 'gamma': np.float64(0.87)}

Grids was dwarfed for computationnal time reasons"""

params = { # From grid search
    "Lasso": {"ppnu": 0.95},
    "Ridge": {"ppnu": 1.0, "nu2": 0.001},
    "LS":    {"ppnu": 1.0},
    "GLA":   {"ppnu": np.repeat(0.2, nG)}, # 0.01
    "GLB":   {"ppnu": np.repeat(0.99, nG)},
    "GLC":   {
                "ppnu": np.repeat(0.8, nG), # 1.0
                "gamma": np.repeat(1.0 / nG, nG)
    }
}

# ---------- Training ---------- #
models = {}

models["Lasso"] = dual_spls_lasso(
    X, y, n_components=n_comp, **params["Lasso"]
)

models["Ridge"] = d_spls_ridge(
    X, y, n_components=n_comp, **params["Ridge"]
)

models["LS"] = d_spls_ls(
    X, y, n_components=n_comp, **params["LS"]
)

models["GLA"] = dual_spls_gla_random(
    X, y, n_components=n_comp, indG=indG, **params["GLA"]
)

models["GLB"] = dual_spls_glb(
    X, y, n_components=n_comp, indG=indG, **params["GLB"]
)

models["GLC"] = dual_spls_glc(
    X, y, n_components=n_comp, indG=indG, **params["GLC"]
)



# In[ ]:


# RMSE from number of latent components for different norms

rmse_curves = {}

for name, model in models.items():
    rmse = []
    for k in range(n_comp):
        Bk = model["Bhat"][:, k]
        intercept = model["intercept"][k]
        y_pred = (X - model["Xmean"]) @ Bk + intercept
        rmse.append(np.sqrt(mean_squared_error(y, y_pred)))
    rmse_curves[name] = np.array(rmse)


plt.figure(figsize=(18,5))

for name, rmse in rmse_curves.items():
    plt.plot(
        range(1, n_comp+1),
        rmse,
        marker="o",
        label=name
    )

plt.xlabel("Latent component")
plt.ylabel("RMSE (validation)")
plt.title("RMSE for components values for different norms")
plt.legend()
plt.show()
plt.tight_layout()
plt.show()


# In[ ]:


# Coefficients of the regression for all norms

best_components = {
    name: np.argmin(rmse) for name, rmse in rmse_curves.items()
}

plt.figure(figsize=(12,6))

for name, model in models.items():
    k_opt = best_components[name]
    beta = model["Bhat"][:, k_opt]

    plt.plot(
        beta,
        label=f"{name} (k={k_opt+1})",
        linewidth=1.5
    )

plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Spectral variable index")
plt.ylabel("Regression coefficient")
plt.title("Dual-sPLS regression coefficients (optimal hyperparameters)")
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# Beta coefficients of the regression for differents values of hyperparameter

def plot_coeff_values(
    models,
    model_names,
    X, y,
    n_components,
    ppnu_values,
    indG=None,
    fixed_params=None,
    use_last_component=True,
    cmap_name="viridis"
):
    n_models = len(models)

    fig, axes = plt.subplots(
        n_models, 1,
        figsize=(12, 2.2 * n_models),
        sharex=True
    )

    if n_models == 1:
        axes = [axes]

    cmap = plt.colormaps[cmap_name]
    norm = colors.Normalize(
        vmin=min(ppnu_values),
        vmax=max(ppnu_values)
    )

    for ax, model_fun, name, params in zip(
        axes, models, model_names, fixed_params
    ):
        for ppnu in ppnu_values:
            try:
                kwargs = params.copy()

                if name in ["GLA", "GLB", "GLC"]:
                    kwargs["ppnu"] = np.repeat(ppnu, indG.max())
                    kwargs["indG"] = indG
                else:
                    kwargs["ppnu"] = ppnu

                model = model_fun(
                    X, y,
                    n_components=n_components,
                    **kwargs
                )

                k = -1 if use_last_component else 0
                beta = model["Bhat"][:, k]

                ax.plot(
                    beta,
                    color=cmap(norm(ppnu)),
                    linewidth=1
                )

            except Exception:
                continue

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(name, loc="left", fontsize=11)
        ax.set_ylabel(r"$\beta$")

        # --- Colorbar proprement attachée ---
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="vertical",
            fraction=0.025,
            pad=0.02
        )
        cbar.set_label(r"$ppnu$")

    axes[-1].set_xlabel("Spectral variable index")

    fig.suptitle(
        "Dual-sPLS regression coefficients paths\n(color = ppnu)",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()


# In[ ]:


# Params
ppnu_values = np.linspace(0.01, 1.0, 8)

models = [
    dual_spls_lasso,
    d_spls_ridge,
    d_spls_ls,
    dual_spls_gla_random,
    dual_spls_glb,
    dual_spls_glc
]

model_names = ["Lasso", "Ridge", "LS", "GLA", "GLB", "GLC"]

fixed_params = [
    {},                         # Lasso
    {"nu2": 0.001},             # Ridge
    {},                         # LS
    {"verbose": False},         # GLA
    {"verbose": False},         # GLB
    {                           # GLC
        "gamma": np.repeat(1.0 / nG, nG),
        "verbose": False
    }
]

plot_coeff_values(
    models=models,
    model_names=model_names,
    X=X,
    y=y,
    n_components=n_comp,
    ppnu_values=ppnu_values,
    indG=indG,
    fixed_params=fixed_params
)


# In[ ]:




