"""Étape 7 — Validation Ridge/Lasso et test avec dix variables de bruit."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from commun import preparer_donnees, DOSSIER_RESULTATS

df, X_train, X_test, y_train, y_test, scaler, X_train_s, X_test_s = preparer_donnees()

# Validation interne au jeu d'entraînement : le test reste de côté.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)
scaler_val = StandardScaler()
X_fit_s = scaler_val.fit_transform(X_fit)
X_val_s = scaler_val.transform(X_val)

alphas = [0.01, 0.1, 1, 10]
lignes = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha).fit(X_fit_s, y_fit)
    pred_r = ridge.predict(X_val_s)
    rmse_r = np.sqrt(mean_squared_error(y_val, pred_r))
    lignes.append(["Ridge", alpha, rmse_r])

    lasso = Lasso(alpha=alpha, max_iter=100000).fit(X_fit_s, y_fit)
    pred_l = lasso.predict(X_val_s)
    rmse_l = np.sqrt(mean_squared_error(y_val, pred_l))
    lignes.append(["Lasso", alpha, rmse_l])

scores = pd.DataFrame(lignes, columns=["Modele", "Alpha", "RMSE_validation"])
print("===== SCORES DE VALIDATION =====")
print(scores.round(5).to_string(index=False))
scores.to_csv(DOSSIER_RESULTATS / "07_scores_validation.csv", index=False)

alpha_ridge = scores[scores.Modele == "Ridge"].sort_values("RMSE_validation").iloc[0]["Alpha"]
alpha_lasso = scores[scores.Modele == "Lasso"].sort_values("RMSE_validation").iloc[0]["Alpha"]
print("\nAlpha Ridge retenu :", alpha_ridge)
print("Alpha Lasso retenu :", alpha_lasso)

ridge_final = Ridge(alpha=alpha_ridge).fit(X_train_s, y_train)
lasso_final = Lasso(alpha=alpha_lasso, max_iter=100000).fit(X_train_s, y_train)
rmse_ridge_test = np.sqrt(mean_squared_error(y_test, ridge_final.predict(X_test_s)))
rmse_lasso_test = np.sqrt(mean_squared_error(y_test, lasso_final.predict(X_test_s)))
print(f"RMSE Ridge sur test : {rmse_ridge_test:.5f}")
print(f"RMSE Lasso sur test : {rmse_lasso_test:.5f}")

# Ajout de dix colonnes de bruit indépendantes, sans effet sur ventes.
rng_bruit = np.random.default_rng(123)
bruit_train = rng_bruit.normal(size=(len(X_train), 10))
bruit_test = rng_bruit.normal(size=(len(X_test), 10))
X_train_aug = np.column_stack([X_train.to_numpy(), bruit_train])
X_test_aug = np.column_stack([X_test.to_numpy(), bruit_test])

scaler_aug = StandardScaler()
X_train_aug_s = scaler_aug.fit_transform(X_train_aug)
X_test_aug_s = scaler_aug.transform(X_test_aug)

lasso_bruit = Lasso(alpha=alpha_lasso, max_iter=200000)
lasso_bruit.fit(X_train_aug_s, y_train)

noms = ["pub", "prix", "saison"] + [f"bruit_{i}" for i in range(1, 11)]
coeffs = pd.Series(lasso_bruit.coef_, index=noms, name="coefficient_lasso")
print("\n===== COEFFICIENTS AVEC DIX VARIABLES DE BRUIT =====")
print(coeffs.round(6))
print("\nNombre de coefficients de bruit nuls :",
      int(np.sum(np.isclose(lasso_bruit.coef_[3:], 0))),
      "/ 10")
coeffs.to_csv(DOSSIER_RESULTATS / "07_coefficients_lasso_bruit.csv")
