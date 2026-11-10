"""Étape 4 — LinearRegression et tableau comparatif des trois méthodes."""
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from commun import (
    preparer_donnees, convertir_coefficients_bruts, DOSSIER_RESULTATS
)

df, X_train, X_test, y_train, y_test, scaler, X_train_s, X_test_s = preparer_donnees()
y = y_train.to_numpy()

# Descente de gradient
gd = np.load(DOSSIER_RESULTATS / "descente_gradient.npz")
w_gd = gd["poids"]
t_gd = float(gd["temps"])
i_gd, c_gd = convertir_coefficients_bruts(w_gd[0], w_gd[1:], scaler)

# Équation normale
norm = np.load(DOSSIER_RESULTATS / "equation_normale.npz")
w_n = norm["poids"]
t_n = float(norm["temps"])
i_n, c_n = convertir_coefficients_bruts(w_n[0], w_n[1:], scaler)

# Scikit-learn
modele = LinearRegression()
debut = time.perf_counter()
modele.fit(X_train_s, y)
t_sk = time.perf_counter() - debut
i_sk, c_sk = convertir_coefficients_bruts(modele.intercept_, modele.coef_, scaler)

table = pd.DataFrame([
    ["Descente de gradient", i_gd, c_gd[0], c_gd[1], c_gd[2], t_gd],
    ["Equation normale", i_n, c_n[0], c_n[1], c_n[2], t_n],
    ["scikit-learn", i_sk, c_sk[0], c_sk[1], c_sk[2], t_sk],
], columns=["Methode", "Constante", "Pub", "Prix", "Saison", "Temps_secondes"])

print(table.round(7).to_string(index=False))
table.to_csv(DOSSIER_RESULTATS / "04_comparaison_methodes.csv", index=False)

np.savez(DOSSIER_RESULTATS / "modele_lineaire.npz",
         intercept=modele.intercept_, coefficients=modele.coef_,
         temps=t_sk, intercept_brut=i_sk, coefficients_bruts=c_sk)
print("\nLes coefficients des trois méthodes doivent être très proches.")
