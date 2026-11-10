"""Étape 3 — Calcul de la solution par équation normale."""
import time
import numpy as np
import pandas as pd
from commun import preparer_donnees, convertir_coefficients_bruts, DOSSIER_RESULTATS

df, X_train, X_test, y_train, y_test, scaler, X_train_s, X_test_s = preparer_donnees()
X_b = np.c_[np.ones(len(X_train_s)), X_train_s]
y = y_train.to_numpy()

debut = time.perf_counter()
w = np.linalg.solve(X_b.T @ X_b, X_b.T @ y)
temps = time.perf_counter() - debut

intercept, coef = convertir_coefficients_bruts(w[0], w[1:], scaler)
print("Poids standardisés [constante, pub, prix, saison] :", w)
print("\nCoefficients en unités originales :")
print(f"Constante : {intercept:.6f}")
for nom, valeur in zip(["pub", "prix", "saison"], coef):
    print(f"{nom} : {valeur:.6f}")
print(f"\nTemps de calcul : {temps:.8f} seconde(s)")

pd.DataFrame([{"methode": "Equation normale", "intercept": intercept,
               "pub": coef[0], "prix": coef[1], "saison": coef[2],
               "temps_secondes": temps}]).to_csv(
    DOSSIER_RESULTATS / "03_equation_normale.csv", index=False
)
np.savez(DOSSIER_RESULTATS / "equation_normale.npz", poids=w, temps=temps)
