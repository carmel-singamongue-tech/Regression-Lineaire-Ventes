"""Étape 6 — Diagnostic graphique des résidus."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from commun import preparer_donnees, DOSSIER_RESULTATS

df, X_train, X_test, y_train, y_test, scaler, X_train_s, X_test_s = preparer_donnees()
modele = LinearRegression().fit(X_train_s, y_train)
pred = modele.predict(X_test_s)
residus = y_test.to_numpy() - pred

plt.figure(figsize=(7, 5))
plt.scatter(pred, residus, alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Ventes prédites")
plt.ylabel("Résidus (réel - prédit)")
plt.title("Résidus en fonction des prédictions")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(DOSSIER_RESULTATS / "06_residus_predictions.png", dpi=150)
plt.show()

plt.figure(figsize=(7, 5))
plt.hist(residus, bins=15, edgecolor="black")
plt.xlabel("Résidus")
plt.ylabel("Fréquence")
plt.title("Histogramme des résidus")
plt.tight_layout()
plt.savefig(DOSSIER_RESULTATS / "06_histogramme_residus.png", dpi=150)
plt.show()

print("Corrélation entre variables explicatives (train) :")
print(X_train.corr().round(3))
print("\nMoyenne des résidus :", round(float(np.mean(residus)), 5))
print("Écart-type des résidus :", round(float(np.std(residus)), 5))
print("\nÀ examiner : résidus répartis autour de zéro, sans tendance ni dispersion croissante.")
