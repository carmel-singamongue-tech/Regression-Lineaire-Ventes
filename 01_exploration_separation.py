"""Étape 1 — Statistiques, corrélations, nuage de points et séparation."""
import matplotlib.pyplot as plt
from commun import preparer_donnees, DOSSIER_RESULTATS

df, X_train, X_test, y_train, y_test, scaler, X_train_s, X_test_s = preparer_donnees()

print("===== APERÇU =====")
print(df.head())
print("\n===== STATISTIQUES DESCRIPTIVES =====")
print(df.describe().round(3))
print("\n===== CORRÉLATIONS =====")
print(df.corr().round(3))

plt.figure(figsize=(7, 5))
plt.scatter(df["pub"], df["ventes"], alpha=0.65)
plt.xlabel("Budget publicitaire (millions)")
plt.ylabel("Ventes (millions)")
plt.title("Ventes en fonction du budget publicitaire")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(DOSSIER_RESULTATS / "01_nuage_ventes_publicite.png", dpi=150)
plt.show()

print("\nTaille apprentissage :", len(X_train), "observations")
print("Taille test :", len(X_test), "observations")
print("\nMoyennes du train utilisées par le scaler :", scaler.mean_)
print("Écarts-types du train utilisés par le scaler :", scaler.scale_)
print("Standardisation effectuée sans ajuster le scaler sur le test.")
