"""Étape 0 — Générer et enregistrer les 300 observations."""
from commun import generer_donnees, DOSSIER_RESULTATS

df = generer_donnees()
chemin = DOSSIER_RESULTATS / "donnees.csv"
df.to_csv(chemin, index=False, encoding="utf-8-sig")

print("Jeu de données créé :", chemin)
print("\nPremières lignes :")
print(df.head().to_string(index=False))
print("\nDimensions :", df.shape)
print("\nDictionnaire des variables :")
print("- pub : budget publicitaire en millions")
print("- prix : prix moyen en milliers de F CFA")
print("- saison : 0 = sèche ; 1 = pluies")
print("- ventes : ventes en millions")
