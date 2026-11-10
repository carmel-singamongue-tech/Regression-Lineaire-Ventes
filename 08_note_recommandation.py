"""Étape 8 — Générer une note de recommandation à partir des résultats."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from commun import preparer_donnees, DOSSIER_RESULTATS, enregistrer_texte

df, X_train, X_test, y_train, y_test, scaler, X_train_s, X_test_s = preparer_donnees()
modele = LinearRegression().fit(X_train_s, y_train)
pred = modele.predict(X_test_s)

# Conversion des coefficients vers les unités d'origine.
coef = modele.coef_ / scaler.scale_
intercept = modele.intercept_ - np.sum(modele.coef_ * scaler.mean_ / scaler.scale_)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

note = f"""
NOTE DE RECOMMANDATION — MODÈLE DE RÉGRESSION LINÉAIRE

Résultats estimés sur le jeu d'entraînement, évalués sur le jeu de test :

- Constante estimée : {intercept:.3f}
- Effet associé à un million supplémentaire de budget publicitaire :
  {coef[0]:.3f} million(s) de ventes, à prix et saison constants.
- Effet associé à une hausse de 1 000 F CFA du prix moyen :
  {coef[1]:.3f} million(s) de ventes, à publicité et saison constantes.
- Différence estimée entre saison des pluies et saison sèche :
  {coef[2]:.3f} million(s) de ventes, toutes choses égales par ailleurs.
- RMSE sur le jeu de test : {rmse:.3f} million(s) de ventes.
- R² sur le jeu de test : {r2:.3f}.

INTERPRÉTATION POUR LE DÉCIDEUR
Le coefficient publicitaire représente l'association moyenne estimée entre le budget publicitaire
et les ventes, lorsque le prix et la saison sont maintenus constants. Le coefficient du prix
représente l'association estimée avec les ventes pour une hausse de 1 000 F CFA. Le coefficient
saisonnier compare la saison des pluies (1) à la saison sèche (0).

La RMSE donne un ordre de grandeur de l'erreur de prédiction dans l'unité des ventes.

LIMITES
Les données sont simulées et ne décrivent pas nécessairement un marché réel. L'échantillon
contient 300 observations. Le modèle suppose des effets linéaires et additifs. Des facteurs
comme la concurrence, la distribution, les stocks, le revenu des clients ou les campagnes
promotionnelles ne sont pas représentés. Une association statistique ne prouve pas à elle seule
une relation de cause à effet.

Conclusion : utiliser ces résultats comme démonstration méthodologique. Avant une décision
commerciale réelle, réestimer et valider le modèle sur des données observées et plus riches.
"""
chemin = enregistrer_texte("08_note_recommandation.txt", note.strip())
print(note)
print("\nNote enregistrée :", chemin)
