# TP — Régression linéaire multiple (VS Code)

Ce dossier contient les fichiers organisés étape par étape selon l'énoncé.
Aucune réinstallation des bibliothèques n'est nécessaire si votre environnement VS Code est déjà configuré.

## 1. Ouvrir le projet
1. Décompressez `TP_Regression_Lineaire_VSCode.zip`.
2. Dans VS Code, choisissez **Fichier > Ouvrir le dossier** et sélectionnez `TP_Regression_Lineaire_VSCode`.
3. Ouvrez le terminal intégré (**Terminal > Nouveau terminal**).
4. Exécutez les fichiers dans l'ordre indiqué ci-dessous.

## 2. Exécution

Depuis le terminal, lancez une commande à la fois :

```powershell
python 00_generation_donnees.py
python 01_exploration_separation.py
python 02_descente_gradient.py
python 03_equation_normale.py
python 04_scikit_learn_comparaison.py
python 05_evaluation_test.py
python 06_diagnostic_residus.py
python 07_regularisation_ridge_lasso.py
python 08_note_recommandation.py
```

Si `python` n'est pas reconnu sous Windows, essayez `py` à la place.

**Important :** exécutez les scripts depuis le dossier racine du projet, pas depuis le sous-dossier `resultats`.
Chaque script importe `commun.py`, qui doit rester dans le même dossier que les scripts.

## 3. Contenu des fichiers
- `commun.py` : génération reproductible des données, séparation train/test et standardisation sans fuite de données.
- `00_generation_donnees.py` : création du jeu de données CSV.
- `01_exploration_separation.py` : statistiques, corrélations, nuage de points et séparation.
- `02_descente_gradient.py` : coût, gradient, descente de gradient et graphique de convergence.
- `03_equation_normale.py` : solution avec `np.linalg.solve`.
- `04_scikit_learn_comparaison.py` : LinearRegression et comparaison des coefficients/temps.
- `05_evaluation_test.py` : MSE, RMSE, MAE et R² face au modèle de la moyenne.
- `06_diagnostic_residus.py` : graphiques et corrélations.
- `07_regularisation_ridge_lasso.py` : validation de Ridge/Lasso et expérience avec dix variables de bruit.
- `08_note_recommandation.py` : synthèse chiffrée à partir des résultats enregistrés.
- `resultats/` : fichiers CSV, graphiques et résultats générés.

## 4. Ordre et précautions
- Les scripts 01 à 08 peuvent être exécutés indépendamment après le premier lancement, car les données sont recréées de manière déterministe dans `commun.py`.
- Les résultats numériques exacts sont ceux de votre exécution. Ne recopiez pas des chiffres approximatifs à la place.
- Le jeu de test n'est pas utilisé pour choisir alpha : le choix de Ridge/Lasso est effectué sur un sous-ensemble de validation tiré du jeu d'entraînement.
- Les variables sont standardisées pour l'optimisation et les modèles pénalisés. Les coefficients sont reconvertis en unités originales pour leur interprétation.
- La saison vaut 0 pour la saison sèche et 1 pour la saison des pluies.

## 5. Résultats attendus
Les coefficients sur les données brutes devraient être proches de :
- constante : 10 ;
- publicité : 2 ;
- prix : -1,5 ;
- saison : 3.

Sur le test, le RMSE devrait être voisin de 2 et le R² voisin de 0,9. Ce sont des repères, pas des valeurs à forcer.

## 6. Dépannage rapide
- `ModuleNotFoundError` : vérifiez l'interpréteur Python sélectionné dans VS Code (**Ctrl+Shift+P > Python: Select Interpreter**).
- `commun` introuvable : vérifiez que le terminal est dans le dossier racine du projet.
- Graphique affiché : fermez sa fenêtre pour poursuivre le script.
- Coût qui diverge : diminuez le taux d'apprentissage `ALPHA` dans `02_descente_gradient.py`.
