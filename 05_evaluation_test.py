"""Étape 5 — Évaluation sur test et comparaison à la prédiction moyenne."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from commun import preparer_donnees, DOSSIER_RESULTATS

df, X_train, X_test, y_train, y_test, scaler, X_train_s, X_test_s = preparer_donnees()
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

modele = LinearRegression().fit(X_train_s, y_train_np)
pred_modele = modele.predict(X_test_s)
pred_moyenne = np.full(len(y_test_np), y_train_np.mean())

def mesures(y_reel, y_pred):
    mse = mean_squared_error(y_reel, y_pred)
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_reel, y_pred),
        "R2": r2_score(y_reel, y_pred)
    }

table = pd.DataFrame(
    [mesures(y_test_np, pred_modele), mesures(y_test_np, pred_moyenne)],
    index=["Regression lineaire", "Toujours la moyenne"]
)
print(table.round(5))
table.to_csv(DOSSIER_RESULTATS / "05_evaluation_test.csv")

np.savez(DOSSIER_RESULTATS / "predictions_test.npz",
         y_reel=y_test_np, predictions=pred_modele)
print("\nLe modèle moyen sert de référence : son R² est proche de zéro.")
