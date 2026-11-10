"""Étape 2 — Descente de gradient implémentée à la main."""
import time
import numpy as np
import matplotlib.pyplot as plt
from commun import preparer_donnees, DOSSIER_RESULTATS

ALPHA = 0.05
ITERATIONS = 10000

def cout(X, y, w):
    m = len(y)
    erreurs = X @ w - y
    return np.sum(erreurs ** 2) / (2 * m)

def gradient(X, y, w):
    m = len(y)
    return (X.T @ (X @ w - y)) / m

def descente_gradient(X, y, alpha=ALPHA, iterations=ITERATIONS):
    w = np.zeros(X.shape[1])
    historique = []
    for _ in range(iterations):
        w -= alpha * gradient(X, y, w)
        historique.append(cout(X, y, w))
    return w, historique

df, X_train, X_test, y_train, y_test, scaler, X_train_s, X_test_s = preparer_donnees()
X_train_b = np.c_[np.ones(len(X_train_s)), X_train_s]
y = y_train.to_numpy()

debut = time.perf_counter()
w_gd, historique = descente_gradient(X_train_b, y)
temps = time.perf_counter() - debut

print("Poids standardisés [constante, pub, prix, saison] :")
print(w_gd)
print("Coût initial :", historique[0])
print("Coût final :", historique[-1])
print(f"Temps : {temps:.6f} seconde(s)")

plt.figure(figsize=(7, 5))
plt.plot(historique)
plt.xlabel("Itérations")
plt.ylabel("Coût J(w)")
plt.title("Convergence de la descente de gradient")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(DOSSIER_RESULTATS / "02_courbe_cout.png", dpi=150)
plt.show()

np.savez(
    DOSSIER_RESULTATS / "descente_gradient.npz",
    poids=w_gd, historique=np.asarray(historique), temps=temps
)
