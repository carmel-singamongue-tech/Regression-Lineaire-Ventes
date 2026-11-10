"""Fonctions communes au TP de régression linéaire multiple."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RACINE = Path(__file__).resolve().parent
DOSSIER_RESULTATS = RACINE / "resultats"
DOSSIER_RESULTATS.mkdir(exist_ok=True)

def generer_donnees():
    """Reproduit exactement la génération du sujet (graine 42)."""
    rng = np.random.default_rng(42)
    m = 300
    pub = rng.uniform(1, 10, m)
    prix = rng.uniform(2, 8, m)
    saison = rng.integers(0, 2, m)  # 0 = sèche, 1 = pluies
    ventes = 10 + 2*pub - 1.5*prix + 3*saison + rng.normal(0, 2, m)
    return pd.DataFrame({
        "pub": pub, "prix": prix, "saison": saison, "ventes": ventes
    })

def preparer_donnees():
    """Sépare les données et ajuste le scaler uniquement sur le train."""
    df = generer_donnees()
    X = df[["pub", "prix", "saison"]]
    y = df["ventes"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return df, X_train, X_test, y_train, y_test, scaler, X_train_s, X_test_s

def convertir_coefficients_bruts(intercept_std, coefficients_std, scaler):
    """Convertit les coefficients d'entrées standardisées vers les unités brutes."""
    coefficients_bruts = np.asarray(coefficients_std) / scaler.scale_
    intercept_brut = intercept_std - np.sum(
        np.asarray(coefficients_std) * scaler.mean_ / scaler.scale_
    )
    return float(intercept_brut), coefficients_bruts

def enregistrer_texte(nom_fichier, contenu):
    chemin = DOSSIER_RESULTATS / nom_fichier
    chemin.write_text(contenu, encoding="utf-8")
    return chemin
