from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from enum import Enum

class ReductionType(Enum):
    PCA = "PCA"     # Regresion
    SVD = "SVD"     # Regresion
    LDA = "LDA"     # Clasification

# Función de regresión lineal
def lineal_regresion_model_apply(x_train, y_train, x_val, y_val):
    model = LinearRegression()
    model.fit(x_train, y_train)
    predicts = model.predict(x_val)
    rmse = np.sqrt(mean_squared_error(y_val, predicts))
    r2 = r2_score(y_val, predicts)
    return rmse, r2
    
def evalueate_dimensionality_reduction(reduction_type: ReductionType, x_train, y_train, x_val, y_val, n):
    model = None
    
    if reduction_type == ReductionType.PCA:
        model = PCA(n_components=n)
        x_train_pca = model.fit_transform(x_train)
        x_val_pca = model.transform(x_val)
    elif reduction_type == ReductionType.SVD:
        model = TruncatedSVD(n_components=n)
        x_train_pca = model.fit_transform(x_train)
        x_val_pca = model.transform(x_val)
    elif reduction_type == ReductionType.LDA:
        model = LinearDiscriminantAnalysis(n_components=n)
        x_train_pca = model.fit_transform(x_train, y_train)
        x_val_pca = model.transform(x_val)
    else:
        raise ValueError("Model not supported")
    
    rmse, r2 = lineal_regresion_model_apply(x_train_pca, y_train, x_val_pca, y_val)

    return rmse, r2, model


# Selección del mejor modelo PCA usando score balanceado
def find_best_dimensionality_reduction(reduction_type: ReductionType, x_train, y_train, x_val, y_val, rmse_baseline, alpha=1.0, beta=1.0):
    best_score = float("inf")
    best_n = None
    best_pca_model = None
    best_rmse = None
    best_r2 = None
    max_components = x_train.shape[1]

    print("\nFinding the principal components for the model ", reduction_type.value)

    for n in range(1, max_components + 1):
        rmse, r2, pca_model = evalueate_dimensionality_reduction(reduction_type, x_train, y_train, x_val, y_val, n)
        r2_distance = abs(1 - r2)
        score = alpha * r2_distance + beta * (rmse / rmse_baseline)
        print(f"{reduction_type.value} {n} - R²: {r2:.4f} - RMSE: {rmse:.4f} - Score: {score:.4f}")
        
        # - The ideal R² is 1 ⇒ r2_distance = abs(1 - r2) measures how far you are from 1 (the smaller, the better).
        # - The ideal RMSE is 0 ⇒ rmse / rmse_baseline measures the relative error with respect to a baseline (the smaller, the better).
        if score < best_score:
            best_score = score
            best_n = n
            best_pca_model = pca_model
            best_rmse = rmse
            best_r2 = r2

    print(f"Best {reduction_type.value} Model: n: {best_n} R²: {best_r2:.4f} RMSE: {best_rmse:.4f} Score: {best_score:.4f}")
    return best_rmse, best_r2, best_pca_model