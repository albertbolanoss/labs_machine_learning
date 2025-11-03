import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from regresion_metrics_column_definition import Metric

def evaluate_model_metrics(model, features, labels):
    """Calcula un diccionario de métricas de regresión para un modelo y datos dados."""
    predictions = model.predict(features)
    
    mae = mean_absolute_error(labels, predictions)
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, predictions)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    }
    return metrics

def show_model_equation(model, features):
    """
    Muestra los coeficientes de un modelo de regresión lineal en un DataFrame
    y también imprime la ecuación del modelo con los nombres de las columnas.

    Args:
        model: El modelo de regresión lineal entrenado.
        features (list): Una lista con los nombres de las columnas.
    """
    intercepto = model.intercept_
    coeficientes = model.coef_
        
    # Iniciar la cadena de la ecuación con el intercepto
    ecuacion_str = f"y = {intercepto[0]:.4f} "

    # Añadir cada coeficiente con su respectivo nombre de columna
    for nombre, coef in zip(features.columns, coeficientes[0]):
        signo = '+' if coef >= 0 else '-'
        ecuacion_str += f"{signo} {abs(coef):.4f} x ({nombre}) "

    print(ecuacion_str)

def get_model_coeficients_dataframe(model, features): 
    coeficientes = model.coef_

    return pd.DataFrame(
        data=coeficientes.T,
        index=features.columns, 
        columns=['Coeficiente (m)']
    )

def get_best_model_RMSE_R2(metrics, w_rmse = 1.0, w_r2 = 1.0):

    # Normalize and get the rmse cost
    rmse = -metrics[Metric.MEAN_TEST_NEG_RMSE.name]
    delta_rmse = rmse.max() - rmse.min()
    cost_rmse = (rmse - rmse.min()) / delta_rmse if delta_rmse > 0 else 0

    # Normalize and get the r2 cost
    r2 = metrics[Metric.MEAN_TEST_R2.name]
    delta_r2 = r2.max() - r2.min()
    cost_r2 = (r2.max() - r2) / delta_r2 if delta_r2 > 0 else 0

    # Getting the custom score
    metrics[Metric.SCORE.name] = (w_rmse * cost_rmse) + (w_r2 * cost_r2)

    # Finding the best model
    best_model_idx = metrics[Metric.SCORE.name].idxmin()
    best_model_info = metrics.loc[best_model_idx]

    print("--- Best found Model with PCA ---")
    print(f"Index: {best_model_idx}")
    # Use .get to avoid KeyError if column not present
    print(f"Model Name: {best_model_info.get('model_name', 'unknown')}")
    print(f"Score: {best_model_info[Metric.SCORE.name]:.8f} (smaller is better)")
    print("\nModel Metrics:")
    # defensive access for expected metric columns
    print(f"  -> RMSE: {-best_model_info.get('mean_test_neg_rmse', 0):.4f}")
    print(f"  -> R2 Score: {best_model_info.get('mean_test_r2', 0):.4f}")
    print("Hyperparameters:")

    return best_model_idx, best_model_info