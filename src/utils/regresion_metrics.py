import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
