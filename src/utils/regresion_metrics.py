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