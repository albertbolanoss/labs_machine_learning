import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_split_distribution(features, features_train, features_validation, features_test, stratify_col):
    """
    Calcula y grafica la distribución de una columna de estratificación
    a través de los conjuntos de datos original, de entrenamiento, validación y prueba.

    Args:
        original_features (pd.DataFrame): DataFrame completo de características antes de la división.
        train_features (pd.DataFrame): DataFrame de características de entrenamiento.
        val_features (pd.DataFrame): DataFrame de características de validación.
        test_features (pd.DataFrame): DataFrame de características de prueba.
        stratify_col (str): El nombre de la columna usada para la estratificación (ej. 'price_cat').
    """
    # # give the proportion of q1, q2, q3, q4 for each set
    original_dist = features[stratify_col].value_counts(normalize=True).sort_index()
    train_dist = features_train[stratify_col].value_counts(normalize=True).sort_index()
    validation_dist = features_validation[stratify_col].value_counts(normalize=True).sort_index()
    test_dist = features_test[stratify_col].value_counts(normalize=True).sort_index()

    comparison_df = pd.DataFrame({
        "Original": original_dist * 100,
        "Train": train_dist * 100,
        "Validation": validation_dist * 100,
        "Test": test_dist * 100
    })


    plot_data = comparison_df.reset_index().melt(
        id_vars=stratify_col,
        var_name='Set',
        value_name='Percentage'
    )


    #plot_data.rename(columns={stratify_col: 'Category'}, inplace=True)

    # Creamos el gráfico de barras (sin cambios)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Category', y='Percentage', hue='Set', data=plot_data, palette='viridis')

    plt.title(f'Category {stratify_col} Distribution:', fontsize=16)
    plt.xlabel(f'Category {stratify_col}', fontsize=12)
    plt.ylabel('Porcentaje (%)', fontsize=12)
    plt.ylim(0, 30)
    plt.legend(title='Set of data')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # plt.savefig('stratified_split_verification.png')
    plt.show()