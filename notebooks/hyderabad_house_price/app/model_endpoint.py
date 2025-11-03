import joblib
import pandas as pd
import io
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file

# --- Configuración Inicial ---
app = Flask(__name__)

# --- Constantes y Listas de Columnas ---
# (Extraídas de tu clase)

NUMERIC_COLUMNS = ['Area', 'No. of Bedrooms']
CATEGORICAL_COLUMNS = ['Location']
CHECKBOX_COLUMNS = [
    'Resale', 'MaintenanceStaff', 'Gymnasium', 'SwimmingPool', 'LandscapedGardens',
    'JoggingTrack', 'RainWaterHarvesting', 'IndoorGames', 'ShoppingMall', 'Intercom',
    'SportsFacility', 'ATM', 'ClubHouse', 'School', '24X7Security', 'PowerBackup',
    'CarParking', 'StaffQuarter', 'Cafeteria', 'MultipurposeRoom', 'Hospital',
    'WashingMachine', 'Gasconnection', 'AC', 'Wifi', "Children'splayarea",
    'LiftAvailable', 'BED', 'VaastuCompliant', 'Microwave', 'GolfCourse', 'TV',
    'DiningTable', 'Sofa', 'Wardrobe', 'Refrigerator'
]
# Todas las columnas que el modelo espera
ALL_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + CHECKBOX_COLUMNS

# NOTA: Esta lista DEBE coincidir con las locaciones con las que tu modelo fue entrenado.
# Idealmente, la guardas durante el entrenamiento y la cargas aquí.
# Por ahora, usamos una lista de ejemplo.
MOCK_LOCATIONS = [
    'Gachibowli', 'Manikonda', 'Kukatpally', 'Miyapur', 'Hitech City',
    'Banjara Hills', 'Jubilee Hills', 'Kondapur', 'Other'
]

# --- Cargar el Modelo ---
MODEL_FILE = 'datasets/processed/housing_prices/best_hyderabad_house_price_model.pkl'
try:
    model = joblib.load(MODEL_FILE)
    print(f"Modelo '{MODEL_FILE}' cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo del modelo '{MODEL_FILE}'")
    print("Asegúrate de haber entrenado y guardado el modelo primero.")
    model = None
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None


# --- Endpoint 1: Cargar la Página Principal (GET) ---
@app.route('/', methods=['GET'])
def index():
    """Muestra el formulario web."""
    return render_template(
        'index.html',
        locations=MOCK_LOCATIONS,
        checkbox_cols=CHECKBOX_COLUMNS
    )

# --- Endpoint 2: Predicción desde Formulario (POST) ---
@app.route('/predict', methods=['POST'])
def predict_form():
    """Maneja los datos del formulario web."""
    if model is None:
        return render_template(
            'index.html',
            locations=MOCK_LOCATIONS,
            checkbox_cols=CHECKBOX_COLUMNS,
            prediction_text="Error: El modelo no está cargado."
        )

    try:
        # 1. Recolectar datos del formulario
        form_data = request.form
        data_dict = {}

        # 2. Procesar columnas numéricas y categóricas
        data_dict['Area'] = [float(form_data.get('Area', 0))]
        data_dict['No. of Bedrooms'] = [int(form_data.get('No. of Bedrooms', 0))]
        data_dict['Location'] = [form_data.get('Location')]

        # 3. Procesar todos los checkboxes (1 si está marcado, 0 si no)
        for col in CHECKBOX_COLUMNS:
            data_dict[col] = [1 if form_data.get(col) else 0]

        # 4. Crear el DataFrame
        # Nos aseguramos que tenga todas las columnas que el modelo espera
        df = pd.DataFrame(data_dict)
        df = df[ALL_COLUMNS] # Reordena si es necesario (aunque el preprocesador lo maneja por nombre)

        # 5. Realizar la predicción
        prediction = model.predict(df)
        prediction_value = prediction[0]

        # 6. Devolver la misma página con el resultado
        return render_template(
            'index.html',
            locations=MOCK_LOCATIONS,
            checkbox_cols=CHECKBOX_COLUMNS,
            prediction_text=f"Precio Estimado: ${prediction_value:,.2f}"
        )

    except Exception as e:
        return render_template(
            'index.html',
            locations=MOCK_LOCATIONS,
            checkbox_cols=CHECKBOX_COLUMNS,
            prediction_text=f"Error en la predicción: {e}"
        )

# --- Endpoint 3: Predicción desde Archivo CSV (POST) ---
@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    """Maneja la subida de un archivo CSV y devuelve un CSV con predicciones."""
    if model is None:
        return "Error: El modelo no está cargado.", 500

    # 1. Validar archivo
    if 'file' not in request.files:
        return "No se encontró ningún archivo.", 400
    
    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.csv'):
        return "Archivo no válido. Se espera un .csv", 400

    try:
        # 2. Leer CSV
        data = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(data))

        # 3. Validar columnas
        if not all(col in df.columns for col in ALL_COLUMNS):
            missing = [col for col in ALL_COLUMNS if col not in df.columns]
            # Si falla, devolvemos un error a la página principal
            return render_template(
                'index.html',
                locations=MOCK_LOCATIONS,
                checkbox_cols=CHECKBOX_COLUMNS,
                csv_error=f"Error: Faltan columnas en el CSV: {missing}"
            )

        # 4. Realizar predicciones
        predictions = model.predict(df[ALL_COLUMNS])
        df['PredictedPrice'] = predictions

        # 5. Preparar archivo de salida
        output = io.BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)

        # 6. Devolver el archivo para descarga
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='predictions.csv'
        )

    except Exception as e:
        return render_template(
            'index.html',
            locations=MOCK_LOCATIONS,
            checkbox_cols=CHECKBOX_COLUMNS,
            csv_error=f"Error procesando el archivo: {e}"
        )


# --- Ejecutar la App ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)