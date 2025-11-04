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
'Kukatpally', 'Kondapur', 'Manikonda', 'Nizampet', 'Gachibowli', 'Miyapur', 'Hitech City', 'Kokapet', 'Nanakramguda', 'Banjara Hills', 'Puppalaguda', 'Narsingi', 'Gajularamaram', 'Tellapur', 'Jubilee Hills', 'Madhapur', 'Nallagandla Gachibowli', 'Appa Junction Peerancheru', 'Bachupally', 'Beeramguda', 'Begumpet', 'Sanath Nagar', 'Appa Junction', 'Balanagar', 'Serilingampally', 'Pragathi Nagar Kukatpally', 'Alwal', 'Mallapur', 'Nagole', 'Krishna Reddy Pet', 'Kollur Road', 'West Marredpally', 'Kompally', 'Bachupally Road', 'LB Nagar', 'East Marredpally', 'Chandanagar', 'Bandlaguda Jagir', 'Shaikpet', 'Malkajgiri', 'Patancheru', 'Attapur', 'Gajulramaram Kukatpally', 'Adibatla', 'Aminpur', 'Gandipet', 'Toli Chowki', 'TellapurOsman Nagar Road', 'Madinaguda', 'Pati', 'Mehdipatnam', 'Somajiguda', 'Nallakunta', 'Mallampet', 'Tarnaka', 'Ashok Nagar', 'Chaitanyapuri', 'AS Rao Nagar', 'ECIL', 'Boduppal', 'Sainikpuri', 'Moosapet', 'Sri Nagar Colony', 'Suchitra', 'ECIL Main Road', 'Moula Ali', 'Uppal', 'Pragathi Nagar', 'Yapral', 'Darga Khaliz Khan', 'Nacharam', 'Himayat Nagar', 'Gopanpally', 'Bolarum', 'Kothaguda', 'Rajendra Nagar', 'Uppal Kalan', 'Kachiguda', 'Saroornagar', 'Patancheru Shankarpalli Road', 'Habsiguda', 'Dammaiguda', 'Kushaiguda', 'Hyder Nagar', 'KPHB', 'Kapra', 'ECIL Cross Road', 'Hafeezpet', 'Medchal', 'Whisper Valley', 'Barkatpura', 'Sun City', 'BK Guda Road', 'new nallakunta', 'Ramnagar Gundu', 'Shadnagar', 'Pragati Nagar', 'Old Bowenpally', 'BK Guda Internal Road', 'Padmarao Nagar', 'Kollur', 'Uppalguda', 'Allwyn Colony', 'Karmanghat', 'Domalguda', 'Zamistanpur', 'Trimalgherry', 'Masab Tank', 'Amberpet', 'Vanasthalipuram', 'Khajaguda Nanakramguda Road', 'Film Nagar', 'financial District', 'Lingampalli', 'Tirumalgiri', 'Nandagiri Hills', 'Alkapur township', 'Bagh Amberpet', 'BHEL', 'KRCR Colony Road', 'Central Excise Colony Hyderabad', 'Miyapur Bachupally Road', 'nizampet road', 'Pragathi Nagar Road', 'Tukkuguda Airport View Point Road', 'Aushapur', 'Safilguda', 'muthangi', 'Ramachandra Puram', 'Qutub Shahi Tombs', 'Kistareddypet', 'Neknampur', 'Bollaram', 'Bowenpally', 'DD Colony', 'Meerpet', 'Venkat Nagar Colony', 'Kondakal', 'Mettuguda', 'raidurgam', 'Pocharam', 'Hakimpet', 'Lakdikapul', 'Mansoorabad', 'Rajbhavan Road Somajiguda', 'Boiguda', 'Hitex Road', 'Cherlapalli', 'Rhoda Mistri Nagar', 'Chintalmet', 'KTR Colony', 'Dilsukh Nagar', 'Abids', 'Quthbullapur', 'Ambedkar Nagar', 'Chintalakunta', 'Ghansi Bazaar', 'Madhura Nagar', 'Chinthal Basthi', 'Murad Nagar', 'Adda Gutta', 'Whitefields', 'ALIND Employees Colony', 'Khizra Enclave', 'Mayuri Nagar', 'Methodist Colony', 'Ameerpet', 'Happy Homes Colony', 'Old Nallakunta', 'Padma Colony', 'Sangeet Nagar', 'Narayanguda', 'NRSA Colony', 'Matrusri Nagar', 'Paramount Colony Toli Chowki', 'D D Colony', 'Isnapur', 'Chititra Medchal', 'Tilak Nagar', 'Kokapeta Village', 'HMT Hills', 'New Maruthi Nagar', 'Madhavaram Nagar Colony', 'IDPL Colony', 'Banjara Hills Road Number 12', 'Panchavati Colony Manikonda', 'Gopal Nagar', 'Bachupaly Road Miyapur', 'Miyapur HMT Swarnapuri Colony', 'Hydershakote', 'Nallagandla Road', 'Basheer Bagh', 'Arvind Nagar Colony', 'Alapathi Nagar', 'Kismatpur', 'Ameenpur', 'Chintradripet', 'Dullapally', 'Vivekananda Nagar Colony', 'Saket', 'Whitefield', 'Karimnagar', 'Dr A S Rao Nagar Rd', 'Sun City Padmasri Estates', 'Beeramguda Road', 'Jhangir Pet', 'Almasguda', 'Mailardevpally', 'Bongloor', 'Moti Nagar', 'Usman Nagar', 'manneguda', 'Kavuri Hills', 'Ring Road', 'JNTU', 'Shamshabad', 'Srisailam Highway', 'Residential Flat Machavaram', 'Santoshnagar', 'Tolichowki', 'Domalguda Road', 'hyderabad', 'Chikkadapally', 'Kothapet', 'Shankarpalli', 'Picket', 'Baghlingampally', 'Neredmet', 'Macha Bolarum', 'Kowkur', 'Sikh Village', 'Rakshapuram', 'west venkatapuram', 'Vidyanagar Adikmet', 'Old Alwal', 'Secunderabad Railway Station Road', 'Balapur', 'Hastinapur', 'chandrayangutta', 'infrequent_if_exist'
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