import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
from preprocessing import PreprocessingTransformer
from feature_selector import FeatureSelector
from onehot_transformer import OneHotEncoderTransformer
from manual_scaler import ManualScaler
from chat_rag_local import responder_pregunta
import traceback

# --- Configuración de logging ---
# Se establece el formato y nivel de logging para registrar eventos e información
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Inicializo la app Flask ---
app = Flask(__name__)
CORS(app)  # Habilitar CORS para permitir peticiones desde otros dominios (útil para frontend externo)

# --- Cargo el pipeline entrenado ---
# Se carga el modelo/pipeline completo previamente entrenado con joblib
modelo = joblib.load("pipeline_modelo_completo.pkl")

# --- Variables numéricas y requeridas ---
# Se definen los nombres de las variables que deben ser numéricas
numeric_vars = {
    "age", "duration", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"
}
# Conjunto de variables que se esperan obligatoriamente para hacer la predicción
variables_requeridas = {
    "age", "job", "marital", "education", "default", "housing", "loan", "contact",
    "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"
}

# --- Rutas de la aplicación Flask ---
@app.route('/')
def home():
    # Renderiza la página principal con el frontend del chatbot
    return render_template("chat.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Recibe los datos enviados en formato JSON desde el frontend
    datos_usuario = request.json or {}
    # Verifica que se haya enviado un JSON válido
    if not datos_usuario:
        return jsonify({"error": "No se enviaron datos JSON válidos."}), 400

    # Verifica si faltan variables obligatorias
    faltantes = variables_requeridas - datos_usuario.keys()
    if faltantes:
        return jsonify({"error": f"Faltan variables obligatorias: {', '.join(sorted(faltantes))}"}), 400

    # Detecta si se enviaron variables extra no reconocidas
    extra = datos_usuario.keys() - variables_requeridas
    if extra:
        logger.warning(f"Se enviaron variables extra: {extra}")

    # Convierte las variables numéricas a float
    for var in numeric_vars:
        if var in datos_usuario:
            try:
                datos_usuario[var] = float(datos_usuario[var])
            except ValueError:
                return jsonify({"error": f"La variable '{var}' debe ser numérica."}), 400

    try:
        # Crea un DataFrame con los datos del usuario para pasarlos al pipeline
        df = pd.DataFrame([datos_usuario])
        logger.info(f"Predicción recibida con columnas: {df.columns.tolist()}")
        # Realiza la predicción y obtiene la probabilidad de la clase positiva
        prediccion = int(modelo.predict(df)[0])
        probabilidad = float(modelo.predict_proba(df)[0][1])
        # Devuelve la predicción y la probabilidad al frontend
        return jsonify({
            "prediccion": prediccion,
            "probabilidad": round(probabilidad, 4)
        })

    except Exception:
        # Captura errores en la predicción y los registra en el log
        logger.error("Error en predicción:\n" + traceback.format_exc())
        return jsonify({"error": "Error al procesar la predicción. Verifique los datos enviados."}), 500

@app.route('/rag_chat', methods=['POST'])
def rag_chat():
    # Recibe la pregunta enviada desde el frontend para el sistema RAG
    pregunta = request.json.get("pregunta", "").strip()
    if not pregunta:
        return jsonify({"error": "No se recibió una pregunta válida."}), 400

    try:
        # Llama a la función de RAG que genera la respuesta basada en los documentos indexados
        respuesta = responder_pregunta(pregunta)
        return jsonify({"pregunta": pregunta, "respuesta": respuesta})
    except Exception:
        # Captura errores en el procesamiento RAG
        logger.error("Error en RAG:\n" + traceback.format_exc())
        return jsonify({"error": "Ocurrió un error interno al procesar la pregunta."}), 500

# --- Ejecución con Waitress (compatible con Windows y producción) ---
if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 5000)) # Permite configurar el puerto vía variable de entorno
    serve(app, host="0.0.0.0", port=port) # Inicia la app usando Waitress, adecuada para producción
