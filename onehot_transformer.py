# ===============================
# Codificador personalizado con OneHotEncoder
# ===============================
# Esta clase replica la codificación One-Hot sobre columnas categóricas especificadas (como las usadas en el Notebook), y la integra como paso del pipeline para app.py


#Importo las librerías necesarias.
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Clase personalizada para codificación One-Hot
class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols):
        # Lista de columnas categóricas a codificar
        self.categorical_cols = categorical_cols
        # Inicializo el codificador OneHotEncoder (devuelve matriz densa y evita errores con categorías desconocidas)
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')


    def fit(self, X, y=None):
        # Convierte las columnas categóricas a string (por seguridad, en caso de que tengan tipos mixtos)
        X_ = X[self.categorical_cols].astype(str)
        # Ajusta el codificador a los datos
        self.encoder.fit(X_)
        # Guarda los nombres de las nuevas columnas codificadas
        self.feature_names_out = self.encoder.get_feature_names_out(self.categorical_cols)
        return self # Retorna self para cumplir con la interfaz de scikit-learn


    def transform(self, X):
        X = X.copy()
        # Selecciona y convierte las columnas categóricas a string
        X_cat = X[self.categorical_cols].astype(str)
        # Aplica la transformación One-Hot
        encoded = self.encoder.transform(X_cat)
        # Convierte el resultado a DataFrame con los nombres de las columnas codificadas
        df_encoded = pd.DataFrame(encoded, columns=self.feature_names_out, index=X.index)
        # Elimina las columnas categóricas originales del DataFrame
        X.drop(columns=self.categorical_cols, inplace=True)
        # Concatena las nuevas columnas codificadas con el resto del DataFrame
        X = pd.concat([X, df_encoded], axis=1)
        return X # Retorno el DataFrame transformado