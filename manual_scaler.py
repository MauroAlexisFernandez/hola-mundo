# ===============================
# Escalador personalizado con MinMaxScaler
# ===============================
# Esta clase aplica el escalado Min-Max sobre columnas numéricas especificadas para usar dentro de pipeline con app.py y así replicar lo realizado en el Notebook.


# Importo las librerías necesarias.
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


class ManualScaler(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols):
        # Lista de columnas numéricas a escalar
        self.numeric_cols = numeric_cols
        # Inicializa el escalador MinMax (escalado al rango [0, 1])
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        # Ajusta el escalador sólo sobre las columnas numéricas
        self.scaler.fit(X[self.numeric_cols])
        return self # Retorna self para integrarse en pipelines

    def transform(self, X):
        X = X.copy()
        # Aplica el escalado Min-Max sobre las columnas numéricas
        X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
        return X # Retorno el DataFrame transformado