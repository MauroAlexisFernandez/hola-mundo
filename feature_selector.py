# ===============================
# Clase FeatureSelector personalizada
# ===============================
# Esta clase permite seleccionar el subconjunto específico necesario de features.
# Es útil como paso del pipeline para mantener únicamente las columnas relevantes que requiere el modelo entrenado para predecir.


# Importo las librerías necesarias
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features):
        # Lista de nombres de columnas que se desean conservar
        self.feature_names = selected_features

    def fit(self, X, y=None):
        # No requiere ajuste (fit) ya que la selección está basada en nombres conocidos
        return self

    def transform(self, X):
        print("➡️ Columnas recibidas en X (shape):", X.shape)
        print("➡️ Número de features seleccionadas:", len(self.feature_names))

        # Si no es DataFrame, intento convertir
        if not isinstance(X, pd.DataFrame):
            try:
                # Si no lo es (ej., una sparse matrix), intenta convertirlo a DataFrame
                X = pd.DataFrame(X.toarray(), columns=self.feature_names)
                print("✅ Se reconstruyó el DataFrame desde matriz dispersa.")
            except Exception as e:
                # Si falla la reconstrucción, da un mensaje de error explícito
                raise ValueError(
                    "❌ No se pudo reconstruir el DataFrame desde sparse matrix.\n"
                    "Es posible que los nombres en self.feature_names no coincidan con las columnas reales."
                ) from e

        # Devuelvo solo las columnas seleccionadas
        try:
            return X[self.feature_names]
        except KeyError as e:
             # Si alguna columna no está presente, muestra cuáles faltan
            raise ValueError(
                f"❌ Algunas columnas de self.feature_names no están presentes en X.\n"
                f"Columnas faltantes: {set(self.feature_names) - set(X.columns)}"
            ) from e

