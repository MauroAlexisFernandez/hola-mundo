# ===============================
# Preprocesamiento manual personalizado como el efectuado en el Notebook para obtener el modelo
# ===============================
# Este transformador replica el preprocesamiento del notebook original para usarlo junto al archivo app.py
# Aplica las reglas manuales, binning, agrupaciones y crea las nuevas variables como en el Notebook.

#Importo las librerías necesarias
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Clase personalizada compatible con scikit-learn
class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    # Método transform: aplica todas las transformaciones al dataset
    def transform(self, X):
        X = X.copy()

        # Reglas personalizadas para imputación
        # Corrige valores 'unknown' en función de otras columnas
        X.loc[(X['age'] > 60) & (X['job'] == 'unknown'), 'job'] = 'retired'
        X.loc[(X['education'] == 'unknown') & (X['job'] == 'management'), 'education'] = 'university.degree'
        X.loc[(X['education'] == 'unknown') & (X['job'] == 'services'), 'education'] = 'high.school'
        X.loc[(X['education'] == 'unknown') & (X['job'] == 'housemaid'), 'education'] = 'basic.4y'
        # Reglas cruzadas para asignar 'job' en base a 'education'
        X.loc[(X['job'] == 'unknown') & (X['education'] == 'basic.4y'), 'job'] = 'blue-collar'
        X.loc[(X['job'] == 'unknown') & (X['education'] == 'basic.6y'), 'job'] = 'blue-collar'
        X.loc[(X['job'] == 'unknown') & (X['education'] == 'basic.9y'), 'job'] = 'blue-collar'
        X.loc[(X['job'] == 'unknown') & (X['education'] == 'professional.course'), 'job'] = 'technician'

        # ==== Limpieza de valores especiales ====
        # pdays = 999 significa "no contactado antes", lo pasamos a 0
        X['pdays'] = X['pdays'].apply(lambda x: 0 if x == 999 else x)

        # Reemplazo de valores de texto por números (orden cronológico)
        X['month'].replace(
            ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'),
            range(1, 13), inplace=True)

        X['day_of_week'].replace(
            ('mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'),
            range(1, 8), inplace=True)

       # Binning de edad
       # Crea la variable categórica 'age_binned' a partir de intervalos
        bins = [0, 25, 40, 60, 140]
        labels = ['young', 'lower middle aged', 'middle aged', 'senior']
        X['age_binned'] = pd.cut(X['age'], bins=bins, labels=labels, right=True, include_lowest=True)

        # Agrupamientos
        # Agrupa tipos de trabajo en categorías más amplias
        job_map = {
            'admin.': 'White-collar', 'management': 'White-collar', 'technician': 'White-collar',
            'blue-collar': 'Blue-collar', 'services': 'Blue-collar', 'housemaid': 'Blue-collar',
            'entrepreneur': 'Self-employed', 'self-employed': 'Self-employed',
            'retired': 'Non-active', 'student': 'Non-active', 'unemployed': 'Non-active',
            'unknown': 'Other'
        }
        X['job_grouped'] = X['job'].map(job_map)

        # Agrupa niveles educativos en categorías más amplias
        education_map = {
            'basic.9y': 'Basic', 'basic.4y': 'Basic', 'basic.6y': 'Basic',
            'high.school': 'Middle', 'professional.course': 'Middle',
            'university.degree': 'Superior', 'unknown': 'Other', 'illiterate': 'Other'
        }
        X['education_grouped'] = X['education'].map(education_map)

        # Nuevas variables combinadas
        # Variable binaria: si fue contactado previamente o no
        X['contacted_previously'] = (X['previous'] >= 1).astype(int)
        # Crea una combinación de edad binned y estado civil
        X['life_stage'] = X['age_binned'].astype(str) + ' & ' + X['marital']
        # Crea una combinación de trabajo y educación (como proxy socioeconómico)
        X['socio-economic'] = X['job'].astype(str) + ' & ' + X['education']

        # Drop columnas redundantes
        # 'nr.employed' es altamente correlacionada con otras variables y fue descartada en el modelo
        X.drop(columns=['nr.employed'], inplace=True)

        return X # Retorna el DataFrame transformado