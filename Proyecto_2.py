from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#instalar aws library
!pip install awscli

import json, os

# Import credentials from an external configuration file
credentials_path = '/content/drive/MyDrive/Fede_salud_analytics/Copy of credentials.json'

with open(credentials_path, 'r') as file:
    credentials = json.load(file)

# Configurar AWS CLI
!aws configure set aws_access_key_id {credentials['AWS_ACCESS_KEY_ID']}
!aws configure set aws_secret_access_key {credentials['AWS_SECRET_ACCESS_KEY']}
!aws configure set region {credentials['AWS_REGION']}

# Define the bucket name
BUCKET = 'proyecto-1'

# Define the directories as constants
TEST = 'archives/'

LOCAL_WORK_DIR = f'/content/drive/MyDrive/Fede_salud_analytics/{BUCKET}/'

if not os.path.exists(LOCAL_WORK_DIR):
  os.makedirs(LOCAL_WORK_DIR)

# select repository
repo = TEST

local_repo = f'{LOCAL_WORK_DIR}{repo}'
if not os.path.exists(local_repo):
  os.makedirs(local_repo)

# list files in repo
!aws s3 ls s3://{BUCKET}/{repo}



import pandas as pd

# Lista de columnas útiles
columnas_utiles = ['height_m', 'weight_kg', 'bmi_converted', 'CVDINFR4', 'CVDCRHD4',
    'CVDSTRK3', 'SMOKE100', 'DRNKANY5', 'DRNK3GE5', 'AVEDRNK3',
    'EXERANY2', 'DIABETE4', 'ASTHMA3', 'ASTHNOW', 'CVDINFR4',  'DRNKANY5',
    'SMOKDAY2', 'INSULIN1', 'BLDSUGAR', 'CHKHEMO3', 'ADDEPEV3', '_TOTINDA', 'HEIGHT3', 'WEIGHT2', '_BMI5']

# Diccionario de corrección de nombres de columnas
mapeo_nombres = {
    '_RFHYPE6 ': '_RFHYPE6',
    ' CHOLMED3 ': 'CHOLMED3',
    ' TOLDHI3 ': 'TOLDHI3'
}

# Función para procesar un archivo
def procesar_archivo(ruta_archivo):
    # Cargar dataset
    df = pd.read_csv(ruta_archivo)

    # Renombrar columnas mal nombradas
    df.rename(columns=mapeo_nombres, inplace=True)

    # Agregar columnas faltantes con NaN
    for columna in columnas_utiles:
        if columna not in df.columns:
            df[columna] = pd.NA

    # Filtrar solo las columnas útiles
    df = df[columnas_utiles]

    return df

# Procesar cada archivo
df_2019 = procesar_archivo('/content/drive/MyDrive/Salud-Analytics/Mentoria/Fede - Salud Analytics/Proyectos/Datasets/2019.csv')
df_2020 = procesar_archivo('/content/drive/MyDrive/Salud-Analytics/Mentoria/Fede - Salud Analytics/Proyectos/Datasets/2020.csv')
df_2021 = procesar_archivo('/content/drive/MyDrive/Salud-Analytics/Mentoria/Fede - Salud Analytics/Proyectos/Datasets/2021.csv')




# Funcion para convertir la altura
def convert_height(height):
    if height == 7777 or height == 9999:  # "Don't know/Not sure" o "Refused"
        return None
    elif 9061 <= height <= 9998:  # Altura en metros/centímetros
        return round((height - 9000) / 100, 2)  # Convertir centímetros a metros y redondear
    elif 0 < height < 7777:  # Altura en pies/pulgadas
        feet = height // 100  # Extraer los pies
        inches = height % 100  # Extraer las pulgadas
        total_inches = (feet * 12) + inches  # Convertir todo a pulgadas
        return round(total_inches * 0.0254, 2)  # Convertir pulgadas a metros y redondear
    else:
        return None  # Para valores no válidos

# Funcion para convertir el peso
def convert_weight(weight):
    if weight == 7777 or weight == 9999:  # "Don't know/Not sure" o "Refused"
        return None
    elif 9023 <= weight <= 9352:  # Peso en kilogramos
        return round(weight - 9000, 2)  # Remover el '9' inicial y redondear
    elif 0 < weight < 7777:  # Peso en libras
        return round(weight * 0.453592, 2)  # Convertir libras a kilogramos y redondear
    else:
        return None  # Para valores no válidos

# Funciona para convertir el BMI

def convert_bmi(bmi):
    if bmi > 1 :  # Caso donde el valor es válido
        return bmi/100
    else:
        return None # El valor tiene 2 decimales implícitos, se redondea a 2 decimales


# Función para procesar los DataFrames
def process_dataframe(df):
    df['height_m'] = df['HEIGHT3'].apply(convert_height)
    df['weight_kg'] = df['WEIGHT2'].apply(convert_weight)
    df['bmi_converted'] = df['_BMI5'].apply(convert_bmi)
    return df

# Aplicar la función a cada DataFrame
df_2019 = process_dataframe(df_2019)
df_2020 = process_dataframe(df_2020)
df_2021 = process_dataframe(df_2021)



# Vamos a concantenar todos los DF en uno

# Concatenate the DataFrames
combined_df = pd.concat([df_2019, df_2020, df_2021], ignore_index=True)

# Remove duplicated columns if any
combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]


# Dejamos solo las variables con las que vamos a trabjar
variables_utiles = ['bmi_converted', '_TOTINDA', 'SMOKE100',
                    'DRNKANY5', 'DIABETE4', 'CVDINFR4', 'CVDCRHD4', 'CVDSTRK3']

combined_df = combined_df[variables_utiles]



combined_df = combined_df[
    combined_df['CVDINFR4'].isin([1, 2]) &
    combined_df['CVDCRHD4'].isin([1, 2]) &
    combined_df['CVDSTRK3'].isin([1, 2])
]



from sklearn.utils import resample
import pandas as pd

# Lista de targets a balancear
targets = ['CVDINFR4','CVDCRHD4', 'CVDSTRK3']

# Crear una lista para almacenar los DataFrames balanceados
balanced_dfs = []

# Iterar sobre cada target
for target in targets:

    # Separar las clases
    class_yes = combined_df[combined_df[target] == 1]  # Clase positiva (Sí)
    class_no = combined_df[combined_df[target] == 2]   # Clase negativa (No)

    # Determinar el tamaño de la clase minoritaria
    min_count = min(len(class_yes), len(class_no))

    # # Reducir la clase mayoritaria a min_count usando resample
    class_yes_balanced = resample(class_yes, replace=False, n_samples=min_count, random_state=42)
    class_no_balanced = resample(class_no, replace=False, n_samples=min_count, random_state=42)


    # # Combinar ambas clases balanceadas
    balanced_df = pd.concat([class_yes_balanced, class_no_balanced]).reset_index(drop=True)

    # # Añadir a la lista de DataFrames balanceados
    balanced_dfs.append(balanced_df)

# # Combinar todos los DataFrames balanceados en uno solo
final_balanced_df = pd.concat(balanced_dfs, ignore_index=True)

# Resultado final
print(final_balanced_df.shape)


def contar_valores(columna):
  print(final_balanced_df[columna].value_counts())

for i in targets:
  contar_valores(i)


print(final_balanced_df.isna().sum())
print(final_balanced_df.shape)

# Eliminamos los NaN
final_balanced_df = final_balanced_df.dropna(subset=['SMOKE100'])
# Filter out rows where SMOKE100 is 7 or 9
final_balanced_df = final_balanced_df[~final_balanced_df['SMOKE100'].isin([7, 9])]
# Eliminaos donde sea 7 o 9 en DRNKANY5
final_balanced_df = final_balanced_df[~final_balanced_df['DRNKANY5'].isin([7, 9])]
# Eliminmos los 9 de _TOTINDA
final_balanced_df = final_balanced_df[final_balanced_df['_TOTINDA'] != 9]


final_balanced_df = final_balanced_df[~final_balanced_df['DIABETE4'].isin([7, 9])]
final_balanced_df['DIABETE4'] = final_balanced_df['DIABETE4'].replace({3:2, 4:1})

# Calculate the average BMI
bmi_mean = final_balanced_df['bmi_converted'].mean()

# Add a higher-than-average value for diabetic individuals
bmi_above_mean = bmi_mean * 1.1  # 10% above the average

# Replace BMI values based on DIABETE4
final_balanced_df['bmi_converted'] = final_balanced_df.apply(
    lambda row: bmi_above_mean if pd.isna(row['bmi_converted']) and row['DIABETE4'] == 1
    else (bmi_mean if pd.isna(row['bmi_converted']) else row['bmi_converted']),
    axis=1
)


# Pasamos los valores de las columnas a 0 y 1
# Convert target columns to binary format (0 and 1)
target_columns = ['CVDINFR4', 'CVDCRHD4', 'CVDSTRK3', '_TOTINDA',	'SMOKE100',	'DRNKANY5',	'DIABETE4']

for col in target_columns:
    final_balanced_df[col] = final_balanced_df[col].apply(lambda x: 0 if x == 2 else 1)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib  # Para guardar los modelos


def run_model(col):
    # Paso 1: Dividimos el DF
    X = final_balanced_df.drop(columns=[col])
    y = final_balanced_df[col]

    # Paso 2: División de Train y Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Paso 3: Escalar los datos (opcional para modelos sensibles al escalado)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Identificar el nombre del target
    if col == 'CVDINFR4':
        col_name = 'Myocardial infarction'
    elif col == 'CVDCRHD4':
        col_name = 'Angina or coronary heart disease'
    elif col == 'CVDSTRK3':
        col_name = 'Stroke'
    print(f'Resultados para predecir {col_name}')

    # Paso 4: Entrenamos los modelos
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

    # XGBoost
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))

    # Devolver los modelos, características y etiquetas ## FUNCION MODIFICADA ACA PARA DEVOLVER LOS MODELOS DE MANERA DE USARLOS DESPUES
    return lr, rf, xgb, X, y



target_columns = ['CVDINFR4', 'CVDCRHD4', 'CVDSTRK3']

for col in target_columns:
  run_model(col)

# Llamamos a los modelos de la funcion y extraemos sus variables mas importantes
for col in target_columns:
    lr, rf, xgb, X, y = run_model(col)  # Capturamos los modelos y datos

    # Variables más importantes para Random Forest
    feature_importances_rf = rf.feature_importances_
    sorted_idx_rf = feature_importances_rf.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(X.columns[sorted_idx_rf], feature_importances_rf[sorted_idx_rf])
    plt.xlabel("Importancia")
    plt.title(f"Importancia de Características - Random Forest ({col})")
    plt.tight_layout()
    plt.show()

    # Variables más importantes para XGBoost
    feature_importances_xgb = xgb.feature_importances_
    sorted_idx_xgb = feature_importances_xgb.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(X.columns[sorted_idx_xgb], feature_importances_xgb[sorted_idx_xgb])
    plt.xlabel("Importancia")
    plt.title(f"Importancia de Características - XGBoost ({col})")
    plt.tight_layout()
    plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib  # Para guardar los modelos


def run_model(col):
    # Paso 1: Dividimos el DF
    X = final_balanced_df.drop(columns=[col])
    y = final_balanced_df[col]

    # Paso 2: División de Train y Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Paso 3: Escalar los datos (opcional para modelos sensibles al escalado)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if col == 'CVDINFR4':
        col_name = 'Myocardial infarction'
    elif col == 'CVDCRHD4':
        col_name = 'Angina or coronary heart disease'
    elif col == 'CVDSTRK3':
        col_name = 'Stroke'
    print(f'Resultados para predecir {col_name}')

    # Paso 4: Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))

    # Paso 5: Random Forest con Grid Search
    param_grid_rf = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)
    best_rf_params = grid_search_rf.best_params_
    print("Mejores hiperparámetros para Random Forest:", best_rf_params)

    # Reentrenar Random Forest con los mejores parámetros
    rf_optimized = RandomForestClassifier(**best_rf_params, random_state=42)
    rf_optimized.fit(X_train, y_train)
    rf_pred = rf_optimized.predict(X_test)
    print("Random Forest Optimized Accuracy:", accuracy_score(y_test, rf_pred))

    # Paso 6: XGBoost con Random Search
    from scipy.stats import randint
    param_dist_xgb = {
        'n_estimators': randint(50, 150),
        'max_depth': randint(3, 10),
        'learning_rate': [0.01, 0.1, 0.2]
    }
    random_search_xgb = RandomizedSearchCV(XGBClassifier(eval_metric='logloss', random_state=42), param_dist_xgb, n_iter=10, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
    random_search_xgb.fit(X_train, y_train)
    best_xgb_params = random_search_xgb.best_params_
    print("Mejores hiperparámetros para XGBoost:", best_xgb_params)

    # Reentrenar XGBoost con los mejores parámetros
    xgb_optimized = XGBClassifier(**best_xgb_params, eval_metric='logloss', random_state=42)
    xgb_optimized.fit(X_train, y_train)
    xgb_pred = xgb_optimized.predict(X_test)
    print("XGBoost Optimized Accuracy:", accuracy_score(y_test, xgb_pred))

    # Paso 7: Guardar los modelos optimizados
    joblib.dump(lr, f"{col}_lr_model.pkl")
    joblib.dump(rf_optimized, f"{col}_rf_model.pkl")
    joblib.dump(xgb_optimized, f"{col}_xgb_model.pkl")
    print(f"Modelos para {col_name} guardados exitosamente.")

    # Devolver los modelos optimizados
    return lr, rf_optimized, xgb_optimized, X, y


# Ejecutar la función para cada target
for col in target_columns:
    lr_model, rf_model, xgb_model, X, y = run_model(col)

    # Variables más importantes para Random Forest
    feature_importances_rf = rf_model.feature_importances_
    sorted_idx_rf = feature_importances_rf.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(X.columns[sorted_idx_rf], feature_importances_rf[sorted_idx_rf])
    plt.xlabel("Importancia")
    plt.title(f"Importancia de Características - Random Forest ({col})")
    plt.tight_layout()
    plt.show()

    # Variables más importantes para XGBoost
    feature_importances_xgb = xgb_model.feature_importances_
    sorted_idx_xgb = feature_importances_xgb.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(X.columns[sorted_idx_xgb], feature_importances_xgb[sorted_idx_xgb])
    plt.xlabel("Importancia")
    plt.title(f"Importancia de Características - XGBoost ({col})")
    plt.tight_layout()
    plt.show()


from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

def run_stacking_model(col, X_train, X_test, y_train, y_test, lr_model, rf_optimized, xgb_optimized):
    print(f"Running stacking model for target: {col}")

    # Crear el modelo de Stacking utilizando los modelos previamente optimizados
    estimators = [
        ('rf', rf_optimized),   # Random Forest Optimizado
        ('xgb', xgb_optimized)  # XGBoost Optimizado
    ]

    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42)
    )

    # Entrenar el modelo de Stacking
    stacking_model.fit(X_train, y_train)

    # Evaluar el modelo de Stacking
    stacking_pred = stacking_model.predict(X_test)
    stacking_pred_proba = stacking_model.predict_proba(X_test)[:, 1]

    print("Stacking Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, stacking_pred)}")
    print(f"AUC-ROC: {roc_auc_score(y_test, stacking_pred_proba)}")
    print("Classification Report:")
    print(classification_report(y_test, stacking_pred))

    # Guardar el modelo de Stacking
    joblib.dump(stacking_model, f"{col}_stacking_model.pkl")
    print(f"Stacking model for {col} saved successfully.")

    # Devolver el modelo de Stacking
    return stacking_model


# Para cada target, entrena y guarda el modelo de Stacking
for col in target_columns:
    # Obtener los modelos optimizados y los datos
    lr_model, rf, xgb, X, y = run_model(col)

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Crear y evaluar el modelo de Stacking
    stacking_model = run_stacking_model(col, X_train, X_test, y_train, y_test, lr_model, rf, xgb)


import shap
from shap import kmeans

# Seleccionar K muestras representativas
K = 10  # ajustar este número según el tamaño de X_train (entre 1 y 100, mas chico K porque train es grande es igual a Menor tiempo de computo)
background = kmeans(X_train, K)

# Crear el explicador usando el conjunto reducido
explainer_stack = shap.KernelExplainer(stacking_model.predict_proba, background)

# Calcular los valores SHAP para los datos de prueba
shap_values_stack = explainer_stack.shap_values(X_test)

# Gráfico resumen
shap.summary_plot(shap_values_stack, X_test)


# Gráfico resumen
shap.summary_plot(shap_values_stack, X_test)

# Seleccionar el índice de la observación
observation_index = 2  # una "observación" se refiere a una fila específica de tu conjunto de datos de prueba (X_test): 2 seria la primera fila de paciente con toda su informacion, en este caso el primer Kluster.

# Extraer los valores SHAP para la observación específica
shap_values_observation = shap_values_stack[1][observation_index]  # Usa [1] para la clase positiva (binario)

# Extraer el valor base (expected_value)
base_value = explainer_stack.expected_value[1]  # Usar [1] para la clase positiva

# Crear el Waterfall Plot
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_observation,        # Valores SHAP para la observación
        base_values=base_value,               # Valor base (esperado)
        data=X_test.iloc[observation_index]   # Valores de las características para la observación
    )
)


print(type(shap_values_stack))  # Debe ser una lista o un array
print(len(shap_values_stack))   # Número de clases (para clasificación)
print(len(X_test))

print(shap_values_stack.shape)

shap_values_class = shap_values_stack[1]  # Valores SHAP para la clase positiva
print(shap_values_class.shape)  # Esto debe ser (n_samples, n_features)


shap_values_class = shap_values_stack[:, :, 1]  # Selecciona la clase positiva
print(shap_values_class.shape)  # Esto debe ser (101710, 7)


assert shap_values_class.shape[0] == X_test.shape[0], "Las filas de shap_values y X_test no coinciden"
assert shap_values_class.shape[1] == X_test.shape[1], "Las columnas de shap_values y X_test no coinciden"


shap.dependence_plot(
    "_TOTINDA",
    shap_values_class,  # Valores SHAP para la clase positiva
    X_test,  # Conjunto de características
    interaction_index="bmi_converted"  # Característica secundaria para analizar
)

