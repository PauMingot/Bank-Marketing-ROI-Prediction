import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
from sklearn.metrics import roc_curve,confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, mean_squared_error, roc_auc_score, cohen_kappa_score
from sklearn.linear_model import LogisticRegression
from patsy import dmatrices
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import os

# Cargar datos
df = pd.read_csv('bank.csv')


# 2. Limpiar columnas
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

# 3. Definir X e y MANUAL
y = df['deposit']
X = df.drop(columns=['deposit', 'duration'])   
X = pd.get_dummies(X, drop_first=True)
y = y.map({'no': 0, 'yes': 1})


pct_test = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=pct_test, random_state=42)

from lineartree import LinearTreeClassifier


features_r = [c for c in X_train.select_dtypes(include=np.number).columns if c not in ['target']]

df.head()



from sklearn.ensemble import RandomForestClassifier
import multiprocessing

results = []
for n in [1,10,25,50,100,250]:
    
    rf = RandomForestClassifier(
        n_estimators=n,
        max_depth=40,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=5,
        bootstrap=True,
        oob_score=False,
        n_jobs=multiprocessing.cpu_count()
    )
    rf.fit(X_train, y_train)

    y_train_hat = rf.predict(X_train)
    y_test_hat = rf.predict(X_test)

    # Model ROC test
    acc_train = accuracy_score(y_train, y_train_hat)
    acc_test = accuracy_score(y_test, y_test_hat)
    
    results.append((n,acc_train,acc_test))
    
results = pd.DataFrame(results, columns=['n_trees','acc_train','acc_test'])
results


plt.plot(results.n_trees, results.acc_train, label='Train acc')
plt.plot(results.n_trees, results.acc_test, label='Test acc')
plt.legend()
plt.title("Evolución de la Accuracy")
plt.xlabel("Número de Árboles")
plt.ylabel("Accuracy")
plt.show()

model_c = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=3,
    bootstrap=True,
    oob_score=False,
    n_jobs=multiprocessing.cpu_count()
)

model_c.fit(X_train, y_train)

features_r = X.columns  # Ahora `features_r` será igual a las columnas de X

# Crear el DataFrame de importancias
importance = pd.DataFrame({
    'feature': features_r,
    'importance': model_c.feature_importances_
})

# Ordenar por importancia descendente
importance.sort_values('importance', ascending=False, inplace=True)

# Mostrar las importancias de las características
print(importance.head())


y_pred = model_c.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# Calcular recall
recall = recall_score(y_test, y_pred)

# Calcular precision
precision = precision_score(y_test, y_pred)

# Calcular F1-score
f1 = f1_score(y_test, y_pred)

# Matriz de confusión
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Calcular Kappa
kappa = cohen_kappa_score(y_test, y_pred)

# Calcular tasas de error
error_rate = (fp + fn) / (tn + fp + fn + tp)
fpr = fp / (fp + tn)  # False Positive Rate
fnr = fn / (fn + tp)  # False Negative Rate

# Mostrar los resultados
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Kappa: {kappa:.4f}")
print(f"Error Rate: {error_rate:.4f}")
print(f"False Positive Rate: {fpr:.4f}")
print(f"False Negative Rate: {fnr:.4f}")

# Obtener las probabilidades de clase 1 (positivo)
y_proba = model_c.predict_proba(X_test)[:, 1]

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Calcular el área bajo la curva (AUC)
auc = roc_auc_score(y_test, y_proba)

# Trazar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Línea de aleatoriedad
plt.title('Curva ROC')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()



y_proba = model_c.predict_proba(X_test)[:, 1]
thresholds = [0.3, 0.32, 0.35, 0.37, 0.4, 0.42, 0.43, 0.45, 0.48, 0.5]  # Probar varios umbrales
for threshold in thresholds:
    y_pred_adjusted = (y_proba >= threshold).astype(int)
    recall = recall_score(y_test, y_pred_adjusted)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_adjusted).ravel()
    fpr = fp / (fp + tn)
    fns = fn / (fn + tp)
    
    print(f"Umbral: {threshold}")
    print(f"Recall: {recall:.4f}")
    print(f"Tasa de Falsos Negativos (FNR): {fns:.4f}")
    print(f"Tasa de Falsos Positivos (FPR): {fpr:.4f}")
    print("-" * 40)


y_proba = model_c.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.43).astype(int)

accuracy = accuracy_score(y_test, y_pred)

# Calcular recall
recall = recall_score(y_test, y_pred)

# Calcular precision
precision = precision_score(y_test, y_pred)

# Calcular F1-score
f1 = f1_score(y_test, y_pred)

# Matriz de confusión
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Calcular Kappa
kappa = cohen_kappa_score(y_test, y_pred)

# Calcular tasas de error
error_rate = (fp + fn) / (tn + fp + fn + tp)
fpr = fp / (fp + tn)  # False Positive Rate
fnr = fn / (fn + tp)  # False Negative Rate

# Mostrar los resultados
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Kappa: {kappa:.4f}")
print(f"Error Rate: {error_rate:.4f}")
print(f"False Positive Rate: {fpr:.4f}")
print(f"False Negative Rate: {fnr:.4f}")






from sklearn.model_selection import GridSearchCV

# Definir el espacio de búsqueda de los hiperparámetros
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Inicializar el modelo y GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(),
                           param_grid=param_grid,
                           cv=5, 
                           scoring='accuracy',
                           n_jobs=multiprocessing.cpu_count())

# Ajustar el modelo a los datos
grid_search.fit(X_train, y_train)

# Mostrar los mejores parámetros
print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
print(f"Mejor puntuación de validación cruzada: {grid_search.best_score_:.4f}")




# Configuración de los mejores parámetros encontrados
rf_model = RandomForestClassifier(
    n_estimators=150,        # Número de árboles
    max_depth=10,            # Profundidad máxima
    min_samples_split=10,    # Mínimo número de muestras para dividir un nodo
    min_samples_leaf=1,      # Mínimo número de muestras en una hoja
    max_features='sqrt',     # Número de características a considerar
    n_jobs=-1                # Usar todos los núcleos disponibles
)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Entrenar el modelo
rf_model.fit(X_train, y_train)

# Realizar predicciones
y_proba = rf_model.predict_proba(X_test)[:, 1]

# Definir una gama de umbrales
thresholds = np.arange(0.0, 1.1, 0.01)

# Listas para almacenar los resultados de cada umbral
precision_list = []
recall_list = []
f1_list = []
accuracy_list = []

# Evaluar el modelo con diferentes umbrales
for threshold in thresholds:
    # Clasificación basada en el umbral
    y_pred_threshold = (y_proba >= threshold).astype(int)
    
    # Calcular las métricas
    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    f1 = f1_score(y_test, y_pred_threshold)
    accuracy = accuracy_score(y_test, y_pred_threshold)
    
    # Almacenar los resultados
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    accuracy_list.append(accuracy)

# Convertir las listas a arrays
precision_list = np.array(precision_list)
recall_list = np.array(recall_list)
f1_list = np.array(f1_list)
accuracy_list = np.array(accuracy_list)

# Encontrar el mejor umbral basado en F1 Score (puedes cambiar esto según el objetivo)
best_threshold_idx = np.argmax(f1_list)
best_threshold = thresholds[best_threshold_idx]
best_f1 = f1_list[best_threshold_idx]

# Mostrar el mejor umbral y su rendimiento
print(f"Mejor umbral: {best_threshold:.2f}")
print(f"F1 Score con el mejor umbral: {best_f1:.4f}")
print(f"Precision: {precision_list[best_threshold_idx]:.4f}")
print(f"Recall: {recall_list[best_threshold_idx]:.4f}")
print(f"Accuracy: {accuracy_list[best_threshold_idx]:.4f}")

# Graficar las métricas para cada umbral
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(thresholds, precision_list, label='Precision', color='blue')
plt.xlabel('Umbral')
plt.ylabel('Precision')
plt.title('Precision vs. Umbral')

plt.subplot(2, 2, 2)
plt.plot(thresholds, recall_list, label='Recall', color='green')
plt.xlabel('Umbral')
plt.ylabel('Recall')
plt.title('Recall vs. Umbral')

plt.subplot(2, 2, 3)
plt.plot(thresholds, f1_list, label='F1 Score', color='red')
plt.xlabel('Umbral')
plt.ylabel('F1 Score')
plt.title('F1 Score vs. Umbral')

plt.subplot(2, 2, 4)
plt.plot(thresholds, accuracy_list, label='Accuracy', color='purple')
plt.xlabel('Umbral')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Umbral')

plt.tight_layout()
plt.show()




y_pred_threshold_39 = (y_proba >= 0.4).astype(int)

# Calcular las métricas
precision = precision_score(y_test, y_pred_threshold_39)
recall = recall_score(y_test, y_pred_threshold_39)
f1 = f1_score(y_test, y_pred_threshold_39)
accuracy = accuracy_score(y_test, y_pred_threshold_39)

# Mostrar las métricas de forma separada
print(f"\nMétricas para el umbral de 0.39:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred_threshold_39)
tn, fp, fn, tp = cm.ravel()

print("\nMatriz de Confusión:")
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

# Calcular tasas de error
error_rate = (fp + fn) / (tn + fp + fn + tp)
fpr = fp / (fp + tn)  # False Positive Rate
fnr = fn / (fn + tp)  # False Negative Rate

print("\nTasas de Error:")
print(f"Error Rate: {error_rate:.4f}")
print(f"False Positive Rate (FPR): {fpr:.4f}")
print(f"False Negative Rate (FNR): {fnr:.4f}")



