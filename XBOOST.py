# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 10:00:22 2025

@author: media
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, roc_curve, auc,
    cohen_kappa_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, plot_importance

# -------------------------------
# CARGA Y PREPROCESAMIENTO
# -------------------------------
df = pd.read_csv('bank.csv')
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
y = df['deposit'].map({'no': 1, 'yes': 0})
X = df.drop(columns=['deposit', 'duration'])
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)


# -------------------------------
# OPTIMIZACIÓN DE HIPERPARÁMETROS
# -------------------------------
param_grid = {
    'n_estimators': [200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
grid_search = GridSearchCV(xgb_clf, param_grid, scoring='roc_auc', cv=3, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Mejores hiperparámetros:")
print(grid_search.best_params_)


# -------------------------------
# EVALUACIÓN
# -------------------------------
def evaluate_model(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Métricas individuales
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    # Tasa de falsos positivos y falsos negativos
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    # Mostrar métricas
    print("\nMétricas del Modelo (Clasificación):")
    print(f"Verdaderos Positivos (TP): {tp}")
    print(f"Falsos Negativos (FN): {fn}")
    print(f"Falsos Positivos (FP): {fp}")
    print(f"Verdaderos Negativos (TN): {tn}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Tasa de Falsos Positivos (FPR): {fpr:.4f}")
    print(f"Tasa de Falsos Negativos (FNR): {fnr:.4f}")


    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()



    return y_proba

# Evaluar el modelo
y_proba = evaluate_model(best_model, X_test, y_test)


# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
plot_importance(best_model, importance_type='gain', max_num_features=15, height=0.5)
plt.title("Importancia de variables (gain)")
plt.tight_layout()
plt.show()


def threshold_analysis(y_true, y_proba, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)

    results = []
    for thr in thresholds:
        y_pred_thr = (y_proba >= thr).astype(int)
        recall = recall_score(y_true, y_pred_thr)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thr).ravel()
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        precision = precision_score(y_true, y_pred_thr)
        accuracy = accuracy_score(y_true, y_pred_thr)
        results.append({
            'Threshold': round(thr, 2),
            'Recall': round(recall, 4),
            'Precision': round(precision, 4),
            'FPR': round(fpr, 4),
            'FNR': round(fnr, 4),
            'Accuracy': round(accuracy, 4)
        })

    results_df = pd.DataFrame(results)
    print(results_df)

    # Graficar métricas por threshold
    plt.figure(figsize=(10, 6))
    for metric in ['Recall', 'Precision', 'FPR', 'FNR', 'Accuracy']:
        plt.plot(results_df['Threshold'], results_df[metric], label=metric)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Evaluación de métricas en distintos umbrales')
    plt.legend()
    plt.grid(True)
    plt.show()

    return results_df

# Llamada a la función
threshold_analysis(y_test, y_proba)


threshold = 0.35
y_pred_threshold = (y_proba >= threshold).astype(int)

# Matriz de confusión
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()

# Métricas de evaluación
accuracy = accuracy_score(y_test, y_pred_threshold)
precision = precision_score(y_test, y_pred_threshold)
recall = recall_score(y_test, y_pred_threshold)
f1 = f1_score(y_test, y_pred_threshold)
auc = roc_auc_score(y_test, y_pred_threshold)
kappa = cohen_kappa_score(y_test, y_pred_threshold)

# Tasas de error
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)

print("🔍 Análisis Final del Mejor Modelo con Threshold = 0.35")
print("-" * 50)
print(f"Threshold aplicado: {threshold}")
print(f"Verdaderos Positivos (TP): {tp}")
print(f"Falsos Negativos (FN): {fn}")
print(f"Falsos Positivos (FP): {fp}")
print(f"Verdaderos Negativos (TN): {tn}")
print()
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensibilidad): {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print()
print(f"Tasa de Falsos Positivos (FPR): {fpr:.4f}")
print(f"Tasa de Falsos Negativos (FNR): {fnr:.4f}")



