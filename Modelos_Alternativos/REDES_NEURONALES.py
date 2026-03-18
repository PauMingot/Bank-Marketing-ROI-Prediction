# -*- coding: utf-8 -*-
"""
Created on Fri May  2 12:38:42 2025

@author: media
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, cohen_kappa_score, classification_report
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Cargar y preparar datos ---
df = pd.read_csv('bank.csv')
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
y = df['deposit'].map({'no': 0, 'yes': 1})
X = df.drop(columns=['deposit', 'duration'])
X = pd.get_dummies(X, drop_first=True)

# --- Dividir ---
pct_val = 0.2
pct_test = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=pct_test, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=pct_val / (1 - pct_test), random_state=42)

# --- Escalar ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# --- Modelo de red neuronal ---
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- Entrenamiento con EarlyStopping ---
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)

# --- Visualizar pérdida ---
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Pérdida durante el entrenamiento')
plt.legend()
plt.show()

# --- Evaluación general ---
y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_prob))
print("Kappa:", cohen_kappa_score(y_test, y_pred))

# --- Confusión y errores ---
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
FPR = fp / (fp + tn)
FNR = fn / (fn + tp)
print(f"Falsos Positivos (FPR): {FPR:.4f}")
print(f"Falsos Negativos (FNR): {FNR:.4f}")

# --- Análisis por thresholds ---
thresholds = [0.3, 0.32, 0.35, 0.37, 0.4, 0.42, 0.43, 0.45, 0.48, 0.5]
print("\n--- Evaluación por Thresholds ---")
for threshold in thresholds:
    y_pred_adj = (y_pred_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_adj).ravel()
    recall = recall_score(y_test, y_pred_adj)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    print(f"Threshold: {threshold:.2f} | Recall: {recall:.4f} | FPR: {fpr:.4f} | FNR: {fnr:.4f}")

# --- Análisis final con el mejor threshold (0.35 por ejemplo) ---
print("\n--- Análisis final con threshold 0.35 ---")
threshold = 0.35
y_final = (y_pred_prob >= threshold).astype(int)

accuracy = accuracy_score(y_test, y_final)
precision = precision_score(y_test, y_final)
recall = recall_score(y_test, y_final)
f1 = f1_score(y_test, y_final)
auc = roc_auc_score(y_test, y_pred_prob)
kappa = cohen_kappa_score(y_test, y_final)
tn, fp, fn, tp = confusion_matrix(y_test, y_final).ravel()
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Kappa: {kappa:.4f}")
print(f"FPR: {fpr:.4f}")
print(f"FNR: {fnr:.4f}")














df['deposit_bin'] = df['deposit'].map({'yes': 1, 'no': 0})


corr = df.corr(numeric_only=True)['deposit_bin'].sort_values(ascending=False)
print(corr)




for col in df.columns:
    if col != 'deposit':
        sns.boxplot(x='deposit', y=col, data=df)
        plt.title(f'Relación entre {col} y deposit')
        plt.show()


cat_vars = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']  # adapta según tu dataset

# 4. Mostrar tasa de aceptación por grupo
for col in cat_vars:
    print(f"\n>>> {col.upper()}")
    print(df.groupby(col)['deposit_bin'].mean().sort_values(ascending=False))

