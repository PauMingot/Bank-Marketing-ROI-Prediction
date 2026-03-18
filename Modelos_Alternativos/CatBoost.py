import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, cohen_kappa_score
)
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool

# -------------------------------
# CARGA Y PREPROCESAMIENTO
# -------------------------------

df = pd.read_csv('bank.csv')
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
y = df['deposit'].map({'no': 0, 'yes': 1})
X = df.drop(columns=['deposit', 'duration'])
X = pd.get_dummies(X, drop_first=True)

# Train / Val / Test split
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.20, random_state=42, stratify=y_train_val
)

# Si hubiera columnas categóricas sin dummy, las indicarías así:
# cat_features = np.where(X_train.dtypes != float)[0]
# Pero como ya usamos get_dummies, no las necesitamos:
cat_features = []

# -------------------------------
# ENTRENAMIENTO CATBOOST CON EARLY STOPPING
# -------------------------------

# Creamos pools para que CatBoost maneje bien el eval_set
train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool   = Pool(X_val,   y_val,   cat_features=cat_features)

cat_clf = CatBoostClassifier(
    iterations=1000,            # límite alto
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3,
    border_count=64,
    random_seed=42,
    eval_metric='AUC',
    early_stopping_rounds=50,   # para detener si no mejora
    use_best_model=True,        # revertir al mejor punto
    verbose=100                 # imprime cada 100 iteraciones
)

# Entrenamos con validación interna
cat_clf.fit(
    train_pool,
    eval_set=val_pool
)

# -------------------------------
# EVALUACIÓN EN TEST SET
# -------------------------------

def evaluate_model(model, X_test, y_test, thresh=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= thresh).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    auc       = roc_auc_score(y_test, y_proba)
    kappa     = cohen_kappa_score(y_test, y_pred)
    fpr       = fp / (fp + tn)
    fnr       = fn / (fn + tp)

    print("\n=== Métricas en TEST (threshold={:.2f}) ===".format(thresh))
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Kappa:     {kappa:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"FNR:       {fnr:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    return y_proba

y_proba = evaluate_model(cat_clf, X_test, y_test, thresh=0.5)

# -------------------------------
# THRESHOLD ANALYSIS
# -------------------------------

def threshold_analysis(y_true, y_proba, thresholds):
    results = []
    for thr in thresholds:
        y_pred_thr = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thr).ravel()
        recall = recall_score(y_true, y_pred_thr)
        precision = precision_score(y_true, y_pred_thr)
        fpr   = fp / (fp + tn)
        fnr   = fn / (fn + tp)
        acc   = accuracy_score(y_true, y_pred_thr)
        results.append({
            'Threshold': thr,
            'Recall':    recall,
            'Precision': precision,
            'FPR':       fpr,
            'FNR':       fnr,
            'Accuracy':  acc
        })
    df_thr = pd.DataFrame(results)
    print(df_thr)
    plt.figure(figsize=(8,5))
    for m in ['Recall','Precision','FPR','FNR','Accuracy']:
        plt.plot(df_thr['Threshold'], df_thr[m], label=m)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend(); plt.grid(True); plt.show()
    return df_thr

thr_list = [0.30,0.32,0.35,0.37,0.40,0.42,0.43,0.45,0.48,0.50]
threshold_analysis(y_test, y_proba, thr_list)
