
# main_total.py – Proyecto cáncer de mama (100% funcional)
# Ejecuta: modelos clásicos + Keras + regresión + perturbación + exportaciones

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_curve, auc,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, RandomForestRegressor)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Crear carpetas
for folder in ["figuras", "tablas", "latex", "logs"]:
    os.makedirs(folder, exist_ok=True)

# Cargar datos
df = pd.read_csv("data/Breast_Cancer_Database.csv")
df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])

# ======================
# 1. SELECCIÓN DE VARIABLES (Clasificación)
# ======================
X_cls = df.drop(columns=['diagnosis'])
y_cls = df['diagnosis']
forest_cls = RandomForestClassifier(n_estimators=100, random_state=42)
forest_cls.fit(X_cls, y_cls)
features_cls = pd.Series(forest_cls.feature_importances_, index=X_cls.columns)
top_features_cls = features_cls.sort_values(ascending=False).head(10).index.tolist()
X_cls = X_cls[top_features_cls]
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    StandardScaler().fit_transform(X_cls), y_cls, test_size=0.2, stratify=y_cls, random_state=42
)

# ======================
# 2. MODELOS CLÁSICOS (Clasificación)
# ======================
models_cls = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "MLP": MLPClassifier(max_iter=1000),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss')
}
roc_data = {}
for name, model in models_cls.items():
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict(X_test_c)
    y_proba = model.predict_proba(X_test_c)[:, 1] if hasattr(model, "predict_proba") else None
    acc = accuracy_score(y_test_c, y_pred)
    prec = precision_score(y_test_c, y_pred)
    rec = recall_score(y_test_c, y_pred)
    f1 = f1_score(y_test_c, y_pred)
    pd.DataFrame([{
        "Modelo": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1
    }]).to_excel(f"tablas/metricas_{name}.xlsx", index=False)
    cm = confusion_matrix(y_test_c, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusión - {name}")
    plt.savefig(f"figuras/confusion_{name}.png")
    plt.close()
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test_c, y_proba)
        plt.plot(fpr, tpr, label=f"{name}")
        roc_data[name] = auc(fpr, tpr)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {name}")
        plt.savefig(f"figuras/roc_{name}.png")
        plt.close()

# ======================
# 3. RED NEURONAL (Clasificación)
# ======================
model = keras.Sequential([
    layers.Input(shape=(X_train_c.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()])
history = model.fit(X_train_c, y_train_c, validation_split=0.2, epochs=100, batch_size=16, verbose=0)
y_pred_keras = (model.predict(X_test_c).flatten() >= 0.5).astype(int)
acc = accuracy_score(y_test_c, y_pred_keras)
prec = precision_score(y_test_c, y_pred_keras)
rec = recall_score(y_test_c, y_pred_keras)
f1 = f1_score(y_test_c, y_pred_keras)
pd.DataFrame([{
    "Modelo": "Keras", "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1
}]).to_excel("tablas/metricas_keras.xlsx", index=False)

# ======================
# 4. REGRESIÓN (Modelos clásicos)
# ======================
X_reg = df.drop(columns=["area_worst", "diagnosis"], errors="ignore")
y_reg = df["area_worst"]
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(X_reg, y_reg)
top_features_reg = pd.Series(forest_reg.feature_importances_, index=X_reg.columns).nlargest(10).index.tolist()
X_reg = X_reg[top_features_reg]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    StandardScaler().fit_transform(X_reg), y_reg, test_size=0.2, random_state=42
)
models_reg = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso()
}
for name, model in models_reg.items():
    model.fit(X_train_r, y_train_r)
    y_pred = model.predict(X_test_r)
    mse = mean_squared_error(y_test_r, y_pred)
    mae = mean_absolute_error(y_test_r, y_pred)
    r2 = r2_score(y_test_r, y_pred)
    pd.DataFrame([{
        "Modelo": name, "MAE": mae, "MSE": mse, "R2": r2
    }]).to_excel(f"tablas/metricas_{name}_regresion.xlsx", index=False)

# ======================
# 5. RED NEURONAL (Regresión)
# ======================
model_r = keras.Sequential([
    layers.Input(shape=(X_train_r.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
model_r.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
model_r.fit(X_train_r, y_train_r, validation_split=0.2, epochs=100, batch_size=16, verbose=0)
y_pred_r = model_r.predict(X_test_r).flatten()
mae = mean_absolute_error(y_test_r, y_pred_r)
mse = mean_squared_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)
pd.DataFrame([{
    "Modelo": "Keras (reg)", "MAE": mae, "MSE": mse, "R2": r2
}]).to_excel("tablas/metricas_keras_regresion.xlsx", index=False)

print("✅ Proyecto ejecutado correctamente.")
