import pandas as pd
import numpy as np
import time
import warnings
import os  # <--- Essencial para criar pastas
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

print("DETECÇÃO DE INTRUSÃO - UNSW-NB15 (LightGBM - Exportação)")

# ========================================
# CONFIGURAÇÃO DE SAÍDA
# ========================================
OUTPUT_DIR = './teste7'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"✓ Pasta criada: {OUTPUT_DIR}")
else:
    print(f"✓ Pasta de saída: {OUTPUT_DIR}")

# ========================================
# 1. CARREGAMENTO DOS DADOS
# ========================================
print("\n[1/9] Carregando dados UNSW-NB15...")

try:
    df_train = pd.read_csv('./unsw/UNSW_NB15_training-set.csv')
    df_test = pd.read_csv('./unsw/UNSW_NB15_testing-set.csv')
    print(f"✓ Training set: {df_train.shape[0]:,} amostras")
    print(f"✓ Testing set:  {df_test.shape[0]:,} amostras")
except Exception as e:
    print(f"✗ Erro ao carregar dados: {e}")
    exit()

# ========================================
# 2. LIMPEZA E PREPARAÇÃO
# ========================================
print("\n[2/9] Limpeza e preparação...")

COLS_TO_REMOVE = ['id', 'attack_cat', 'srcip', 'sport', 'dstip', 'dsport']
cols_to_remove_actual = [col for col in COLS_TO_REMOVE if col in df_train.columns]

if 'label' in df_train.columns:
    X_train = df_train.drop(columns=cols_to_remove_actual + ['label'], errors='ignore')
    y_train = df_train['label'].astype(int)
    
    X_test = df_test.drop(columns=cols_to_remove_actual + ['label'], errors='ignore')
    y_test = df_test['label'].astype(int)
else:
    raise ValueError("Coluna 'label' não encontrada!")

# Validação (20% do Treino)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# ========================================
# 3. PRÉ-PROCESSAMENTO
# ========================================
print("\n[3/9] Aplicando pré-processamento...")

categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# 3.1 Label Encoding
if categorical_cols:
    for col in categorical_cols:
        le = LabelEncoder()
        all_values = pd.concat([X_train[col], X_val[col], X_test[col]]).unique()
        le.fit(all_values)
        
        # Transform com tratamento de missing
        X_train[col] = X_train[col].fillna('missing').apply(lambda x: x if x in le.classes_ else 'missing')
        X_val[col] = X_val[col].fillna('missing').apply(lambda x: x if x in le.classes_ else 'missing')
        X_test[col] = X_test[col].fillna('missing').apply(lambda x: x if x in le.classes_ else 'missing')
        
        X_train[col] = le.transform(X_train[col])
        X_val[col] = le.transform(X_val[col])
        X_test[col] = le.transform(X_test[col])

# 3.2 Imputação e Normalização
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X_train_final = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(X_train)), columns=X_train.columns)
X_val_final = pd.DataFrame(scaler.transform(imputer.transform(X_val)), columns=X_val.columns)
X_test_final = pd.DataFrame(scaler.transform(imputer.transform(X_test)), columns=X_test.columns)

# ========================================
# 4. TREINAMENTO (LightGBM)
# ========================================
print("\n[4/9] Treinando LightGBM...")

lgbm_params = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

model = LGBMClassifier(**lgbm_params)

start_train = time.time()
model.fit(
    X_train_final, y_train,
    eval_set=[(X_val_final, y_val)],
    eval_metric='logloss'
)
end_train = time.time()
print(f"✓ Treino concluído em {end_train - start_train:.2f}s")

# ========================================
# 5. AVALIAÇÃO
# ========================================
print("\n[5/9] Avaliando modelo...")

def get_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

# Predições
y_pred_train = model.predict(X_train_final)
y_pred_val = model.predict(X_val_final)

start_test = time.time()
y_pred_test = model.predict(X_test_final)
end_test = time.time()

train_metrics = get_metrics(y_train, y_pred_train)
val_metrics = get_metrics(y_val, y_pred_val)
test_metrics = get_metrics(y_test, y_pred_test)

print(f"\n📊 RESULTADOS TESTE:")
print(f"  Acurácia:  {test_metrics['accuracy']:.4f}")
print(f"  F1-Score:  {test_metrics['f1']:.4f}")

# ========================================
# 6. EXPORTAÇÃO RELATÓRIO TXT
# ========================================
print("\n[6/9] Salvando relatório de texto...")

report_path = os.path.join(OUTPUT_DIR, 'relatorio_unsw_lightgbm.txt')
with open(report_path, 'w') as f:
    f.write("RELATORIO DE PERFORMANCE - UNSW-NB15 (LightGBM)\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Tempo de Treinamento: {end_train - start_train:.4f}s\n")
    f.write(f"Tempo de Inferência:  {end_test - start_test:.4f}s\n\n")
    
    f.write("Métricas de Teste:\n")
    f.write(f"  Acurácia:  {test_metrics['accuracy']:.4f}\n")
    f.write(f"  Precision: {test_metrics['precision']:.4f}\n")
    f.write(f"  Recall:    {test_metrics['recall']:.4f}\n")
    f.write(f"  F1-Score:  {test_metrics['f1']:.4f}\n\n")
    
    gap = train_metrics['accuracy'] - test_metrics['accuracy']
    f.write("Overfitting Check:\n")
    f.write(f"  Gap Treino-Teste: {gap:.4f}\n")
    f.write(f"  Status: {'ALERTA' if gap > 0.05 else 'OK'}\n\n")
    
    f.write("Classification Report Detalhado:\n")
    f.write(classification_report(y_test, y_pred_test, target_names=['Normal', 'Attack']))

print(f"✓ Relatório salvo em: {report_path}")

# ========================================
# 7. EXPORTAÇÃO GRÁFICOS
# ========================================
print("\n[7/9] Gerando e salvando gráficos...")

# 7.1 Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12, 6))
plt.title("Top 20 Features - UNSW-NB15 (LightGBM)", fontsize=14)
plt.bar(range(20), importances[indices], color='orange', edgecolor='red')
plt.xticks(range(20), [X_train_final.columns[i] for i in indices], rotation=45, ha='right')
plt.tight_layout()

feat_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
plt.savefig(feat_path, dpi=300)
plt.close()
print(f"✓ Features salvas em: {feat_path}")

# 7.2 Matriz de Confusão (Validação e Teste)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm_val = confusion_matrix(y_val, y_pred_val)
cm_test = confusion_matrix(y_test, y_pred_test)

ConfusionMatrixDisplay(cm_val, display_labels=['Normal', 'Attack']).plot(ax=axes[0], cmap='Blues')
axes[0].set_title(f'Validação (Acc: {val_metrics["accuracy"]:.4f})')

ConfusionMatrixDisplay(cm_test, display_labels=['Normal', 'Attack']).plot(ax=axes[1], cmap='Oranges')
axes[1].set_title(f'Teste (Acc: {test_metrics["accuracy"]:.4f})')

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrices.png')
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"✓ Matrizes salvas em: {cm_path}")

# 7.3 Comparação de Métricas
metrics_names = ['accuracy', 'precision', 'recall', 'f1']
train_v = [train_metrics[m] for m in metrics_names]
test_v = [test_metrics[m] for m in metrics_names]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, train_v, width, label='Train', color='#2ecc71')
ax.bar(x + width/2, test_v, width, label='Test', color='#e74c3c')

ax.set_ylabel('Scores')
ax.set_title('LightGBM: Comparação Treino vs Teste')
ax.set_xticks(x)
ax.set_xticklabels([m.capitalize() for m in metrics_names])
ax.legend()
ax.set_ylim([0.7, 1.05])

comp_path = os.path.join(OUTPUT_DIR, 'metrics_comparison.png')
plt.savefig(comp_path, dpi=300)
plt.close()
print(f"✓ Comparativo salvo em: {comp_path}")

print("\n[8/9] Processo Finalizado!")