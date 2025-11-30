import pandas as pd
import numpy as np
import time
import warnings
import os  
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

print("DETECÇÃO DE INTRUSÃO - ToN-IoT NETWORK DATASET (Com Exportação)")

# ========================================
# CONFIGURAÇÃO DE SAÍDA
# ========================================
OUTPUT_DIR = './teste5'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"✓ Pasta criada: {OUTPUT_DIR}")
else:
    print(f"✓ Pasta de saída: {OUTPUT_DIR}")

# ========================================
# 1. CARREGAMENTO DOS DADOS
# ========================================
print("\n[1/9] Carregando dados ToN-IoT...")

FILE_PATH = './ton-iot/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv'

try:
    df = pd.read_csv(FILE_PATH)
    print(f"✓ Dataset carregado: {df.shape[0]:,} amostras, {df.shape[1]} colunas")
except Exception as e:
    print(f"✗ Erro ao carregar dados: {e}")
    exit()

# ========================================
# 2. LIMPEZA E PREPARAÇÃO
# ========================================
print("\n[2/9] Limpeza e preparação dos dados...")

# Colunas a remover
COLS_TO_REMOVE = [
    'ts', 'timestamp', 'date', 'time',
    'src_ip', 'src_port',
    'dst_ip', 'dst_port',
    'type',
    'id'
]

cols_to_drop = [col for col in COLS_TO_REMOVE if col in df.columns]
X = df.drop(columns=cols_to_drop + ['label'], errors='ignore')
y = df['label'].astype(int)

# ========================================
# 3. DIVISÃO DOS DADOS
# ========================================
print("\n[3/9] Dividindo em Treino, Validação e Teste...")

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print(f"✓ Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")
del df, X_temp, y_temp

# ========================================
# 4. PRÉ-PROCESSAMENTO
# ========================================
print("\n[4/9] Aplicando pré-processamento...")

categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# 4.1 Label Encoding
if categorical_cols:
    for col in categorical_cols:
        le = LabelEncoder()
        all_values = pd.concat([X_train[col], X_val[col], X_test[col]]).astype(str).unique()
        le.fit(all_values)
        
        X_train[col] = le.transform(X_train[col].astype(str))
        X_val[col] = le.transform(X_val[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

# 4.2 Imputação
imputer = SimpleImputer(strategy='mean')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# 4.3 Normalização
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# ========================================
# 5. TREINAMENTO
# ========================================
print("\n[5/9] Treinando XGBoost Otimizado...")

xgb_params = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 1.0,
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1
}

xgb_model = XGBClassifier(**xgb_params)
start_train = time.time()
xgb_model.fit(X_train, y_train)
end_train = time.time()
print(f"✓ Treino concluído em {end_train - start_train:.2f}s")

# ========================================
# 6. AVALIAÇÃO
# ========================================
print("\n[6/9] Avaliando modelo...")

start_test = time.time()
y_pred_test = xgb_model.predict(X_test)
end_test = time.time()

acc = accuracy_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test, zero_division=0)

print(f"\n📊 RESULTADOS FINAIS:")
print(f"  Acurácia:  {acc:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# ========================================
# 7. EXPORTAÇÃO DOS RESULTADOS (TXT)
# ========================================
print("\n[7/9] Salvando relatório em texto...")
report_path = os.path.join(OUTPUT_DIR, 'relatorio_metricas.txt')
with open(report_path, 'w') as f:
    f.write("RELATORIO DE PERFORMANCE - XGBoost (ToN-IoT)\n")
    f.write("="*50 + "\n")
    f.write(f"Acuracia: {acc:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred_test))

print(f"✓ Relatório salvo em: {report_path}")

# ========================================
# 8. EXPORTAÇÃO DOS GRÁFICOS
# ========================================
print("\n[8/9] Gerando e salvando gráficos...")

# 8.1 Top Features
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1][:15]

plt.figure(figsize=(12, 6))
plt.title("Top 15 Features - ToN-IoT Network", fontsize=14)
plt.bar(range(len(indices)), importances[indices], align="center", color='purple')
plt.xticks(range(len(indices)), [X_train.columns[i] for i in indices], rotation=45, ha='right')
plt.tight_layout()

# SALVAR
feat_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
plt.savefig(feat_path, dpi=300)
print(f"✓ Gráfico Features salvo em: {feat_path}")
plt.close() # Fecha para liberar memória

# 8.2 Matriz de Confusão
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Attack'])
disp.plot(ax=ax, cmap='Purples')
plt.title("Matriz de Confusão (Teste)", fontsize=14)

# SALVAR
cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=300)
print(f"✓ Matriz Confusão salva em: {cm_path}")
plt.close()

print("\n[9/9] Processo Finalizado!")