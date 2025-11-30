import pandas as pd
import numpy as np
import time
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix, 
    classification_report, 
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

print("="*80)
print("SISTEMA DE DETECÇÃO DE AMEAÇAS DE REDE - VERSÃO CORRIGIDA")
print("="*80)

# ========================================
# 1. CARREGAMENTO DOS DADOS
# ========================================
print("\n[1/10] Carregando dados...")
try:
    df = pd.read_csv('./ton-iot/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv')
    print(f"✓ Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")
    print(f"✓ Distribuição das classes:")
    print(df['label'].value_counts(normalize=True))
except Exception as e:
    print(f"✗ Erro ao carregar dados: {e}")
    raise

# ========================================
# 2. ANÁLISE INICIAL
# ========================================
print("\n[2/10] Análise inicial do dataset...")
print(f"✓ Colunas categóricas: {df.select_dtypes(include=['object']).columns.tolist()}")
print(f"✓ Valores ausentes: {df.isnull().sum().sum()}")

# ========================================
# 3. SEPARAÇÃO DE FEATURES E TARGET
# ========================================
print("\n[3/10] Separando features e target...")

# Colunas a remover ANTES de qualquer processamento
COLS_TO_REMOVE = [
    # Target e tipo
    'label', 'type',
    
    # IPs (vazam informação!)
    'src_ip', 'dst_ip',
    
    # Colunas de assinatura (comportamento específico)
    'dns_query', 'http_uri', 'ssl_subject', 'ssl_issuer', 
    'http_user_agent', 'weird_name', 'weird_addl', 'weird_notice', 
    'http_orig_mime_types', 'http_resp_mime_types',
    
    # Timestamps (se existirem)
    'ts', 'timestamp'
]

print(f"✓ Removendo colunas problemáticas: {[c for c in COLS_TO_REMOVE if c in df.columns]}")

X = df.drop(columns=[col for col in COLS_TO_REMOVE if col in df.columns], errors='ignore')
y = df['label'].astype(int)

print(f"✓ Features finais: {X.shape[1]} colunas")
print(f"✓ Colunas restantes: {X.columns.tolist()[:10]}... (mostrando primeiras 10)")

# ========================================
# 4. TRAIN/VAL/TEST SPLIT
# ========================================
print("\n[4/10] Dividindo dados (60% train, 20% val, 20% test)...")

# Primeiro split: 80% treino+val, 20% teste
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# Segundo split: 75% do temp para treino (60% total), 25% para val (20% total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, 
    test_size=0.25, 
    random_state=42, 
    stratify=y_temp
)

print(f"✓ Train: {X_train.shape[0]:,} amostras ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Val:   {X_val.shape[0]:,} amostras ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Test:  {X_test.shape[0]:,} amostras ({X_test.shape[0]/len(X)*100:.1f}%)")

# Verificar distribuição de classes
print(f"\n✓ Distribuição Train: {y_train.value_counts(normalize=True).to_dict()}")

# ========================================
# 5. PRÉ-PROCESSAMENTO
# ========================================
print("\n[5/10] Aplicando pré-processamento...")

# 5.1 Identificar colunas categóricas
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

print(f"✓ Colunas numéricas: {len(numerical_cols)}")
print(f"✓ Colunas categóricas: {len(categorical_cols)}")
if categorical_cols:
    print(f"  - {categorical_cols}")

# 5.2 ONE-HOT ENCODING (se houver categóricas)
if categorical_cols:
    print("\n  Aplicando One-Hot Encoding...")
    
    # Fit no treino
    X_train_encoded = pd.get_dummies(
        X_train, 
        columns=categorical_cols, 
        drop_first=True,  # Remove primeira categoria (evita multicolinearidade)
        dtype=int
    )
    
    # Transform em val e test
    X_val_encoded = pd.get_dummies(X_val, columns=categorical_cols, drop_first=True, dtype=int)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True, dtype=int)
    
    # IMPORTANTE: Alinhar colunas (val e test podem ter categorias diferentes)
    train_cols = X_train_encoded.columns
    X_val_encoded = X_val_encoded.reindex(columns=train_cols, fill_value=0)
    X_test_encoded = X_test_encoded.reindex(columns=train_cols, fill_value=0)
    
    print(f"  ✓ Colunas após OHE: {len(train_cols)}")
else:
    X_train_encoded = X_train.copy()
    X_val_encoded = X_val.copy()
    X_test_encoded = X_test.copy()
    train_cols = X_train.columns

# 5.3 IMPUTAÇÃO (preencher valores ausentes)
print("\n  Aplicando imputação de valores ausentes...")
imputer = SimpleImputer(strategy='mean')

X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train_encoded),
    columns=train_cols,
    index=X_train_encoded.index
)
X_val_imputed = pd.DataFrame(
    imputer.transform(X_val_encoded),
    columns=train_cols,
    index=X_val_encoded.index
)
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test_encoded),
    columns=train_cols,
    index=X_test_encoded.index
)

print(f"  ✓ Valores ausentes após imputação: {X_train_imputed.isnull().sum().sum()}")

# 5.4 NORMALIZAÇÃO (StandardScaler)
print("\n  Aplicando normalização (StandardScaler)...")
scaler = StandardScaler()

X_train_final = pd.DataFrame(
    scaler.fit_transform(X_train_imputed),
    columns=train_cols,
    index=X_train_imputed.index
)
X_val_final = pd.DataFrame(
    scaler.transform(X_val_imputed),
    columns=train_cols,
    index=X_val_imputed.index
)
X_test_final = pd.DataFrame(
    scaler.transform(X_test_imputed),
    columns=train_cols,
    index=X_test_imputed.index
)

print(f"  ✓ Dados normalizados")
print(f"  ✓ Shape final: {X_train_final.shape}")

# ========================================
# 6. TREINAMENTO DO MODELO
# ========================================
print("\n[6/10] Treinando modelo XGBoost...")

# Parâmetros do modelo
xgb_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1
}

print(f"✓ Parâmetros: {xgb_params}")

xgb_model = XGBClassifier(**xgb_params)

# Treinar (sem early stopping para compatibilidade)
start_train = time.time()
xgb_model.fit(X_train_final, y_train)
end_train = time.time()

print(f"✓ Treinamento concluído em {end_train - start_train:.2f}s")
print(f"✓ Número de árvores treinadas: {xgb_model.n_estimators}")

# ========================================
# 7. AVALIAÇÃO NO CONJUNTO DE TREINO
# ========================================
print("\n[7/10] Avaliando no conjunto de TREINO...")
y_pred_train = xgb_model.predict(X_train_final)
train_metrics = {
    'accuracy': accuracy_score(y_train, y_pred_train),
    'precision': precision_score(y_train, y_pred_train),
    'recall': recall_score(y_train, y_pred_train),
    'f1': f1_score(y_train, y_pred_train)
}

print(f"✓ Acurácia:  {train_metrics['accuracy']:.4f}")
print(f"✓ Precision: {train_metrics['precision']:.4f}")
print(f"✓ Recall:    {train_metrics['recall']:.4f}")
print(f"✓ F1-Score:  {train_metrics['f1']:.4f}")

# ========================================
# 8. AVALIAÇÃO NO CONJUNTO DE VALIDAÇÃO
# ========================================
print("\n[8/10] Avaliando no conjunto de VALIDAÇÃO...")
y_pred_val = xgb_model.predict(X_val_final)
val_metrics = {
    'accuracy': accuracy_score(y_val, y_pred_val),
    'precision': precision_score(y_val, y_pred_val),
    'recall': recall_score(y_val, y_pred_val),
    'f1': f1_score(y_val, y_pred_val)
}

print(f"✓ Acurácia:  {val_metrics['accuracy']:.4f}")
print(f"✓ Precision: {val_metrics['precision']:.4f}")
print(f"✓ Recall:    {val_metrics['recall']:.4f}")
print(f"✓ F1-Score:  {val_metrics['f1']:.4f}")

# ========================================
# 9. AVALIAÇÃO NO CONJUNTO DE TESTE
# ========================================
print("\n[9/10] Avaliando no conjunto de TESTE...")
start_predict = time.time()
y_pred_test = xgb_model.predict(X_test_final)
end_predict = time.time()

test_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_test),
    'precision': precision_score(y_test, y_pred_test),
    'recall': recall_score(y_test, y_pred_test),
    'f1': f1_score(y_test, y_pred_test)
}

print(f"✓ Acurácia:  {test_metrics['accuracy']:.4f}")
print(f"✓ Precision: {test_metrics['precision']:.4f}")
print(f"✓ Recall:    {test_metrics['recall']:.4f}")
print(f"✓ F1-Score:  {test_metrics['f1']:.4f}")
print(f"✓ Tempo de inferência: {end_predict - start_predict:.4f}s para {len(X_test):,} amostras")

print("\n✓ Classification Report Detalhado:")
print(classification_report(y_test, y_pred_test, target_names=['Benigno', 'Malicioso']))

# ========================================
# 10. VERIFICAÇÃO DE OVERFITTING
# ========================================
print("\n[10/10] Verificação de Overfitting...")
print("="*80)
print(f"{'Métrica':<15} {'Train':<12} {'Val':<12} {'Test':<12} {'Diferença (Train-Test)'}")
print("="*80)

for metric in ['accuracy', 'precision', 'recall', 'f1']:
    diff = train_metrics[metric] - test_metrics[metric]
    status = "⚠️" if diff > 0.05 else "✅"
    print(f"{metric.capitalize():<15} {train_metrics[metric]:.4f}      "
          f"{val_metrics[metric]:.4f}      {test_metrics[metric]:.4f}      "
          f"{diff:+.4f} {status}")

print("="*80)

# Diagnóstico
train_test_diff = train_metrics['accuracy'] - test_metrics['accuracy']
if train_test_diff > 0.05:
    print("\n⚠️  ATENÇÃO: Possível OVERFITTING detectado!")
    print(f"   Diferença Train-Test: {train_test_diff:.4f} (> 0.05)")
    print("   Recomendações:")
    print("   - Reduzir max_depth")
    print("   - Aumentar min_child_weight")
    print("   - Reduzir n_estimators")
    print("   - Aumentar regularização (reg_alpha, reg_lambda)")
elif train_metrics['accuracy'] < 0.85:
    print("\n⚠️  ATENÇÃO: Possível UNDERFITTING detectado!")
    print(f"   Acurácia no treino muito baixa: {train_metrics['accuracy']:.4f}")
    print("   Recomendações:")
    print("   - Aumentar max_depth")
    print("   - Aumentar n_estimators")
    print("   - Adicionar mais features")
else:
    print("\n✅ MODELO ESTÁ GENERALIZANDO BEM!")
    print(f"   Diferença Train-Test: {train_test_diff:.4f} (< 0.05)")

# ========================================
# VISUALIZAÇÕES
# ========================================
print("\n📊 Gerando visualizações...")

# 1. Matriz de Confusão
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_val = confusion_matrix(y_val, y_pred_val)
cm_test = confusion_matrix(y_test, y_pred_test)

disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=['Benigno', 'Malicioso'])
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=['Benigno', 'Malicioso'])

disp_val.plot(ax=axes[0], cmap='Blues')
axes[0].set_title(f'Validação (Acc: {val_metrics["accuracy"]:.4f})')

disp_test.plot(ax=axes[1], cmap='Greens')
axes[1].set_title(f'Teste (Acc: {test_metrics["accuracy"]:.4f})')

plt.tight_layout()
plt.show()

# 2. Top 15 Features Mais Importantes
print("\n📊 Top 15 Features Mais Importantes:")
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1][:15]

print("="*60)
for i, idx in enumerate(indices, 1):
    print(f"{i:2d}. {train_cols[idx]:<40} {importances[idx]:.4f}")
print("="*60)

# Verificar se há features suspeitas
suspicious_features = [f for f in train_cols[indices[:15]] 
                      if any(x in str(f).lower() for x in ['ip_', 'mac_', 'addr_'])]
if suspicious_features:
    print(f"\n⚠️  ALERTA: Features suspeitas encontradas: {suspicious_features}")
    print("   Essas features podem estar vazando informação!")
else:
    print("\n✅ Nenhuma feature suspeita detectada nas top 15")

# Gráfico de importância
plt.figure(figsize=(12, 6))
plt.bar(range(15), importances[indices], color='steelblue')
plt.xticks(range(15), [train_cols[i] for i in indices], rotation=45, ha='right')
plt.title('Top 15 Features Mais Importantes', fontsize=14, fontweight='bold')
plt.ylabel('Importância', fontsize=12)
plt.xlabel('Feature', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Comparação de Métricas
metrics_names = list(train_metrics.keys())
train_values = [train_metrics[m] for m in metrics_names]
val_values = [val_metrics[m] for m in metrics_names]
test_values = [test_metrics[m] for m in metrics_names]

x = np.arange(len(metrics_names))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, train_values, width, label='Train', color='#2ecc71')
ax.bar(x, val_values, width, label='Val', color='#3498db')
ax.bar(x + width, test_values, width, label='Test', color='#e74c3c')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Comparação de Métricas (Train/Val/Test)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([m.capitalize() for m in metrics_names])
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0.7, 1.05])

plt.tight_layout()
plt.show()

# ========================================
# CROSS-VALIDATION (OPCIONAL)
# ========================================
print("\n🔄 Executando Cross-Validation (5-fold)...")
cv_scores = cross_val_score(
    xgb_model, 
    X_train_final, 
    y_train, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1
)

print(f"✓ CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"✓ Média: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

if cv_scores.std() > 0.05:
    print("⚠️  Alta variância nos folds - modelo pode ser instável")
else:
    print("✅ Baixa variância - modelo é estável")

# ========================================
# RESUMO FINAL
# ========================================
print("\n" + "="*80)
print("RESUMO FINAL")
print("="*80)
print(f"✓ Dataset: {len(df):,} amostras")
print(f"✓ Features utilizadas: {len(train_cols)}")
print(f"✓ Tempo de treinamento: {end_train - start_train:.2f}s")
print(f"✓ Tempo de inferência: {(end_predict - start_predict)/len(X_test)*1000:.2f}ms por amostra")
print(f"\n✓ Acurácia Final (Test): {test_metrics['accuracy']:.4f}")
print(f"✓ F1-Score Final (Test): {test_metrics['f1']:.4f}")
print(f"✓ Status: {'✅ APROVADO' if train_test_diff < 0.05 else '⚠️  REVISAR'}")
print("="*80)