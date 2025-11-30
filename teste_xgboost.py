import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("TESTE BASELINE - MODELO COM FEATURES MÍNIMAS")
print("Objetivo: Verificar se o dataset TON-IoT tem separação trivial")
print("="*80)

# ========================================
# 1. CARREGAR DADOS
# ========================================
print("\n[1/5] Carregando dados...")
df = pd.read_csv('./ton-iot/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv')
print(f"✓ Dataset: {df.shape[0]:,} amostras, {df.shape[1]} colunas")
print(f"✓ Distribuição: {df['label'].value_counts(normalize=True).to_dict()}")

# ========================================
# 2. USAR APENAS FEATURES BÁSICAS DE FLUXO
# ========================================
print("\n[2/5] Selecionando APENAS features básicas de fluxo...")

# Lista ULTRA RESTRITA: apenas estatísticas de tráfego genéricas
BASIC_FLOW_FEATURES = [
    'duration',      # Duração da conexão
    'src_bytes',     # Bytes enviados
    'dst_bytes',     # Bytes recebidos
    'src_pkts',      # Pacotes enviados
    'dst_pkts',      # Pacotes recebidos
    'src_port',      # Porta origem
    'dst_port',      # Porta destino
]

# Verificar quais existem
available_features = [f for f in BASIC_FLOW_FEATURES if f in df.columns]
missing_features = [f for f in BASIC_FLOW_FEATURES if f not in df.columns]

print(f"\n✓ Features disponíveis ({len(available_features)}):")
for f in available_features:
    print(f"   - {f}")

if missing_features:
    print(f"\n⚠️  Features não encontradas ({len(missing_features)}):")
    for f in missing_features:
        print(f"   - {f}")

# Criar dataset baseline
X = df[available_features].copy()
y = df['label'].astype(int)

print(f"\n✓ Dataset baseline: {X.shape[1]} features")
print(f"✓ Tipos de dados:")
print(X.dtypes)

# ========================================
# 3. ANÁLISE EXPLORATÓRIA
# ========================================
print("\n[3/5] Análise exploratória das features básicas...")

# Estatísticas por classe
print("\n📊 Estatísticas por classe (primeiras 3 features):")
print("="*70)
for feature in available_features[:3]:
    print(f"\n{feature}:")
    print(f"  Benigno  (0): média={df[df['label']==0][feature].mean():.2f}, "
          f"std={df[df['label']==0][feature].std():.2f}")
    print(f"  Malicioso(1): média={df[df['label']==1][feature].mean():.2f}, "
          f"std={df[df['label']==1][feature].std():.2f}")

# ========================================
# 4. TREINAR MODELOS BASELINE
# ========================================
print("\n[4/5] Treinando modelos baseline...")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

# Pré-processamento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- MODELO 1: Random Forest Simples ---
print("\n🌲 Modelo 1: Random Forest (baseline)")
rf_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

print(f"   Acurácia: {rf_acc:.4f}")
print(f"   F1-Score: {rf_f1:.4f}")

# --- MODELO 2: XGBoost Simplificado ---
print("\n⚡ Modelo 2: XGBoost (simplificado)")
xgb_model = XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

xgb_acc = accuracy_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)

print(f"   Acurácia: {xgb_acc:.4f}")
print(f"   F1-Score: {xgb_f1:.4f}")

# ========================================
# 5. DIAGNÓSTICO
# ========================================
print("\n[5/5] DIAGNÓSTICO FINAL...")
print("="*80)
print(f"{'Modelo':<20} {'Features':<15} {'Acurácia':<15} {'F1-Score':<15} {'Diagnóstico'}")
print("="*80)

# Diagnóstico Random Forest
rf_diag = "🚨 TRIVIAL!" if rf_acc > 0.95 else ("✅ RAZOÁVEL" if rf_acc > 0.85 else "❌ RUIM")
print(f"{'Random Forest':<20} {len(available_features):<15} {rf_acc:<15.4f} {rf_f1:<15.4f} {rf_diag}")

# Diagnóstico XGBoost
xgb_diag = "🚨 TRIVIAL!" if xgb_acc > 0.95 else ("✅ RAZOÁVEL" if xgb_acc > 0.85 else "❌ RUIM")
print(f"{'XGBoost':<20} {len(available_features):<15} {xgb_acc:<15.4f} {xgb_f1:<15.4f} {xgb_diag}")

print("="*80)

# Interpretação
print("\n🔍 INTERPRETAÇÃO DOS RESULTADOS:\n")

if xgb_acc > 0.95:
    print("🚨 ALERTA CRÍTICO: PROBLEMA TRIVIALMENTE SEPARÁVEL!")
    print("\nCONCLUSÃO:")
    print("   O dataset TON-IoT tem separação quase perfeita entre classes")
    print("   mesmo usando APENAS {0} features básicas de fluxo.".format(len(available_features)))
    print("\nISSO SIGNIFICA QUE:")
    print("   ✓ NÃO há data leakage técnico (IPs, timestamps, etc)")
    print("   ✓ MAS os ataques têm padrões MUITO distintos no tráfego")
    print("   ✓ O problema é 'fácil demais' para ser realista")
    print("\nIMPLICAÇÕES:")
    print("   ⚠️  Modelo pode não generalizar para ataques reais")
    print("   ⚠️  Dataset pode ser sintético ou muito controlado")
    print("   ⚠️  Ataques podem ter sido gerados de forma artificial")
    
elif xgb_acc > 0.85:
    print("✅ RESULTADO RAZOÁVEL")
    print(f"\nCom apenas {len(available_features)} features básicas, alcançamos {xgb_acc:.2%}.")
    print("Isso indica que:")
    print("   ✓ O problema tem dificuldade moderada")
    print("   ✓ Ataques têm padrões identificáveis mas não triviais")
    print("   ✓ Modelo tem potencial de generalização")
    
else:
    print("❌ PERFORMANCE BAIXA")
    print("\nIsso seria esperado se:")
    print("   ✓ Ataques fossem muito similares a tráfego normal")
    print("   ✓ Features básicas não fossem suficientes")
    print("   ⚠️  Mas isso é RARO em datasets de segurança")

# ========================================
# VISUALIZAÇÕES
# ========================================
print("\n📊 Gerando visualizações...")

# 1. Matriz de Confusão
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=['Benigno', 'Malicioso'],
            yticklabels=['Benigno', 'Malicioso'])
axes[0].set_title(f'Random Forest (Acc: {rf_acc:.4f})')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Benigno', 'Malicioso'],
            yticklabels=['Benigno', 'Malicioso'])
axes[1].set_title(f'XGBoost (Acc: {xgb_acc:.4f})')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('baseline_confusion_matrix.png', dpi=150)
plt.show()

# 2. Feature Importance (XGBoost)
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices], color='steelblue')
plt.xticks(range(len(importances)), 
           [available_features[i] for i in indices], 
           rotation=45, ha='right')
plt.title('Importância das Features Básicas (XGBoost)', fontsize=14, fontweight='bold')
plt.ylabel('Importância')
plt.xlabel('Feature')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('baseline_feature_importance.png', dpi=150)
plt.show()

print("\n✓ Visualizações salvas:")
print("   - baseline_confusion_matrix.png")
print("   - baseline_feature_importance.png")

# ========================================
# CLASSIFICATION REPORT
# ========================================
print("\n📋 Classification Report Detalhado (XGBoost):")
print("="*70)
print(classification_report(y_test, y_pred_xgb, 
                          target_names=['Benigno', 'Malicioso'],
                          digits=4))

# ========================================
# RESUMO FINAL
# ========================================
print("\n" + "="*80)
print("RESUMO EXECUTIVO")
print("="*80)
print(f"✓ Features utilizadas: {len(available_features)} (APENAS fluxo básico)")
print(f"✓ Melhor modelo: XGBoost")
print(f"✓ Acurácia baseline: {xgb_acc:.4f}")
print(f"✓ F1-Score baseline: {xgb_f1:.4f}")

if xgb_acc > 0.95:
    print(f"\n🚨 CONCLUSÃO: Dataset TON-IoT é TRIVIALMENTE SEPARÁVEL")
    print(f"   Mesmo com features mínimas, alcançamos {xgb_acc:.2%}")
    print(f"   Isso explica por que seu modelo complexo chegou a 99.9%")
elif xgb_acc > 0.85:
    print(f"\n✅ CONCLUSÃO: Problema tem dificuldade MODERADA")
    print(f"   Performance de {xgb_acc:.2%} é razoável com features básicas")
else:
    print(f"\n❓ CONCLUSÃO: Performance ABAIXO do esperado")
    print(f"   Apenas {xgb_acc:.2%} - investigar qualidade dos dados")

print("="*80)