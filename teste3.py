import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import glob

print("="*80)
print("BOT-IOT - CARREGAMENTO SELETIVO ESTRATÉGICO")
print("="*80)

# ========================================
# 1. ESTRATÉGIA DE SELEÇÃO DE ARQUIVOS
# ========================================
print("\n[1/7] Definindo estratégia de seleção...")

# ARQUIVOS ESSENCIAIS (ajuste os nomes conforme sua estrutura)
# Pega 1-2 arquivos de cada categoria principal

SELECTED_FILES = {
    'Normal': [
        'benign_traffic.csv',  # Tráfego legítimo
    ],
    'DDoS': [
        'DDoS_HTTP.csv',       # DDoS via HTTP
        'DDoS_TCP.csv',        # DDoS via TCP
    ],
    'DoS': [
        'DoS_HTTP.csv',        # DoS via HTTP
    ],
    'Reconnaissance': [
        'OS_Fingerprint.csv',  # OS Fingerprinting
        'Service_Scan.csv',    # Port/Service scanning
    ],
    'Theft': [
        'Data_Exfiltration.csv',  # Roubo de dados
        'Keylogging.csv',         # Keylogger
    ]
}

print("✓ Estratégia: Amostragem balanceada por categoria de ataque")
print(f"✓ Total de arquivos selecionados: {sum(len(files) for files in SELECTED_FILES.values())}")
print("\n  Arquivos por categoria:")
for category, files in SELECTED_FILES.items():
    print(f"    {category}: {len(files)} arquivo(s)")

# ========================================
# 2. FUNÇÃO PARA ENCONTRAR ARQUIVOS
# ========================================
def find_files(base_path, pattern_dict):
    """
    Busca arquivos no diretório baseado em padrões
    """
    found_files = {}
    
    # Listar TODOS os CSVs disponíveis
    all_csvs = glob.glob(os.path.join(base_path, '**/*.csv'), recursive=True)
    
    print(f"\n  Total de CSVs encontrados no diretório: {len(all_csvs)}")
    
    if len(all_csvs) == 0:
        print(f"  ⚠️  Nenhum CSV encontrado em: {base_path}")
        return {}
    
    # Mostrar estrutura encontrada
    print("\n  Estrutura de pastas encontrada:")
    folders = set(os.path.dirname(f).replace(base_path, '') for f in all_csvs)
    for folder in sorted(folders)[:10]:
        print(f"    {folder if folder else '(raiz)'}")
    if len(folders) > 10:
        print(f"    ... e mais {len(folders)-10} pastas")
    
    # Tentar mapear arquivos
    for category, file_list in pattern_dict.items():
        found_files[category] = []
        for filename in file_list:
            # Buscar arquivo (case-insensitive)
            matches = [f for f in all_csvs if filename.lower() in os.path.basename(f).lower()]
            if matches:
                found_files[category].append(matches[0])
                print(f"  ✓ Encontrado: {filename} -> {matches[0]}")
            else:
                print(f"  ⚠️  Não encontrado: {filename}")
    
    return found_files

# ========================================
# 3. BUSCAR ARQUIVOS
# ========================================
print("\n[2/7] Buscando arquivos no diretório...")

BASE_PATH = './Bot-IoT/'  # Ajuste conforme necessário

found_files = find_files(BASE_PATH, SELECTED_FILES)

# Se nenhum arquivo específico for encontrado, usar abordagem alternativa
if sum(len(files) for files in found_files.values()) == 0:
    print("\n⚠️  Arquivos específicos não encontrados!")
    print("   Usando abordagem alternativa: primeiros N arquivos de cada pasta\n")
    
    # Estratégia alternativa: pegar primeiros arquivos de cada pasta
    all_csvs = glob.glob(os.path.join(BASE_PATH, '**/*.csv'), recursive=True)
    
    # Agrupar por pasta
    by_folder = {}
    for csv in all_csvs:
        folder = os.path.dirname(csv)
        if folder not in by_folder:
            by_folder[folder] = []
        by_folder[folder].append(csv)
    
    print(f"  Pastas encontradas: {len(by_folder)}")
    
    # Pegar 1-2 arquivos de cada pasta
    found_files = {'mixed': []}
    for folder, files in list(by_folder.items())[:5]:  # Primeiras 5 pastas
        selected = files[:2]  # Primeiros 2 arquivos de cada
        found_files['mixed'].extend(selected)
        print(f"  ✓ Pasta {os.path.basename(folder)}: {len(selected)} arquivo(s)")

# ========================================
# 4. CARREGAR E AMOSTRAR DADOS
# ========================================
print("\n[3/7] Carregando e amostrando dados...")

SAMPLE_SIZE_PER_FILE = 50000  # 50k por arquivo

def load_with_sampling(filepath, sample_size=50000):
    """
    Carrega CSV com amostragem
    """
    print(f"\n  Processando: {os.path.basename(filepath)}")
    
    try:
        # Ler primeiras linhas para verificar estrutura
        df_peek = pd.read_csv(filepath, nrows=100)
        print(f"    Colunas: {len(df_peek.columns)}")
        
        # Verificar tamanho do arquivo
        total_lines = sum(1 for _ in open(filepath, 'r')) - 1  # -1 para header
        print(f"    Total de linhas: {total_lines:,}")
        
        if total_lines <= sample_size:
            # Arquivo pequeno, carregar tudo
            df = pd.read_csv(filepath)
            print(f"    ✓ Carregado completamente: {len(df):,} linhas")
        else:
            # Arquivo grande, amostrar
            # Calcular skip para amostragem uniforme
            skip = sorted(np.random.choice(range(1, total_lines), 
                                          size=total_lines - sample_size, 
                                          replace=False))
            df = pd.read_csv(filepath, skiprows=skip)
            print(f"    ✓ Amostrado: {len(df):,} de {total_lines:,} linhas")
        
        # Limpeza básica
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        print(f"    ✓ Após limpeza: {len(df):,} linhas")
        
        return df
        
    except Exception as e:
        print(f"    ✗ Erro: {e}")
        return None

# Carregar todos os arquivos selecionados
dataframes = []
total_samples = 0

for category, file_list in found_files.items():
    if not file_list:
        continue
    
    print(f"\n  Categoria: {category}")
    for filepath in file_list:
        df = load_with_sampling(filepath, SAMPLE_SIZE_PER_FILE)
        if df is not None and len(df) > 0:
            dataframes.append(df)
            total_samples += len(df)
            print(f"    Total acumulado: {total_samples:,} amostras")

if not dataframes:
    print("\n❌ Nenhum dado foi carregado!")
    print("\n💡 SOLUÇÕES:")
    print("   1. Verifique se o caminho está correto:")
    print(f"      BASE_PATH = '{BASE_PATH}'")
    print("   2. Liste os arquivos disponíveis:")
    print("      import glob")
    print(f"      files = glob.glob('{BASE_PATH}/**/*.csv', recursive=True)")
    print("      print(files[:10])")
    print("   3. Ou use abordagem manual:")
    print("      df = pd.read_csv('caminho/do/arquivo.csv')")
    exit()

# ========================================
# 5. CONCATENAR E PADRONIZAR
# ========================================
print("\n[4/7] Concatenando e padronizando dados...")

# Concatenar todos
df = pd.concat(dataframes, ignore_index=True)
print(f"✓ Dataset combinado: {len(df):,} amostras")

# Identificar coluna de label (pode ter nomes diferentes)
label_candidates = ['label', 'Label', 'attack', 'category', 'Category', 'class', 'Class']
label_col = None

for candidate in label_candidates:
    if candidate in df.columns:
        label_col = candidate
        break

if label_col is None:
    print("\n⚠️  Coluna de label não encontrada automaticamente")
    print("   Colunas disponíveis:")
    print(df.columns.tolist()[:20])
    
    # Tentar última coluna (geralmente é o label)
    label_col = df.columns[-1]
    print(f"\n   Usando última coluna como label: '{label_col}'")

print(f"\n✓ Coluna de label: '{label_col}'")
print(f"✓ Classes únicas: {df[label_col].nunique()}")
print(f"✓ Distribuição:")
print(df[label_col].value_counts().head(10))

# ========================================
# 6. CRIAR CLASSIFICAÇÃO BINÁRIA
# ========================================
print("\n[5/7] Criando classificação binária...")

# Bot-IoT geralmente tem labels numéricos ou texto
# 0 = Normal/Benign, outros = Ataques

# Detectar valores de "normal"
normal_values = ['0', 0, 'normal', 'Normal', 'NORMAL', 'benign', 'Benign', 'BENIGN']

df['label_binary'] = df[label_col].apply(
    lambda x: 0 if x in normal_values else 1
)

print(f"✓ Classificação binária criada")
print(f"✓ Distribuição binária:")
print(df['label_binary'].value_counts(normalize=True))

# ========================================
# 7. SEPARAR FEATURES E TARGET
# ========================================
print("\n[6/7] Separando features e target...")

# Colunas a remover (identificadores)
cols_to_remove = [
    label_col, 'label_binary',
    'pkSeqID', 'saddr', 'daddr', 'flgs', 'proto',  # Bot-IoT específicos
    'seq', 'stddev', 'N_IN_Conn_P_SrcIP'  # Features problemáticas
]

# Remover apenas as que existem
cols_removed = [col for col in cols_to_remove if col in df.columns]
X = df.drop(columns=cols_removed, errors='ignore')
y = df['label_binary']

# Garantir apenas colunas numéricas
X = X.select_dtypes(include=[np.number])

print(f"✓ Features: {X.shape[1]} colunas")
print(f"✓ Amostras: {len(X):,}")
print(f"✓ Colunas removidas: {len(cols_removed)}")

# Listar primeiras features
print(f"\n✓ Primeiras 15 features:")
for i, col in enumerate(X.columns[:15], 1):
    print(f"   {i:2d}. {col}")
if len(X.columns) > 15:
    print(f"   ... e mais {len(X.columns)-15} features")

# ========================================
# 8. SPLIT E NORMALIZAÇÃO
# ========================================
print("\n[7/7] Train/Test split e normalização...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Train: {len(X_train):,} amostras ({len(X_train)/len(X)*100:.1f}%)")
print(f"✓ Test:  {len(X_test):,} amostras ({len(X_test)/len(X)*100:.1f}%)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Normalização aplicada")

# ========================================
# SALVAR DADOS PROCESSADOS
# ========================================
print("\n💾 Salvando dados processados...")

try:
    import pickle
    
    data = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'feature_names': X.columns.tolist(),
        'scaler': scaler
    }
    
    with open('botiot_processed.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print("✓ Arquivo salvo: botiot_processed.pkl")
    
    # Também salvar CSVs para facilitar
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('botiot_X_train.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('botiot_X_test.csv', index=False)
    pd.Series(y_train.values).to_csv('botiot_y_train.csv', index=False, header=['label'])
    pd.Series(y_test.values).to_csv('botiot_y_test.csv', index=False, header=['label'])
    
    print("✓ CSVs salvos: botiot_X_train.csv, botiot_X_test.csv, etc.")
    
except Exception as e:
    print(f"⚠️  Erro ao salvar: {e}")

# ========================================
# RESUMO FINAL
# ========================================
print("\n" + "="*80)
print("RESUMO - BOT-IOT PROCESSADO")
print("="*80)
print(f"✓ Arquivos processados: {len(dataframes)}")
print(f"✓ Total de amostras: {len(df):,}")
print(f"✓ Features finais: {X.shape[1]}")
print(f"✓ Train samples: {len(X_train):,}")
print(f"✓ Test samples: {len(X_test):,}")
print(f"✓ Proporção de ataques (train): {(y_train==1).sum()/len(y_train)*100:.2f}%")
print(f"✓ Proporção de ataques (test): {(y_test==1).sum()/len(y_test)*100:.2f}%")
print("="*80)

print("\n✅ Bot-IoT pronto para treinamento!")

print("\n📊 ESTATÍSTICAS DE MEMÓRIA:")
memory_mb = (X_train_scaled.nbytes + X_test_scaled.nbytes) / 1024 / 1024
print(f"   Tamanho em memória: ~{memory_mb:.1f} MB")
print(f"   Viável para processamento: {'✅ SIM' if memory_mb < 500 else '⚠️ PODE SER PESADO'}")

print("\n📝 PRÓXIMOS PASSOS:")
print("   1. Treinar modelo XGBoost")
print("   2. Comparar com TON-IoT")
print("   3. Analisar features mais importantes")
print("   4. Avaliar generalização")