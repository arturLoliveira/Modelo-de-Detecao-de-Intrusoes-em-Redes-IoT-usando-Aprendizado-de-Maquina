import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from lightgbm import LGBMClassifier
import pandas as pd
import glob

# # Lista todos os arquivos que terminam com .csv na pasta (assumindo que você baixou só a versão 5%)
# arquivos = glob.glob('./ton-iot/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv')
# # arquivos = glob.glob('./ton-iot/Train_Test_datasets/Train_Test_Windows_dataset/Train_Test_Windows_10.csv')

# # arquivos = glob.glob('./reduce/reduced_data_1.csv')


# arquivos = glob.glob('./ton-iot/Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Garage_Door.csv')
# lista_dfs = []

# for i, f in enumerate(arquivos):
#     print(f"[{i+1}/{len(arquivos)}] Lendo: {f}...", end='\r') # Mostra progresso
#     try:
#         # Tenta ler rápido e pular linhas com erro
#         df_temp = pd.read_csv(f, on_bad_lines='skip')
#         lista_dfs.append(df_temp)
#     except Exception as e:
#         print(f"\n❌ Falha crítica no arquivo {f}: {e}")

# print("\nConcatenando tudo...")
# if lista_dfs:
#     df = pd.concat(lista_dfs, ignore_index=True)
#     print("Concluído com sucesso!")
# else:
#     print("Nenhum dado carregado.")


# # Lê e junta todos em um único DataFrame
# lista_dfs = []

# for i, f in enumerate(arquivos):
#     print(f"[{i+1}/{len(arquivos)}] Lendo: {f}...", end='\r') # Mostra progresso
#     try:
#         # Tenta ler rápido e pular linhas com erro
#         df_temp = pd.read_csv(f, on_bad_lines='skip')
#         lista_dfs.append(df_temp)
#     except Exception as e:
#         print(f"\n❌ Falha crítica no arquivo {f}: {e}")

# print("\nConcatenando tudo...")
# if lista_dfs:
#     df = pd.concat(lista_dfs, ignore_index=True)
#     print("Concluído com sucesso!")
# else:
#     print("Nenhum dado carregado.")

# 1. Carregamento
print("Carregando dataset...")
df = pd.read_csv('./ton-iot/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv')

# 2. REMOÇÃO DAS "COLAS" (ASSINATURAS)
# Aqui removemos tudo que identifica o ataque por texto ou porta específica
cols_to_drop = [
    'src_ip', 'dst_ip',       # Identificadores de origem/destino
    'src_port', 'dst_port',   # Portas (Muitos ataques usam portas fixas, viciando o modelo)
    'dns_query',              # Assinatura de texto (ex: url maliciosa)
    'http_uri',               # Assinatura de texto (ex: /admin/backdoor)
    'http_user_agent',        # Assinatura de ferramenta de ataque
    'ssl_subject', 'ssl_issuer', # Certificados específicos
    'weird_name', 'weird_addl', 'weird_notice', # Alertas específicos do Zeek
    'label', 'type'           # Targets (removemos label/type de X)
]
print("Colunas carregadas:", df.columns.tolist())
print("\nValores mais frequentes em 'DeviceName':")
# Contando os tipos de ataque/tráfego (coluna 'label')
print(df['label'].value_counts().head(5))


# Removemos colunas que existem no DF mas não queremos usar
existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
X = df.drop(columns=existing_cols_to_drop)
y = df['label'] # 0 ou 1

print(f"Colunas removidas para evitar vício: {existing_cols_to_drop}")
print(f"Features restantes para aprendizado comportamental: {X.shape[1]}")

# 3. Pré-processamento (Encoding e Limpeza)
# Tratamento do caractere '-' como NaN
X = X.replace('-', np.nan)

# Separar numéricas e categóricas
cat_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(include=['number']).columns

print("Processando categóricas (One-Hot Encoding)...")
# Usamos dummy_na=True para tratar nulos como uma categoria separada
X = pd.get_dummies(X, columns=cat_cols, dummy_na=True, drop_first=True)

# Imputação de valores numéricos (caso haja NaN)
imputer = SimpleImputer(strategy='mean')
X[num_cols] = imputer.fit_transform(X[num_cols])

# 4. Divisão Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Normalização (Importante para redes neurais, mas bom manter aqui)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Treinamento e Medição de Tempo (LightGBM)
print("\n--- Iniciando Treinamento 'Prova de Fogo' (LightGBM) ---")
model = LGBMClassifier(random_state=42, n_jobs=-1)

start_train = time.time()
model.fit(X_train, y_train)
end_train = time.time()

train_time = end_train - start_train
print(f"Tempo de Treino: {train_time:.4f} segundos")

# 7. Inferência e Medição de Tempo
print("\n--- Iniciando Inferência ---")
start_pred = time.time()
y_pred = model.predict(X_test)
end_pred = time.time()

pred_time = end_pred - start_pred
print(f"Tempo de Inferência ({len(X_test)} amostras): {pred_time:.4f} segundos")
print(f"Tempo médio por amostra: {(pred_time / len(X_test)) * 1000:.4f} ms")

# 8. Resultados
acc = accuracy_score(y_test, y_pred)
print(f"\n=== Acurácia Realista: {acc:.4f} ===")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Matriz de Confusão (Sem Assinaturas)")
plt.show()