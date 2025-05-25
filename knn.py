import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def load_data(url):
    """Carrega os dados a partir de uma URL."""
    return pd.read_csv(url)

def identify_features_and_target(data):
    """Identifica as features e o target no dataset."""
    features = data.columns[:-1]  # Todas as colunas, exceto a última
    target = data.columns[-1]     # Última coluna como alvo
    return features, target

def split_data(data, features, target, test_size=0.2, random_state=42):
    """Divide os dados em conjuntos de treino e validação."""
    X = data[features]
    y = data[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_data(X_train, X_val):
    """Padroniza as variáveis numéricas."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled

def train_knn(X_train, y_train, k):
    """Treina o modelo KNN com o valor de K fornecido."""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

def evaluate_model(knn, X_val, y_val):
    """Avalia o modelo e retorna a acurácia."""
    y_pred = knn.predict(X_val)
    return accuracy_score(y_val, y_pred)

def analyze_k_values(X_train, y_train, X_val, y_val, k_values):
    """Analisa diferentes valores de K e retorna o melhor K e suas acurácias."""
    accuracies = []
    for k in k_values:
        knn = train_knn(X_train, y_train, k)
        acc = evaluate_model(knn, X_val, y_val)
        accuracies.append(acc)
    best_k = k_values[accuracies.index(max(accuracies))]
    return best_k, accuracies

# URL da base de dados
url = "https://raw.githubusercontent.com/professortiagoinfnet/inteligencia_artificial/main/heart.csv"
data = load_data(url)

# Identificar features e target
features, target = identify_features_and_target(data)
print("Features:", features)
print("Target:", target)

# Dividir os dados
X_train, X_val, y_train, y_val = split_data(data, features, target)
print("Tamanho do conjunto de treino:", X_train.shape)
print("Tamanho do conjunto de validação:", X_val.shape)

# Transformar variáveis categóricas em numéricas
categorical_columns = X_train.select_dtypes(include=['object']).columns
X_train = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
X_val = pd.get_dummies(X_val, columns=categorical_columns, drop_first=True)

# Padronizar os dados
X_train_scaled, X_val_scaled = preprocess_data(X_train, X_val)

# Treinar modelo inicial
k_initial = 3
knn = train_knn(X_train_scaled, y_train, k_initial)

# Avaliar modelo inicial
accuracy = evaluate_model(knn, X_val_scaled, y_val)
print(f"Acurácia do modelo com K={k_initial}: {accuracy:.2f}")

# Analisar diferentes valores de K
k_values = range(1, 21)
best_k, accuracies = analyze_k_values(X_train_scaled, y_train, X_val_scaled, y_val, k_values)
print(f"Melhor valor de K: {best_k} com acurácia de {max(accuracies):.2f}")