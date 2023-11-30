import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics  import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

def main():
    # Carrega os dados do arquivo
    data_df = load_data("data.csv")

    # Normaliza os dados
    normalized_data_df = normalize_data(data_df)

    # Extract features and target variable
    X = normalized_data_df.drop(["class"], axis=1) # features
    y = normalized_data_df["class"] # targets
    class_labels = y.unique()

    # Calcula a media dos coeficientes para cada classe
    class_means = normalized_data_df.groupby('class').mean()

    # Calcula a matriz de correlação entre as classes
    class_corr_matrix = class_means.T.corr()
    plot_correlation_matrix(class_corr_matrix)

    # Cria os modelos
    mlp = MLPClassifier(**get_mlp_parameters())
    qda = QuadraticDiscriminantAnalysis()
    lda = LinearDiscriminantAnalysis()
    svm = SVC(kernel='linear')

    models = [mlp, qda, lda, svm]
    results = []

    for model in models:
        # Treino e teste do modelo utilizando validacao cruzada
        y_pred = cross_val_predict(model, X, y, cv=5)

        # Metricas do modelo (acuracia, precisao, recall, f1-score)
        report = classification_report(y, y_pred, target_names=class_labels, output_dict=True)
        metrics = get_metrics_from_report(report)
        results.append({
            'Model': type(model).__name__,
            **metrics
        })

        # Matriz de confusão
        cm = confusion_matrix(y, y_pred, labels=class_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot()
        plt.show()
    
    # Salva os resultados para um CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('results.csv', index=False)

# Carrega dados do CSV em um Pandas DataFrame
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Normaliza as features usando escalonamento Min-Max
def normalize_data(data: pd.DataFrame, range=(-1, 1)):
    features = data.drop(["audio_name", "class"], axis=1)
    scaler = MinMaxScaler(range)
    normalized_features = scaler.fit_transform(features)

    # Cria um novo DataFrame com as features normalizadas
    data_normalized = pd.DataFrame(data=normalized_features, columns=features.columns)
    data_normalized["class"] = data["class"]
    return data_normalized

# Define os parametros do MLP
def get_mlp_parameters():
    return {
        'hidden_layer_sizes': (14),
        'activation': 'tanh',
        'max_iter': 10000,
        'solver': 'adam',
        'random_state': 42,
        'learning_rate': 'constant',
        'learning_rate_init': 0.01
    }

def plot_correlation_matrix(correlation_matrix):
    # Define o tamanho da figura
    fig, ax = plt.subplots(figsize=(10, 8))

    # Cria um mapa de calor
    cax = ax.matshow(correlation_matrix, cmap='coolwarm')
    fig.colorbar(cax)

    # Configura os labels
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)

    # Adiciona o valor numerico no quadrado
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='w')

    plt.title("Matriz de Correlação")
    plt.show()

# Extrai as metricas do dicionario
def get_metrics_from_report(report):
    avg_precision = report['weighted avg']['precision']
    avg_recall = report['weighted avg']['recall']
    avg_f1_score = report['weighted avg']['f1-score']
    accuracy = report['accuracy']

    return {
        'Avg Precision': round(avg_precision,6),
        'Avg Recall': round(avg_recall, 6),
        'Avg F1 Score': round(avg_f1_score, 6),
        'Accuracy': round(accuracy, 6)
    }

if __name__ == "__main__":
    main()