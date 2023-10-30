import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

def load_wine_data():
    data = np.genfromtxt("files/wine/wine.data", delimiter=',')
    print(f"=== Dados Vinhos ({len(data)}) ===")
    features = data[:, 1:] # todas as linhas, e todas colunas exceto a primeira
    labels = data[:, 0] # todas as linhas, e apenas a primeira coluna
    return features, labels

def load_phishing_data(): # https://archive.ics.uci.edu/dataset/379/website+phishing
    data = np.genfromtxt("files/pishing/data.txt", delimiter=',')
    print(f"=== Dados Website Pishing ({len(data)}) ===")
    features = data[:, :-1] # todas as linhas, e todas colunas exceto a ultima
    labels = data[:, -1] # todas as linhas, e apenas a ultima coluna
    return features, labels

def train_and_test(features, labels, test_size, type="QDA"):
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=test_size, random_state=42)

    if type == "QDA":
        classifier = QuadraticDiscriminantAnalysis()
    elif type == "LDA":
        classifier = LinearDiscriminantAnalysis()
    elif type == "SVM":
        classifier = svm.SVC(kernel="linear")

    print(f"\n[!] Proporcao treino: {1 - test_size:.6f}")
    print(f"[!] Proporcao teste: {test_size:.6f}")
    print(f"[!] Discriminante: {type}")    
    print(f"\n> Treinamento ({len(features_train)})")
    classifier.fit(features_train, labels_train)
    labels_pred_train = classifier.predict(features_train)
    print_metrics(labels_train, labels_pred_train)
    
    print(f"\n> Teste ({len(features_test)})")
    labels_pred_test = classifier.predict(features_test)
    print_metrics(labels_test, labels_pred_test)
    print("\n---------------------------------")

def print_metrics(target, output):
    conf_matrix = confusion_matrix(target, output) # Como comprimir matriz em uma binaria?
    accuracy = accuracy_score(target, output) # Mede o percentual de acertos
    precision = precision_score(target, output, average='weighted') # TP / TP + FP
    recall = recall_score(target, output, average='weighted') # TP / TP + FN
    f1 = f1_score(target, output, average='weighted') # media harmonica entre a precisão e o recall

    print("Matriz de Confusao:\n", conf_matrix)
    print(f"Acuracia: {accuracy:.6f}")
    print(f"Precisao: {precision:.6f}")
    print(f"Recall/Sensibilidade: {recall:.6f}")
    print(f"F1-Score: {f1:.6f}")
    print("-")

def main():
    wine_features, wine_labels = load_wine_data()
    train_and_test(wine_features, wine_labels, test_size=1/3)
    train_and_test(wine_features, wine_labels, test_size=2/3)
    train_and_test(wine_features, wine_labels, test_size=1/3, type="LDA")
    train_and_test(wine_features, wine_labels, test_size=1/3, type="SVM")

    phishing_features, phishing_labels = load_phishing_data()
    train_and_test(phishing_features, phishing_labels, test_size=1/3)
    train_and_test(phishing_features, phishing_labels, test_size=1/3, type="LDA")
    train_and_test(phishing_features, phishing_labels, test_size=1/3, type="SVM")

if __name__ == "__main__":
    np.set_printoptions(precision=6, floatmode="maxprec", suppress=True)
    main()
