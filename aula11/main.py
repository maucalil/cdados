import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def main():
  data = np.genfromtxt("files/wine.data", delimiter=',')

  # First column values are the targets
  X = data[:, 1:]
  y = data[:, 0]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

  qda = QuadraticDiscriminantAnalysis()
  qda.fit(X_train, y_train)

  y_pred = qda.predict(X_test)
  labels = np.unique(y)
  print(labels)

  accuracy = accuracy_score(y_test, y_pred)
  conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)

  print(f"Dados totais: {len(data)}")
  print(f"Dados treinamento: {len(X_train)}")
  print(f"Dados teste: {len(X_test)}")
  print("Matriz de Confusão\n", conf_matrix)
  print(f"Acurácia: {accuracy:.6f}")

  return

if __name__ == "__main__":
   np.set_printoptions(precision=6, floatmode="maxprec", suppress=True)
   main()