import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def main():
  iris = datasets.load_iris()

  # Divide os dados em conjuntos de treinamento e teste (2/3 para treinamento e 1/3 para teste)
  X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)

  qda = QuadraticDiscriminantAnalysis()
  qda.fit(X_train, y_train)
  
  y_pred = qda.predict(X_test)

  print(f"=== Teste ({len(X_test)}) ===")
  for target, output in zip(y_test, y_pred):
    target_name = iris.target_names[target]
    output_name = iris.target_names[output]
    print(f"Saida desejada: {target_name}")
    print(f"Saida real: {output_name}")
    print("-")

  accuracy = accuracy_score(y_test, y_pred)
  conf_matrix = confusion_matrix(y_test, y_pred)

  print(f"Acurácia: {accuracy}")
  print("Matriz de Confusão:\n", conf_matrix)

  return

if __name__ == "__main__":
   np.set_printoptions(precision=6, floatmode="fixed", suppress=True)
   main()