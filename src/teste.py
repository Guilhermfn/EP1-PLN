import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import os

MODEL_PATH = "models/best_logistic_regression_tf-idf_model.pkl" 
TEST_DATA_PATH = "data/test_data_simples_complexo.csv"                
TEXT_COLUMN = "text"                              
LABEL_COLUMN = "style"                
OUTPUT_PREDICTIONS_FILENAME = "results/predictions_output.csv"

if not os.path.exists(MODEL_PATH):
    print(f"Erro: O arquivo do modelo \'{MODEL_PATH}\' não foi encontrado.")
    exit(1)
if not os.path.exists(TEST_DATA_PATH):
    print(f"Erro: O arquivo de dados de teste \'{TEST_DATA_PATH}\' não foi encontrado.")
    exit(1)

print(f"Carregando modelo de: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

print(f"Carregando dados de teste de: {TEST_DATA_PATH}")
test_data = pd.read_csv(TEST_DATA_PATH, sep=";")

if TEXT_COLUMN not in test_data.columns:
    print(f"Erro: A coluna de texto \'{TEXT_COLUMN}\' não foi encontrada nos dados de teste.")
    exit(1)

X_new = test_data[TEXT_COLUMN]

print("Fazendo previsões...")
predictions = model.predict(X_new)

test_data["predicted_style"] = predictions

print("\n--- Previsões --- ")
print(test_data[[TEXT_COLUMN, "predicted_style"]].head())
print(f"Total de previsões realizadas: {len(predictions)}")

if LABEL_COLUMN and LABEL_COLUMN in test_data.columns:
    y_true = test_data[LABEL_COLUMN]
    print("\n--- Avaliação do Modelo --- ")
    print(f"Acurácia: {accuracy_score(y_true, predictions):.4f}")
    print("\nRelatório de Classificação:\n")
    print(classification_report(y_true, predictions))
    print("\nMatriz de Confusão:\n")
    print(confusion_matrix(y_true, predictions, labels=y_true.unique()))
    print(f"Classes: {y_true.unique().tolist()}")
elif LABEL_COLUMN and LABEL_COLUMN not in test_data.columns:
    print(f"Aviso: A coluna de rótulo \'{LABEL_COLUMN}\' não foi encontrada nos dados de teste. Nenhuma avaliação será realizada.")
else:
    print("Nenhuma coluna de rótulo fornecida. Nenhuma avaliação será realizada.")

# Salvar as previsões em um novo arquivo CSV
test_data.to_csv(OUTPUT_PREDICTIONS_FILENAME, index=False)
print(f"\nPrevisões salvas em: {OUTPUT_PREDICTIONS_FILENAME}")