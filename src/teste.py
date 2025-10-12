import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import os
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_PATH = "src/models/<atividade>/<modelo_escolhido>.pkl"
TEST_DATA_PATH = "src/data/teste/<arquivo_teste>.csv"                
TEXT_COLUMN = "text"                              
LABEL_COLUMN = "style"                
OUTPUT_PREDICTIONS_FILENAME = "src/results/<atividade>/predictions_output_<atividade>.csv"

if not os.path.exists(MODEL_PATH):
    print(f"Erro: O arquivo do modelo \'{MODEL_PATH}\' não foi encontrado.")
    exit(1)
if not os.path.exists(TEST_DATA_PATH):
    print(f"Erro: O arquivo de dados de teste \'{TEST_DATA_PATH}\' não foi encontrado.")
    exit(1)

class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Transformer sklearn-compatible que treina um Word2Vec no fit()
    e transforma documentos em vetores pela média (ou média ponderada por TF-IDF).
    Importante: o __init__ NÃO deve modificar argumentos (ex.: não fazer `or {}`).
    """

    def __init__(self, vector_size=100, window=5, min_count=1, workers=4, sg=1,
                 seed=42, use_tfidf_weighting=False, tfidf_params=None):
        # Apenas armazenar os parâmetros exatamente como recebidos.
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.seed = seed
        self.use_tfidf_weighting = use_tfidf_weighting
        self.tfidf_params = tfidf_params  # NÃO substituir None por {} aqui

        # atributos preenchidos durante fit()
        self.w2v_model = None
        self.tfidf_vectorizer = None
        self._index_to_token = None

    def _tokenize(self, doc):
        if doc is None:
            return []
        return simple_preprocess(str(doc), deacc=True)

    def fit(self, X, y=None):
        # Normalizar tfidf_params aqui, não no __init__
        tfidf_params_local = {} if self.tfidf_params is None else dict(self.tfidf_params)

        # Tokenizar documentos e treinar Word2Vec
        tokenized = [self._tokenize(d) for d in X]
        self.w2v_model = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            seed=self.seed,
        )

        # Se usar ponderação por TF-IDF, ajustar um TfidfVectorizer nos textos do fit
        if self.use_tfidf_weighting:
            # garantir parâmetros razoáveis por padrão
            tfidf_default = {"analyzer": "word", "token_pattern": r"(?u)\b\w+\b"}
            tfidf_default.update(tfidf_params_local)
            self.tfidf_vectorizer = TfidfVectorizer(**tfidf_default)
            self.tfidf_vectorizer.fit(X)
            # criar mapa índice -> token para uso posterior
            inv_vocab = {v: k for k, v in self.tfidf_vectorizer.vocabulary_.items()}
            self._index_to_token = inv_vocab
        else:
            self.tfidf_vectorizer = None
            self._index_to_token = None

        return self

    def transform(self, X):
        # garante que o modelo foi treinado
        if self.w2v_model is None:
            raise RuntimeError("Word2VecVectorizer não foi fit() antes do transform()")

        out = np.zeros((len(X), self.vector_size), dtype=float)

        for i, doc in enumerate(X):
            if self.use_tfidf_weighting and self.tfidf_vectorizer is not None:
                tfidf_vec = self.tfidf_vectorizer.transform([doc])
                if tfidf_vec.nnz == 0:
                    out[i] = np.zeros(self.vector_size)
                    continue
                indices = tfidf_vec.indices
                data = tfidf_vec.data
                vec = np.zeros(self.vector_size, dtype=float)
                weight_sum = 0.0
                for idx, w in zip(indices, data):
                    token = self._index_to_token.get(idx)
                    if token is None:
                        continue
                    if token in self.w2v_model.wv:
                        vec += w * self.w2v_model.wv[token]
                        weight_sum += w
                if weight_sum > 0:
                    out[i] = vec / weight_sum
                else:
                    out[i] = np.zeros(self.vector_size)
            else:
                tokens = self._tokenize(doc)
                vec = np.zeros(self.vector_size, dtype=float)
                count = 0
                for t in tokens:
                    if t in self.w2v_model.wv:
                        vec += self.w2v_model.wv[t]
                        count += 1
                if count > 0:
                    out[i] = vec / count
                else:
                    out[i] = np.zeros(self.vector_size)
        return out


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