import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.base import clone, TransformerMixin, BaseEstimator
import joblib
import datetime
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")  # Ignorar warnings de UndefinedMetricWarning, etc.

if len(sys.argv) < 2:
    print("Erro: Forneça o nome da atividade como um argumento.")
    print("Exemplo de uso: python treino3.py arcaico_moderno")
    sys.exit(1)

atividade = sys.argv[1]
print(f"🚀 Iniciando treino para a atividade: {atividade}")

# Arquivo CSV de treino
TRAIN_CSV = f"src/data/train/train_{atividade}.csv"

# Diretórios de saída
MODEL_DIR = f"src/models/{atividade}"
RESULTS_DIR = f"src/results/{atividade}"

# Arquivos de log (serão gravados dentro de RESULTS_DIR)
LOG_FILE_ALL_RESULTS = os.path.join(RESULTS_DIR, "training_results.txt")
LOG_FILE_BEST_MODELS = os.path.join(RESULTS_DIR, "best_models_summary.txt")

# Certificar que diretórios existem
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------
# Word2Vec transformer
# -------------------------
try:
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
except Exception as e:
    raise ImportError(
        "gensim não encontrado. Instale com: pip install gensim\n"
        "O pipeline Word2Vec depende do gensim."
    )

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer

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


# -------------------------
# Carregar dados e limpeza
# -------------------------
data = pd.read_csv(TRAIN_CSV, sep=";")
before = len(data)
data = data.dropna(subset=["text"])              # remove NaN em text
data = data[data["text"].str.strip() != ""]      # remove strings vazias ou só espaços
after = len(data)

print(f"Arquivo de treino: {TRAIN_CSV}")
print(f"Linhas removidas: {before - after}")
print(f"Total de linhas após limpeza: {after}")

X = data["text"]
y = data["style"]

# Dividir os dados em treino e teste (para simular um conjunto de teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------
# Pipelines base (TF-IDF) - mantidos
# -------------------------
pipeline_mnb_tfidf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB()),
])

pipeline_lr_tfidf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=2000, solver="liblinear")),
])

pipeline_rf_tfidf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", RandomForestClassifier(random_state=42)),
])

pipeline_gb_tfidf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", GradientBoostingClassifier(random_state=42)),
])

pipeline_ab_tfidf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", AdaBoostClassifier(random_state=42)),
])

pipeline_dummy_tfidf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", DummyClassifier(strategy="most_frequent")),
])

# -------------------------
# Pipelines base (Word2Vec)
# -------------------------
# NOTE: MultinomialNB não entra aqui pois espera features não-negativas (contagens/probabilidades).
pipeline_lr_w2v = Pipeline([
    ("w2v", Word2VecVectorizer()),
    ("clf", LogisticRegression(max_iter=2000, solver="liblinear")),
])

pipeline_rf_w2v = Pipeline([
    ("w2v", Word2VecVectorizer()),
    ("clf", RandomForestClassifier(random_state=42)),
])

pipeline_gb_w2v = Pipeline([
    ("w2v", Word2VecVectorizer()),
    ("clf", GradientBoostingClassifier(random_state=42)),
])

pipeline_ab_w2v = Pipeline([
    ("w2v", Word2VecVectorizer()),
    ("clf", AdaBoostClassifier(random_state=42)),
])

pipeline_dummy_w2v = Pipeline([
    ("w2v", Word2VecVectorizer()),
    ("clf", DummyClassifier(strategy="most_frequent")),
])

# -------------------------
# Parâmetros (TF-IDF) - seus originais com leve ajuste
# -------------------------
parameters_mnb = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_df": [0.75, 1.0],
    "tfidf__min_df": [1, 5],
    "clf__alpha": [0.1, 0.5, 1.0],
}

parameters_lr = {
    "tfidf__ngram_range": [(1, 2), (1, 3)],
    "tfidf__max_df": [0.85, 1.0],
    "tfidf__min_df": [1, 5],
    "clf__C": [25, 300, 400],
    "clf__penalty": ["l1", "l2"],
}

parameters_rf = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_df": [0.75],
    "tfidf__min_df": [1, 3],
    "clf__n_estimators": [150, 200],
    "clf__max_depth": [None],
    "clf__min_samples_leaf": [3, 5],
}

parameters_gb = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_df": [0.75],
    "tfidf__min_df": [1, 3],
    "clf__n_estimators": [200, 250],
    "clf__learning_rate": [0.2, 0.3],
    "clf__max_depth": [7, 10],
}

parameters_ab = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_df": [0.75, 1.0],
    "tfidf__min_df": [1, 5],
    "clf__n_estimators": [50, 100],
    "clf__learning_rate": [0.5, 1.0],
}

parameters_dummy = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_df": [1.0],
    "tfidf__min_df": [1],
}

# -------------------------
# Parâmetros (Word2Vec) - keep small to control tempo
# -------------------------
parameters_lr_w2v = {
    "w2v__vector_size": [2000,5000],
    "w2v__window": [30],
    "w2v__min_count": [5],
    "w2v__sg": [1],
    "w2v__use_tfidf_weighting": [False],
    "clf__C": [600,1000],
    "clf__penalty": ["l2"],
}

parameters_rf_w2v = {
    "w2v__vector_size": [100],
    "w2v__window": [3,5],
    "w2v__min_count": [1],
    "w2v__sg": [1],
    "w2v__use_tfidf_weighting": [False, True],
    "clf__n_estimators": [150, 200],
    "clf__min_samples_leaf": [3, 5],
}

parameters_gb_w2v = {
    "w2v__vector_size": [100],
    "w2v__window": [305],
    "w2v__min_count": [1],
    "w2v__sg": [1],
    "w2v__use_tfidf_weighting": [False, True],
    "clf__n_estimators": [200],
    "clf__learning_rate": [0.2, 0.3],
    "clf__max_depth": [7],
}

parameters_ab_w2v = {
    "w2v__vector_size": [100],
    "w2v__window": [3,5],
    "w2v__min_count": [1],
    "w2v__sg": [1],
    "w2v__use_tfidf_weighting": [False, True],
    "clf__n_estimators": [50, 100],
    "clf__learning_rate": [0.5, 1.0],
}

parameters_dummy_w2v = {
    "w2v__vector_size": [100],
    "w2v__window": [3],
    "w2v__min_count": [1],
    "w2v__sg": [1],
    "w2v__use_tfidf_weighting": [False, True],
}

# -------------------------
# Construir lista unificada de modelos a testar (duplica TF-IDF + Word2Vec)
# -------------------------
models_to_test = {
    # TF-IDF variants (mantidos)
    "Multinomial Naive Bayes (TF-IDF)": (pipeline_mnb_tfidf, parameters_mnb),
    "Random Forest (TF-IDF)": (pipeline_rf_tfidf, parameters_rf),
    # "Gradient Boosting (TF-IDF)": (pipeline_gb_tfidf, parameters_gb),
    "AdaBoost (TF-IDF)": (pipeline_ab_tfidf, parameters_ab),
    "DummyClassifier (TF-IDF)": (pipeline_dummy_tfidf, parameters_dummy),

    # Word2Vec variants (NOTE: MNB omitted aqui)
    "Random Forest (Word2Vec)": (pipeline_rf_w2v, parameters_rf_w2v),
    # "Gradient Boosting (Word2Vec)": (pipeline_gb_w2v, parameters_gb_w2v),
    "AdaBoost (Word2Vec)": (pipeline_ab_w2v, parameters_ab_w2v),
    "DummyClassifier (Word2Vec)": (pipeline_dummy_w2v, parameters_dummy_w2v),
    "Logistic Regression (TF-IDF)": (pipeline_lr_tfidf, parameters_lr),
    "Logistic Regression (Word2Vec)": (pipeline_lr_w2v, parameters_lr_w2v),
}

# # Limpar o arquivo de log principal anterior, se existir (opcional)
# with open(LOG_FILE_ALL_RESULTS, "w", encoding="utf-8") as f:
#     f.write("Início do Log de Treinamento de Modelos (Resultados Incrementais)\n\n")
#     class_counts = y_train.value_counts()
#     total_samples = len(y_train)
#     f.write("Proporção de Classes no Conjunto de Treinamento:\n")
#     for cls, count in class_counts.items():
#         f.write(f"  {cls}: {count} ({count/total_samples:.2%})\n")
#     f.write(f"Total de amostras de treinamento: {total_samples}\n\n")
#
# # Limpar o arquivo de log de melhores modelos, se existir (opcional)
# with open(LOG_FILE_BEST_MODELS, "w", encoding="utf-8") as f:
#     f.write("Sumário das Melhores Arquiteturas por Modelo\n\n")

# -------------------------
# Loop principal de treino (grid manual com KFold)
# -------------------------
for model_name, (base_pipeline, param_grid) in models_to_test.items():
    print(f"\nIniciando testes para o modelo: {model_name}...")

    best_score_model = -1  # Para rastrear a melhor acurácia/f1 para este modelo
    best_params_model = None
    best_estimator_model = None

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    all_params_combinations = list(ParameterGrid(param_grid))
    for i, params in enumerate(all_params_combinations):
        print(f"  Testando combinação {i+1} de {len(all_params_combinations)} para {model_name} com parâmetros: {params}")

        # clone para não poluir o pipeline base
        current_pipeline = clone(base_pipeline).set_params(**params)

        fold_accuracies = []
        fold_f1_scores = []

        for fold, (train_index, val_index) in enumerate(kf.split(X_train), start=1):
            n_splits = kf.get_n_splits()
            print(f"    Processando Fold {fold}/{n_splits} para {model_name} - Combinação {i+1}...")
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            # Treina no fold
            current_pipeline.fit(X_train_fold, y_train_fold)
            y_pred_val = current_pipeline.predict(X_val_fold)

            fold_accuracies.append(accuracy_score(y_val_fold, y_pred_val))
            fold_f1_scores.append(f1_score(y_val_fold, y_pred_val, average="weighted"))

        avg_accuracy_cv = float(np.mean(fold_accuracies))
        avg_f1_cv = float(np.mean(fold_f1_scores))

        # Avaliar no conjunto de teste (treinando com todo o X_train)
        current_pipeline.fit(X_train, y_train)
        y_pred_test = current_pipeline.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_f1_score = f1_score(y_test, y_pred_test, average="weighted")

        # Salvar resultados desta combinação no log principal
        with open(LOG_FILE_ALL_RESULTS, "a", encoding="utf-8") as f:
            f.write(f"\n\n{'='*50}\n")
            f.write(f"Modelo: {model_name} - Combinação de Parâmetros {i+1}\n")
            f.write(f"Data e Hora: {datetime.datetime.now()}\n")
            f.write(f"Parâmetros: {params}\n")
            f.write(f"Média Acurácia (Validação Cruzada): {avg_accuracy_cv:.4f}\n")
            f.write(f"Média F1-Score (Validação Cruzada): {avg_f1_cv:.4f}\n")
            f.write(f"Acurácia no Conjunto de Teste: {test_accuracy:.4f}\n")
            f.write(f"F1-Score (weighted) no Conjunto de Teste: {test_f1_score:.4f}\n")
            f.write(f"Relatório de Classificação no Conjunto de Teste:\n")
            f.write(classification_report(y_test, y_pred_test) + "\n")
            f.write("Matriz de Confusão no Conjunto de Teste:\n")
            cm = confusion_matrix(y_test, y_pred_test, labels=y.unique())
            f.write(str(cm) + "\n")
            f.write(f"Classes: {y.unique().tolist()}\n")
            f.write(f"\n\n{'='*50}\n")

        # Rastrear o melhor (por CV accuracy)
        if avg_accuracy_cv > best_score_model:
            best_score_model = avg_accuracy_cv
            best_params_model = params
            best_estimator_model = current_pipeline

    # Salvar melhor estimador (se houver)
    if best_estimator_model is not None:
        # nome do arquivo inclui indicação TF-IDF ou Word2Vec
        safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        model_filename = f"best_{safe_name}_model.pkl"
        model_path = os.path.join(MODEL_DIR, model_filename)
        joblib.dump(best_estimator_model, model_path)

        with open(LOG_FILE_ALL_RESULTS, "a", encoding="utf-8") as f:
            f.write(f"\n\n{'='*50}\n")
            f.write(f"Melhor Arquitetura para {model_name} salva em {model_path}\n")
            f.write(f"Melhores Parâmetros: {best_params_model}\n")
            f.write(f"Melhor Acurácia (Validação Cruzada): {best_score_model:.4f}\n")
            f.write(f"\n\n{'='*50}\n")

        with open(LOG_FILE_BEST_MODELS, "a", encoding="utf-8") as f:
            f.write(f"\n\n{'='*50}\n")
            f.write(f"Modelo: {model_name}\n")
            f.write(f"Melhores Parâmetros: {best_params_model}\n")
            f.write(f"Melhor Acurácia (Validação Cruzada): {best_score_model:.4f}\n")
            f.write(f"Caminho do Modelo Salvo: {model_path}\n")
            f.write(f"\n\n{'='*50}\n")

print(f"\nTodos os testes de parâmetros foram concluídos e os resultados detalhados foram salvos em: {LOG_FILE_ALL_RESULTS}")
print(f"Os melhores modelos para cada tipo de classificador foram salvos como arquivos .pkl em: {MODEL_DIR} e um sumário está em: {LOG_FILE_BEST_MODELS}")