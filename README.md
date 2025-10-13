# EP1-PLN# EP1-PLN

**Resumo**

Repositório criado para o *Exercício Prático 1* da disciplina *Introdução ao Processamento de Língua Natural (PLN)* — objetivo: construir classificadores binários que avaliem se textos gerados pertencem ao estilo desejado. Este repositório implementa a **Parte 1** do trabalho descrito nos slides, entregando infraestrutura de treino, experimentação (grid-search / cross-validation) e geração de previsões.

---

## Estrutura do repositório

```
EP1-PLN/
├─ src/
│  ├─ data/train                       # datasets de treino/teste
│  │  ├─ train_arcaico_moderno.csv
│  │  ├─ train_complexo_simples.csv
│  │  └─ train_literal_dinamico.csv
│  │  
│  ├─ models/                       # modelos treinados (salvos automaticamente)
│  │  ├─ arcaico_moderno/
│  │  ├─ complexo_simples/
│  │  └─ literal_dinamico/
│  ├─ results/                      # logs & predições
│  │  ├─ arcaico_moderno/
│  │  ├─ complexo_simples/
│  │  └─ literal_dinamico/
│  ├─ treino3.py                    # script principal de treino/experimentos
│  └─ teste.py                      # script para carregar modelo e gerar predições
└─ 04c-EP1 (1).pdf                  # slides / enunciado do EP
```
---

## O que este repositório resolve (mapeamento com os slides)

Conforme o enunciado (PDF), a Parte 1 pede a construção de classificadores para 3 tarefas binárias (arcaico vs moderno, complexo vs simples, literal vs dinâmico). Este repositório implementa:

- Pipeline TF-IDF + classif. tradicionais (MultinomialNB, LogisticRegression, RandomForest, AdaBoost, etc.).
- Pipeline Word2Vec custom `Word2VecVectorizer` (treina um W2V no `fit()` e transforma documentos em vetores por média, opcionalmente ponderada por TF-IDF).
- Realização de grid-search manual com `ParameterGrid` + KFold (validação cruzada) e log incremental dos resultados.
- Serialização dos melhores modelos com `joblib` em `src/models/<tarefa>/`.
- Script para carregar modelo salvo e gerar predições (`src/teste.py`).

Essas etapas cobrem o esperado na Parte 1: preparar pipeline, treinar com validação cruzada e salvar resultados (acurácia média de 10 folds, logs e arquivos de predições).

---

## Dependências (instalar antes de rodar)

Recomendado criar um virtualenv e instalar as dependências:

```bash
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
.\.venv\Scripts\activate      # Windows (PowerShell: .\.venv\Scripts\Activate.ps1)

pip install --upgrade pip
pip install -r requirements.txt || pip install pandas scikit-learn numpy joblib gensim
```

---

## Como rodar — Treinamento (Parte 1)

O script principal de treino é `src/treino3.py`. Ele já vem configurado, por padrão, para executar sobre o dataset `train_complexo_simples.csv`.

1. Abra `src/treino3.py` e verifique as variáveis de configuração no topo do arquivo:

```python
# Arquivo CSV de treino (substitua pelo dataset desejado)
TRAIN_CSV = "src/data/train/train_complexo_simples.csv"

# Diretórios de saída
MODEL_DIR = "src/models/complexo_simples"
RESULTS_DIR = "src/results/complexo_simples"
```

2. Para treinar a **tarefa complexo vs simples** (padrão):

```bash
python src/treino3.py
```

3. Para treinar outra tarefa (ex.: arcaico vs moderno):

- Edite as variáveis `TRAIN_CSV`, `MODEL_DIR` e `RESULTS_DIR` no topo de `src/treino3.py` para apontarem para `train_arcaico_moderno.csv` e suas pastas correspondentes (por exemplo `src/models/arcaico_moderno` / `src/results/arcaico_moderno`) e então rode `python src/treino3.py`.

**O que o script faz**
- Carrega o CSV (`;` como separador — verifique o cabeçalho dos arquivos CSV em `src/data`).
- Executa várias pipelines (TF-IDF + classificadores; Word2Vec + classificadores) e combinações de parâmetros via `ParameterGrid`.
- Realiza K-Fold CV (variável `K` definida no código) e calcula métricas (acurácia, F1 weighted).
- Salva logs incrementais em `src/results/<tarefa>/training_results.txt` e sumário dos melhores modelos em `best_models_summary.txt`.
- Serializa modelos com `joblib.dump()` em `src/models/<tarefa>/` com nomes do tipo `best_<modelo>_<representacao>_model.pkl`.

---

## Como rodar — Teste / Geração de predições

O script `src/teste.py` foi criado para carregar um modelo salvo e gerar predições num arquivo de teste.

Por padrão, `src/teste.py` usa as constantes no topo do arquivo (ajuste conforme necessário):

```python
MODEL_PATH = "src/models/complexo_simples/best_multinomial_naive_bayes_tf-idf_model.pkl"
TEST_DATA_PATH = "src/data/teste/test_data_simples_complexo.csv"
OUTPUT_PREDICTIONS_FILENAME = "src/results/complexo_simples/predictions_output.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "style"  # se presente, o script calcula métricas
```

Para executar:

```bash
python src/teste.py
```

Saída:
- Arquivo `src/results/<tarefa>/predictions_output.csv` com o texto original e coluna `prediction` (e coluna de rótulo `style` se havia no arquivo de teste).
- Impressão no terminal do relatório de avaliação (se `LABEL_COLUMN` existir no CSV de teste).

---

## Formato dos dados

Os arquivos CSV na pasta `src/data/train/` possuem formato com cabeçalho e separador `;`. As colunas que o código espera são:

- `text` — texto (string) a ser classificado.
- `style` — rótulo (classe binária, por exemplo: `complexo` / `simples`) — usado apenas durante treino ou quando disponível no CSV de teste para avaliação.

Se você usar seus próprios dados, mantenha o mesmo formato (ou ajuste `TEXT_COLUMN` / `LABEL_COLUMN` em `teste.py`).



