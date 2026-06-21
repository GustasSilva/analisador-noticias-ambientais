# 🌳 Analisador de Notícias Ambientais

Aplicação desktop que utiliza **Machine Learning** para classificar notícias sobre
meio ambiente em dois eixos:

- **Assunto** — Queimadas, Desmatamento, Poluição, Biodiversidade, etc.
- **Sentimento** — Positiva, Negativa ou Neutra (com percentuais de confiança)

O texto pode ser inserido manualmente ou coletado automaticamente a partir da
**URL** de uma notícia. Os resultados podem ser corrigidos pelo usuário e salvos,
fazendo o modelo **reaprender** com os novos dados (aprendizado incremental).

> Projeto desenvolvido como APS (Atividade Prática Supervisionada) na faculdade.

---

## ✨ Funcionalidades

- 🔎 Classificação automática de **assunto** e **sentimento** de notícias
- 🌐 Coleta de texto diretamente de uma **URL** (via `newspaper3k` / `BeautifulSoup`)
- ✏️ **Correção manual** dos resultados, com reaprendizado do modelo
- 💾 Persistência das análises em CSV
- 📊 **Relatórios visuais** com gráficos (evolução por dia e por assunto) usando `matplotlib`
- 🧠 Métricas de avaliação (acurácia, F1-score, matriz de confusão) impressas a cada treino

---

## 🛠️ Tecnologias

| Camada | Ferramentas |
|---|---|
| Interface | `tkinter`, `ttk` |
| Machine Learning | `scikit-learn` (TF-IDF + SVM), `nltk` |
| Manipulação de dados | `pandas`, `numpy` |
| Coleta web | `requests`, `beautifulsoup4`, `newspaper3k`, `lxml` |
| Gráficos | `matplotlib` |
| Persistência de modelos | `joblib` |

**Como funciona o modelo:** o texto é pré-processado (minúsculas, remoção de
acentos e pontuação, stopwords em português), vetorizado com **TF-IDF**
(`max_features=500`) e classificado por dois modelos **SVM**:

- Sentimento → `SVC(kernel="rbf", C=2)` com probabilidades
- Assunto → `SVC(kernel="linear", C=1)`

---

## 🚀 Como executar

### Pré-requisitos
- Python 3.10+ (testado em 3.13)

### Passo a passo

```bash
# 1. Clone o repositório
git clone https://github.com/<seu-usuario>/analisador-noticias-ambientais.git
cd analisador-noticias-ambientais

# 2. (Recomendado) Crie e ative um ambiente virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute a aplicação
python interface.py
```

Na primeira execução, o modelo é **treinado automaticamente** a partir do
`dataset_noticias.csv` e as stopwords do NLTK são baixadas. Os arquivos
`modelo_sentimento.pkl` e `modelo_assunto.pkl` são gerados localmente.

---

## 📁 Estrutura do projeto

```
analisador-noticias-ambientais/
├── interface.py          # Interface gráfica (Tkinter) e fluxo da aplicação
├── modelo.py             # Classe AnalisadorNoticias: pré-processamento, ML e relatórios
├── dataset_noticias.csv  # Base de notícias rotuladas para treino
├── requirements.txt      # Dependências do projeto
├── .gitignore
└── README.md
```

---

## 📝 Observações

- O arquivo `noticias_analisadas.csv` e os modelos `*.pkl` são gerados em tempo
  de execução e, por isso, não são versionados (ver `.gitignore`).
- A extração por URL depende da estrutura da página; algumas fontes podem
  bloquear a coleta automática.
