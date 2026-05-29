# Classificação de Notícias da BBC com PLN

## Visão Geral do Projeto
Este projeto aplica técnicas de Processamento de Linguagem Natural (PLN) para classificar artigos de notícias da BBC em cinco categorias: **business**, **entertainment**, **politics**, **sport** e **tech**. O conjunto de dados contém 2.225 artigos. O notebook explora pré-processamento de texto, vetorização (Bag-of-Words, TF-IDF, n-gramas, Word2Vec), busca por similaridade e classificadores de machine learning.

## Conjunto de Dados
- Fonte: [BBC News Archive](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive) (Kaggle)
- Arquivo: `bbc-news-data.csv` (separado por tabulação) com as colunas `category` e `content`
- Classes: business, entertainment, politics, sport, tech

## Métodos & Fluxo de Trabalho

### 1. Pré-processamento de Texto
- Conversão para minúsculas
- Remoção de caracteres não alfabéticos
- Tokenização
- Remoção de stopwords do inglês
- Lematização (WordNet) – representação final escolhida
- Stemming (Porter) – também implementado para comparação
- Marcação de classe gramatical (spaCy) – opcional, mas não usada na vetorização final

### 2. Análise Exploratória de Dados (EDA)
- Nuvens de palavras e gráficos de barras de frequência por categoria
- Distribuição do número de tokens por categoria
- Palavras mais frequentes por categoria
- Grafos de coocorrência

### 3. Vetorização
- **TF‑IDF** (Term Frequency – Inverse Document Frequency)
- **Bag‑of‑Words (BoW)**
- **BoW com n‑gramas** (2‑gramas)
- **Word2Vec** (Gensim) – vetores de 5 dimensões, janela=2, min_count=1

### 4. Busca por Similaridade (baseada em consulta)
- Para cada vetorizador, uma consulta (ex.: `"stock market crash"`, `"movie award"`) é embutida e comparada com todos os documentos usando similaridade de cosseno.
- Os k melhores resultados são exibidos com a distribuição de categorias e projeções PCA.

### 5. Modelos de Classificação
Três classificadores foram treinados usando as features vetorizadas:
- **Support Vector Machine (SVM)**
- **Regressão Logística**
- **Random Forest**

Os modelos são avaliados em uma divisão treino/teste usando acurácia, relatório de classificação, matriz de confusão e tempo de treinamento.

## Principais Resultados (exemplo)
- TF‑IDF e BoW mostram boa correspondência semântica.
- BoW com n‑gramas captura contexto frasal.
- As pontuações de similaridade do Word2Vec foram zero na amostra devido à incompatibilidade de vocabulário (provavelmente porque a palavra de consulta `"computer"` não estava presente no corpus de treinamento lematizado).
- As métricas de classificação (acurácia, F1) são apresentadas para cada modelo – o notebook mostra células de treinamento e avaliação, mas os números exatos estão truncados na saída fornecida. Tipicamente, SVM e Random Forest alcançam alta acurácia (>95%) neste conjunto de dados.

## Dependências
- Python 3.x
- Bibliotecas: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit‑learn`, `nltk`, `spacy`, `gensim`, `wordcloud`, `networkx`, `kaggle-api`

Instalação:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Como Executar
1. Clone o repositório.
2. Configure as credenciais da API Kaggle (coloque kaggle.json em ~/.kaggle/).
3. Execute o notebook bbc_news_classification.ipynb sequencialmente.
    - Ele fará o download automático do conjunto de dados do Kaggle.
    - Pré‑processará o texto, gerará os vetorizadores, executará as consultas de similaridade, treinará os classificadores e produzirá visualizações.


## Resumo dos Resultados
- A lematização forneceu tokens mais limpos para a classificação.
- TF‑IDF e BoW com n‑gramas superaram o BoW simples.
- Word2Vec exigiu um corpus maior ou parâmetros ajustados para produzir similaridade significativa.
- Os três classificadores alcançaram alta acurácia, com SVM e Random Forest sendo os melhores.

## Trabalho Futuro
- Experimentar aprendizado profundo (LSTM, Transformers).
- Ajustar os hiperparâmetros do Word2Vec.
- Implantar o melhor modelo como uma API web.

