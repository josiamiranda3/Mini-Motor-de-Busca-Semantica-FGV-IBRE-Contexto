# Mini Motor de Busca Semântica — FGV IBRE Contexto

Sistema de recuperação de notícias econômicas que combina **busca semântica, busca lexical BM25 e busca híbrida**, com API REST, avaliação quantitativa e testes automatizados.

Este projeto foi desenvolvido no contexto do Desafio Técnico para Estágio em Ciência de Dados do FGV IBRE e evoluiu para uma aplicação modular de recuperação de informação utilizando técnicas de Processamento de Linguagem Natural (NLP).

## Visão geral

O objetivo é recuperar notícias econômicas relevantes a partir de consultas em linguagem natural, considerando tanto o significado dos textos quanto a correspondência entre palavras-chave.

A solução possui três estratégias de recuperação:

- **Busca semântica:** utiliza embeddings e similaridade de cosseno para identificar notícias semanticamente relacionadas à consulta.
- **Busca lexical BM25:** utiliza correspondência lexical e relevância dos termos pesquisados.
- **Busca híbrida:** combina os rankings dos dois métodos por meio de Reciprocal Rank Fusion (RRF).

## Arquitetura

O pipeline de processamento e recuperação segue este fluxo:

```text
noticias_brutas.json
        |
        v
Limpeza e tratamento dos textos
        |
        v
noticias_limpas.json
        |
        v
Geração de embeddings
        |
        v
embeddings.npy
        |
        v
Motores de busca
  |       |       |
  v       v       v
Semântica BM25  Híbrida
        |
        v
Avaliação dos resultados
        |
        v
API REST com FastAPI
```

## Tecnologias utilizadas

- **Python:** linguagem principal.
- **BeautifulSoup:** remoção de conteúdo HTML.
- **Sentence Transformers:** geração de embeddings multilíngues.
- **Scikit-learn:** cálculo de similaridade de cosseno.
- **NumPy:** armazenamento e processamento de vetores.
- **Pandas:** suporte ao processamento e à análise de dados.
- **rank-bm25:** recuperação lexical com BM25.
- **FastAPI:** criação da API REST.
- **Uvicorn:** servidor ASGI.
- **Pytest:** testes automatizados.

## Estratégias de busca

### 1. Busca semântica

Cada notícia é representada por um embedding produzido pelo modelo `paraphrase-multilingual-MiniLM-L12-v2`, utilizando o título e o conteúdo textual tratado.

A consulta também é transformada em embedding. Em seguida, são calculadas as similaridades entre a consulta e as notícias indexadas.

O motor retorna os resultados mais relevantes, incluindo identificador, título, fonte, data, similaridade e trecho do texto.

### 2. Busca lexical com BM25

O BM25 avalia a relevância lexical dos documentos considerando os termos da consulta e suas características no conjunto de notícias.

Esse método complementa a busca semântica, especialmente quando a correspondência entre os termos pesquisados e o conteúdo dos documentos é importante.

### 3. Busca híbrida com RRF

A busca híbrida combina os rankings produzidos pelos motores semântico e BM25 por meio de Reciprocal Rank Fusion (RRF).

A combinação considera as posições dos documentos em cada ranking, permitindo aproveitar informações complementares dos dois métodos.

## Preparação e tratamento dos dados

O pipeline de limpeza contempla operações como:

- Remoção de tags HTML e decodificação de entidades HTML.
- Tratamento de URLs e outros ruídos textuais.
- Normalização de espaços e quebras de linha.
- Tratamento de textos curtos e duplicados.
- Padronização do conteúdo para as etapas seguintes.

Os dados processados são armazenados em `dados/noticias_limpas.json`.

## Geração e validação de embeddings

Os embeddings são armazenados em `dados/embeddings.npy`.

O projeto também utiliza metadados para verificar a compatibilidade dos embeddings com os dados e o modelo utilizados, ajudando a identificar arquivos desatualizados ou inconsistentes.

Os arquivos derivados podem ser regenerados a partir dos dados de entrada e do pipeline correspondente.

## Avaliação dos métodos

O projeto implementa métricas de recuperação de informação:

- **Precision@K:** proporção de resultados relevantes entre os primeiros K documentos retornados.
- **Recall@K:** proporção dos documentos relevantes conhecidos que foram recuperados.
- **Mean Reciprocal Rank (MRR):** avalia a posição do primeiro resultado relevante.

Também há um conjunto de consultas com documentos relevantes identificados para comparar os métodos.

Em uma avaliação preliminar com 10 consultas e 19 notícias, os resultados registrados foram:

| Métrica | Semântica | BM25 | Híbrida |
|---|---:|---:|---:|
| Precision@5 | 0,36 | 0,40 | 0,40 |
| Recall@5 | 0,9333 | 1,00 | 1,00 |
| MRR | 1,00 | 0,95 | 1,00 |

Esses resultados são exploratórios. O conjunto de avaliação é pequeno e os documentos relevantes foram identificados manualmente. Portanto, os números não devem ser interpretados como uma garantia de desempenho em bases maiores.

## API REST

A aplicação disponibiliza uma API com FastAPI e documentação interativa pelo Swagger.

### Endpoints

| Método | Endpoint | Descrição |
|---|---|---|
| GET | `/` | Informações gerais da API |
| GET | `/health` | Verificação de funcionamento e quantidade de notícias carregadas |
| GET | `/buscar` | Pesquisa de notícias |

### Parâmetros da busca

O endpoint `/buscar` aceita os seguintes parâmetros:

- `consulta`: texto da pesquisa.
- `metodo`: `semantica`, `bm25` ou `hibrida`.
- `top_k`: quantidade máxima de resultados, entre 1 e 19 na configuração atual.

Exemplo de consulta:

```text
GET /buscar?consulta=inflação%20e%20taxa%20Selic&metodo=hibrida&top_k=5
```

Exemplo de resposta:

```json
{
  "consulta": "inflação e taxa Selic",
  "metodo": "hibrida",
  "total": 5,
  "resultados": [
    {
      "id": 1,
      "titulo": "Título da notícia",
      "fonte": "Fonte da notícia",
      "data": "2024-01-01",
      "similaridade": 0.85,
      "trecho": "Trecho ilustrativo da notícia...",
      "pontuacao_rrf": 0.03
    }
  ]
}
```

O JSON acima é apenas ilustrativo. Os valores e o conteúdo real dependem dos documentos recuperados.

## Como executar o projeto

### 1. Clonar o repositório

```bash
git clone git@github.com:josiamiranda3/Mini-Motor-de-Busca-Semantica-FGV-IBRE-Contexto.git
cd Mini-Motor-de-Busca-Semantica-FGV-IBRE-Contexto
```

O comando SSH pressupõe que sua chave SSH esteja configurada no GitHub.

### 2. Criar e ativar o ambiente virtual

No Windows com Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 3. Instalar as dependências

```bash
python -m pip install -r requirements.txt
```

### 4. Executar o pipeline

```bash
python main.py
```

Consulte o código do pipeline para verificar as etapas executadas por esse comando.

### 5. Iniciar a API

Na raiz do projeto, execute:

```bash
python -m uvicorn src.api:app --reload
```

A API ficará disponível localmente em:

- Aplicação: `http://127.0.0.1:8000`
- Documentação Swagger: `http://127.0.0.1:8000/docs`
- Verificação de saúde: `http://127.0.0.1:8000/health`

Na primeira inicialização, o modelo de embeddings pode precisar ser carregado. É necessário ter acesso ao modelo local ou à internet para baixá-lo quando ainda não estiver disponível.

## Testes automatizados

O projeto conta com testes para a API, os motores de busca, a avaliação e a limpeza dos dados.

Para executar toda a suíte:

```bash
python -m pytest -v
```

Na validação registrada durante o desenvolvimento, **67 testes passaram**, sem falhas, em 27,52 segundos.

Esse resultado representa a execução naquele ambiente e não constitui garantia de desempenho idêntico em outras máquinas.

## Estrutura do projeto

```text
Mini-Motor-de-Busca-Semantica-FGV-IBRE-Contexto/
├── dados/
│   ├── noticias_brutas.json
│   ├── noticias_limpas.json
│   ├── embeddings.npy
│   ├── embeddings_metadata.json
│   └── consultas_avaliacao.json
├── src/
│   ├── api.py
│   ├── avaliacao.py
│   ├── busca.py
│   ├── busca_bm25.py
│   ├── busca_hibrida.py
│   ├── comparar_buscas.py
│   ├── embeddings.py
│   └── limpeza.py
├── tests/
│   ├── test_api.py
│   ├── test_avaliacao.py
│   ├── test_busca.py
│   ├── test_busca_hibrida.py
│   └── test_limpeza.py
├── main.py
├── requirements.txt
└── README.md
```

A estrutura representa os principais arquivos esperados. Confira os nomes existentes no repositório antes de publicar esta árvore.

## Limitações atuais

- A base de notícias e o conjunto de consultas de avaliação são pequenos.
- A avaliação de relevância depende de rótulos definidos manualmente.
- O desempenho ainda precisa ser medido em bases maiores e sob carga.
- A aplicação não possui, nesta versão, uma interface web dedicada para usuários finais.

## Possíveis evoluções

- Ampliar o conjunto de notícias e consultas de avaliação.
- Comparar sistematicamente os métodos com benchmarks maiores.
- Implementar re-ranking com cross-encoder.
- Avaliar índices vetoriais como FAISS para bases maiores.
- Adicionar filtros por data e fonte.
- Implementar logs, monitoramento e implantação em ambiente de produção.
- Criar uma interface web para pesquisa e comparação de resultados.

## Considerações finais

O projeto demonstra a aplicação prática de NLP, recuperação de informação, processamento de dados, avaliação quantitativa e desenvolvimento de APIs em Python.

A combinação de busca semântica, BM25 e RRF permite estudar diferentes estratégias de recuperação de notícias econômicas, enquanto os testes automatizados ajudam a verificar a confiabilidade dos componentes.

Autor: Josias Miranda Oliveira de Lima
