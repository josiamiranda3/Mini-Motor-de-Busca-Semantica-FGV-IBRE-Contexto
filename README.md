 Mini Motor de Busca Semântica — FGV IBRE
 Contexto

Este projeto foi desenvolvido como parte do Desafio Técnico para Estágio em Ciência de Dados do FGV IBRE.

O objetivo é construir um mini motor de busca semântico capaz de recuperar notícias econômicas relevantes a partir de consultas em linguagem natural, utilizando técnicas de processamento de linguagem natural (NLP).

 Visão Geral da Solução

O pipeline foi dividido em três etapas principais:

noticias_brutas.json

     
        ↓
        
[Limpeza de texto]
        
        ↓
        
dados_limpos.json
        
        ↓
        
[Geração de embeddings]
        ↓
        
  embeddings.npy
        
        ↓
        
[Busca semântica]

 Tecnologias Utilizadas
Python
BeautifulSoup (remoção de HTML)
Sentence Transformers (embeddings)
Scikit-learn (similaridade de cosseno)
NumPy
Pandas

 Etapa 1 — Limpeza e Tratamento de Texto
Problema

O campo texto contém diversos ruídos:

Tags HTML (<p>, <strong>, etc.)
Entidades HTML (&ccedil;, &nbsp;, etc.)
Quebras de linha excessivas
Links
Metadados embutidos (datas, horários)
Espaços duplicados
Casos de texto mínimo
Solução adotada

Foi implementado um pipeline de limpeza que realiza:

Conversão de entidades HTML → html.unescape
Remoção de tags HTML → BeautifulSoup
Remoção de links e URLs → regex
Remoção de timestamps e metadados → regex
Normalização de espaços → regex
Strip final para padronização
Resultado

Os dados limpos são salvos em:

dados/noticias_limpas.json

 Etapa 2 — Geração de Embeddings
 
Abordagem

Cada notícia foi transformada em um vetor numérico (embedding), combinando:

titulo + texto_limpo
Modelo escolhido
paraphrase-multilingual-MiniLM-L12-v2
Justificativa

O modelo foi escolhido por:

Suporte a português (multilíngue)
Bom desempenho em similaridade semântica
Leve e rápido (ideal para desafio técnico)
Amplamente utilizado em tarefas de busca semântica
Observação

Os embeddings são normalizados para melhorar o cálculo da similaridade de cosseno.

Output
dados/embeddings.npy

(Arquivo não versionado no GitHub por ser reproduzível)

 Etapa 3 — Busca Semântica
Funcionamento

A busca segue os passos:

Recebe uma consulta em linguagem natural
Gera embedding da consulta
Calcula similaridade de cosseno com todas as notícias
Retorna os Top-K resultados mais relevantes
Métrica utilizada
Similaridade de cosseno
Output da busca

Para cada resultado:

ID
Título
Fonte
Data
Similaridade
Trecho do texto

 Como Executar o Projeto
 
1. Clonar repositório
git clone (https://github.com/jmolima2000/Desafio-Tecnico-Estagio-em-Ciencia-de-Dados.git)
cd desafio-ciencia-de-dados-ibre

2. Criar ambiente virtual
python -m venv venv

4. Ativar ambiente
   

Windows (Git Bash):


source venv/Scripts/activate

4. Instalar dependências
pip install -r requirements.txt

5. Executar pipeline completo
python main.py

 Validação das Consultas

Consulta 1
"mudanças na taxa de juros"

✔ Retornos relevantes:

Selic
Copom
Banco Central
Política monetária
Consulta 2
"mercado de trabalho e desemprego"

✔ Retornos relevantes:

Taxa de desemprego
Desemprego juvenil
PNAD Contínua
Mercado de trabalho
Consulta 3
"inflação e preços ao consumidor"

✔ Retornos relevantes:

IPCA
IGP-M
IPA
Inflação ao produtor
Preços ao consumidor

 Avaliação Qualitativa

O modelo demonstrou boa capacidade de capturar similaridade semântica, mesmo quando:

As palavras exatas não estavam presentes
Havia variações de contexto econômico
Pontos fortes
Boa generalização semântica
Recuperação coerente de temas econômicos
Robustez mesmo com textos ruidosos
Limitações
Algumas notícias menos relevantes aparecem no Top-K
Falta de re-ranqueamento fino
Não considera peso explícito de palavras-chave importantes

 Melhorias Futuras
 
Uso de FAISS para busca vetorial mais eficiente
Re-ranking dos resultados (cross-encoder)
Aplicação de stemming/lemmatization
Peso maior para títulos
Interface web para consultas
Filtros por data e fonte

 Estrutura do Projeto:
.
├── dados/

│   ├── noticias_brutas.json
│   ├── noticias_limpas.json
│

├── src/

│   ├── limpeza.py
│   ├── embeddings.py
│   └── busca.py
│

├── main.py
├── requirements.txt
└── README.md

Josias Miranda Oliveira de Lima

GitHub: https://github.com/jmolima2000
LinkedIn: https://www.linkedin.com/in/josias-miranda-lima/
Considerações Finais

A solução foi construída com foco em:

Clareza
Reprodutibilidade
Organização
Aplicação prática de NLP

Mesmo sendo um projeto simples, ele simula um fluxo real de sistemas de busca semântica utilizados em ambientes corporativos.
