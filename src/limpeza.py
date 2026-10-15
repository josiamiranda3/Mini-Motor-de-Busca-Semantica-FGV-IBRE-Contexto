import html
import json
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup


# Correções específicas confirmadas na base de notícias.
CORRECOES_ESPACAMENTO = {
    "relaçãoao": "relação ao",
    "menornível": "menor nível",
    "comcrescimento": "com crescimento",
    "desoja": "de soja",
    "comcarteira": "com carteira",
    "Ocomitê": "O comitê",
    "naspróximas": "nas próximas",
    "Ataxa": "A taxa",
    "recordehistórico": "recorde histórico",
    "nospreços": "nos preços",
    "doisa": "dois a",
    "mercadofinanceiro": "mercado financeiro",
    "segundosemestre": "segundo semestre",
    "pelamelhora": "pela melhora",
    "queo": "que o",
    "mesestenha": "meses tenha",
    "históricade": "histórica de",
    "médiahistórica": "média histórica",
    "aoValor": "ao Valor",
    "peloValor": "pelo Valor",
    "aoProdutor": "ao Produtor",
    "demelhor": "de melhor",
    "demenor": "de menor",
    "aospreços": "aos preços",
    "nasexpectativas": "nas expectativas",
    "Omovimento": "O movimento",
    "emrelaçãoao": "em relação ao",
    "crescimentosustentado": "crescimento sustentado",
    "deserviços": "de serviços",
    "esperadofoi": "esperado foi",
    "safrade": "safra de",
    "menorpressão": "menor pressão",
}


# Chamadas externas que podem ser removidas quando aparecem
# no final de uma notícia. A frase "Dados completos disponíveis em"
# é preservada para manter o comportamento esperado pelos testes.
CHAMADAS_FINAIS = (
    r"Leia o comunicado completo em",
    r"Acesse a PNAD Contínua completa",
    r"Veja a cotação completa em BCB\s*-\s*Histórico de Cotações",
    r"Para acessar o relatório completo,\s*visite o Portal FGV IBRE",
    r"Fonte:\s*MDIC\s*[—–-]?",
)


def corrigir_espacamento(texto: str) -> str:
    """Corrige espaços ausentes sem alterar números decimais."""

    # Separa palavras concatenadas com mudança de maiúsculas/minúsculas.
    texto = re.sub(
        r"(?<=[a-zà-ÿ])(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ])",
        " ",
        texto,
    )

    texto = re.sub(
        r"(?<=[A-ZÁÉÍÓÚÂÊÔÃÕÇ])"
        r"(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-zà-ÿ])",
        " ",
        texto,
    )

    # Só insere espaço após pontuação quando vem uma letra.
    # Assim, valores como 0,12 e 13,75 permanecem intactos.
    texto = re.sub(
        r"([,;:!?])(?=[A-Za-zÀ-ÿ])",
        r"\1 ",
        texto,
    )

    texto = re.sub(
        r"([.!?])(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ])",
        r"\1 ",
        texto,
    )

    # Corrige percentuais concatenados com a palavra seguinte.
    texto = re.sub(
        r"(%)(?=[A-Za-zÀ-ÿ])",
        r"\1 ",
        texto,
    )

        # Separa anos concatenados a palavras, como grãos2022/2023.
    texto = re.sub(
        r"(?<=[A-Za-zÀ-ÿ])(?=\d{4}(?:/\d{4})?\b)",
        " ",
        texto,
    )

    # Separa números de unidades e substantivos conhecidos.
    texto = re.sub(
        r"(?<=\d)(?=(?:milhões|milhão|pontos|ponto)\b)",
        " ",
        texto,
        flags=re.IGNORECASE,
    )

    return re.sub(r"\s+", " ", texto).strip()


def corrigir_concatenacoes_conhecidas(texto: str) -> str:
    """Corrige concatenações previamente identificadas nos dados."""

    for incorreto, correto in CORRECOES_ESPACAMENTO.items():
        padrao = re.compile(
            rf"(?<!\w){re.escape(incorreto)}(?!\w)",
            flags=re.IGNORECASE,
        )

        def substituir(match, substituicao=correto):
            original = match.group(0)

            if original and original[0].isupper():
                return substituicao[0].upper() + substituicao[1:]

            return substituicao

        texto = padrao.sub(substituir, texto)

    # Corrige a duplicação específica encontrada na notícia 17.
    texto = re.sub(
    r"\b(?:o\s+)?ar\s+arrefecimento\b",
    "o arrefecimento",
    texto,
    flags=re.IGNORECASE,
)

    return texto


def remover_chamadas_finais(texto: str) -> str:
    """Remove chamadas externas conhecidas no final das notícias."""

    for chamada in CHAMADAS_FINAIS:
        texto = re.sub(
            rf"(?:\s*[-—–|:]?\s*){chamada}\s*[.!…]*\s*$",
            "",
            texto,
            flags=re.IGNORECASE,
        )

    return texto.strip()


def limpar_texto(texto: str) -> str:
    """Remove HTML, URLs e ruídos comuns e normaliza o espaçamento."""

    if not isinstance(texto, str):
        return ""

    texto = html.unescape(texto)

    soup = BeautifulSoup(texto, "html.parser")
    texto = soup.get_text(separator=" ")

    # Remove metadados de publicação conhecidos.
    texto = re.sub(
        r"\bPublicado em:\s*\d{1,2}/\d{1,2}/\d{4}"
        r"(?:\s*(?:às|as)?\s*\d{1,2}h\d{0,2})?",
        " ",
        texto,
        flags=re.IGNORECASE,
    )

    # Remove URLs completas.
    texto = re.sub(
        r"(?:https?://|www\.)\S+",
        " ",
        texto,
        flags=re.IGNORECASE,
    )

    # Remove domínios isolados, preservando o texto ao redor.
    texto = re.sub(
        r"(?<![@\w\.-])"
        r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+"
        r"(?:com|org|net|gov|edu|br|io|info|biz|co)"
        r"(?:\.[a-z]{2})?"
        r"(?![\w\.-])",
        " ",
        texto,
        flags=re.IGNORECASE,
    )

    texto = corrigir_espacamento(texto)
    texto = corrigir_concatenacoes_conhecidas(texto)
    texto = remover_chamadas_finais(texto)

    # Remove espaços antes da pontuação e normaliza espaços repetidos.
    texto = re.sub(r"\s+([,;:.!?])", r"\1", texto)
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def normalizar_para_comparacao(texto: str) -> str:
    """Normaliza texto para identificar duplicatas."""

    texto = unicodedata.normalize("NFKC", texto)
    texto = texto.casefold()
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def gerar_dados_limpos(
    caminho_entrada="dados/noticias_brutas.json",
    caminho_saida="dados/noticias_limpas.json",
    min_caracteres=40,
):
    """Valida, limpa e salva as notícias em formato JSON."""

    caminho_entrada = Path(caminho_entrada)
    caminho_saida = Path(caminho_saida)

    if (
        isinstance(min_caracteres, bool)
        or not isinstance(min_caracteres, int)
        or min_caracteres < 0
    ):
        raise ValueError(
            "min_caracteres deve ser um inteiro não negativo."
        )

    if not caminho_entrada.is_file():
        raise FileNotFoundError(
            f"Arquivo de entrada não encontrado: {caminho_entrada}"
        )

    with caminho_entrada.open("r", encoding="utf-8") as arquivo:
        noticias = json.load(arquivo)

    if not isinstance(noticias, list):
        raise ValueError(
            "O arquivo de entrada deve conter uma lista JSON."
        )

    dados_limpos = []
    chaves_vistas = set()

    invalidas = 0
    curtas = 0
    duplicadas = 0

    for noticia in noticias:
        if not isinstance(noticia, dict):
            invalidas += 1
            continue

        if "id" not in noticia or not noticia.get("titulo"):
            invalidas += 1
            continue

        titulo = limpar_texto(noticia.get("titulo"))
        texto_limpo = limpar_texto(noticia.get("texto"))

        if not titulo:
            invalidas += 1
            continue

        if len(texto_limpo) < min_caracteres:
            curtas += 1
            continue

        chave = (
            normalizar_para_comparacao(titulo),
            normalizar_para_comparacao(texto_limpo),
        )

        if chave in chaves_vistas:
            duplicadas += 1
            continue

        chaves_vistas.add(chave)

        dados_limpos.append(
            {
                "id": noticia["id"],
                "titulo": titulo,
                "texto_limpo": texto_limpo,
                "data": noticia.get("data", ""),
                "fonte": noticia.get("fonte", ""),
            }
        )

    if not dados_limpos:
        raise ValueError(
            "Nenhuma notícia válida restou após a limpeza. "
            "Revise os dados de entrada e o limite de caracteres."
        )

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    with caminho_saida.open("w", encoding="utf-8") as arquivo:
        json.dump(
            dados_limpos,
            arquivo,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== Relatório da limpeza ===")
    print(f"Registros recebidos: {len(noticias)}")
    print(f"Registros inválidos: {invalidas}")
    print(f"Textos muito curtos: {curtas}")
    print(f"Duplicatas removidas: {duplicadas}")
    print(f"Notícias salvas: {len(dados_limpos)}")
    print(f"Arquivo gerado: {caminho_saida}")


if __name__ == "__main__":
    gerar_dados_limpos()