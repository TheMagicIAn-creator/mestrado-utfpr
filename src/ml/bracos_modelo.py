"""
bracos_modelo.py — Al IAdo PV / os DOIS braços do cerne da dissertação

Registro único dos modelos comparados: o **Autoencoder DENSO proposto** e o
**AE-LSTM temporal de Ibrahim (2022)**.

POR QUE ESTE MÓDULO EXISTE
==========================
O pesquisador foi explícito em 15/08/2026: *"Dividir o pipeline em Autoencoder
Denso (e seus resultados) e o Autoencoder LSTM (e seus resultados). Não quero
mistura entre eles, exceto que eu peça."*

O estado anterior misturava de três formas:

1. **Na pasta.** `macro_proposto` e `macro_ibrahim` gravavam os dois em
   `resultados/macro/`, distinguidos só por prefixo de arquivo
   (`proposto_tabela.md`, `ibrahim_tabela.md`). Pedir "os resultados do LSTM"
   obrigava a filtrar por nome.
2. **No nome do modelo.** A identidade vivia como string literal repetida em
   `macro_proposto.NOME`, `macro_ibrahim.NOME`, no dicionário `SLUG` de
   `macro_weibull` e nos rótulos de `contracts.py`. Quatro cópias que já
   divergiram uma vez.
3. **No conceito.** O AE-LSTM não tinha braço: existia só como efeito colateral
   dos macro-códigos, treinado no bloco de calibração e descartado. Não havia
   "os resultados do LSTM" para pedir.

Aqui a identidade de cada braço é declarada UMA vez: id, rótulo, cor, pasta de
resultados e como construir o seu detector. Quem precisar de qualquer uma dessas
coisas pergunta ao registro.

O QUE É CRUZAMENTO E O QUE NÃO É
================================
Resultado de braço é sempre de UM modelo, e mora na pasta dele. Comparação é
outra coisa: mora em `PASTA_COMPARACAO`, e só é produzida quando pedida
explicitamente. Essa separação é o que permite ao agente responder "os
resultados do denso" sem arrastar o LSTM junto — e, quando a pergunta for
comparativa, saber que precisa dos dois e ir buscá-los.

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).parent.parent.parent
PASTA_MODELOS = RAIZ / "resultados" / "modelos"
PASTA_COMPARACAO = RAIZ / "resultados" / "comparacao"

# Checkpoint do detector do pipeline. O braço denso o LÊ congelado; o braço LSTM
# reaproveita dele apenas o scaler, as colunas e a normalização de baseline —
# nunca os pesos, que são de outra arquitetura.
PASTA_AE = RAIZ / "resultados" / "autoencoder"


@dataclass(frozen=True)
class BracoModelo:
    """Um dos dois modelos comparados, com tudo que o identifica."""

    id: str
    nome: str
    cor: str
    arquitetura: str
    papel: str
    # Nome histórico gravado dentro dos artefatos já publicados. Existe só para
    # que uma tabela antiga continue sendo reconhecida como deste braço; código
    # novo usa `nome`.
    nomes_historicos: tuple[str, ...] = field(default=())

    @property
    def pasta(self) -> Path:
        """Onde os resultados DESTE braço moram. Nunca compartilhada."""
        return PASTA_MODELOS / self.id

    def caminho(self, arquivo: str) -> Path:
        return self.pasta / arquivo

    def reconhece(self, nome: str) -> bool:
        """O nome citado num artefato pertence a este braço?"""
        return str(nome) in {self.nome, self.id, *self.nomes_historicos}


DENSO = BracoModelo(
    id="ae_denso",
    nome="Autoencoder denso (proposto)",
    cor="#2a78d6",
    arquitetura="autoencoder denso por janela, MSE de reconstrução",
    papel=(
        "método proposto pela dissertação: cada janela de ~20 ms é reduzida a "
        "24 features espectrais orientadas pela FMECA e reconstruída por um "
        "autoencoder denso treinado só em operação saudável"
    ),
    nomes_historicos=(
        "Proposto (AE denso + MSE p99)",
        "Proposto (AE denso + escore localizado experimental)",
        "proposto",
    ),
)

LSTM = BracoModelo(
    id="ae_lstm",
    nome="AE-LSTM temporal (Ibrahim, 2022)",
    cor="#1baf7a",
    arquitetura="autoencoder LSTM sobre sequências de janelas, MSE no último passo",
    papel=(
        "braço comparativo do cerne: encoder/decoder recorrentes que capturam a "
        "correlação na SÉRIE, arquitetura que a literatura assume superior por "
        "esse motivo (Ibrahim et al., Energies 15:1082, §3.1)"
    ),
    nomes_historicos=("Ibrahim 2022 (AE-LSTM temporal)", "ibrahim"),
)

BRACOS: tuple[BracoModelo, ...] = (DENSO, LSTM)


def por_id(identificador: str) -> BracoModelo:
    """Braço pelo id, aceitando também o rótulo e os nomes históricos."""
    alvo = str(identificador).strip()
    for braco in BRACOS:
        if braco.reconhece(alvo):
            return braco
    validos = ", ".join(b.id for b in BRACOS)
    raise KeyError(
        f"Braço de modelo desconhecido: {identificador!r}. Válidos: {validos}."
    )


def identificar(nome_no_artefato: str) -> BracoModelo | None:
    """Descobre o braço de um artefato já publicado, sem levantar erro.

    Artefatos antigos gravam o nome longo do método; os novos gravam o `id`.
    Devolve None quando o nome não pertence a nenhum braço — o chamador decide
    se isso é erro ou apenas um artefato de outra natureza.
    """
    try:
        return por_id(nome_no_artefato)
    except KeyError:
        return None


def preparar_pastas() -> dict[str, Path]:
    """Cria a pasta de cada braço e a de comparação. Idempotente."""
    caminhos = {braco.id: braco.pasta for braco in BRACOS}
    caminhos["comparacao"] = PASTA_COMPARACAO
    for caminho in caminhos.values():
        caminho.mkdir(parents=True, exist_ok=True)
    return caminhos


def construir_scorer(braco: BracoModelo, janelas_calibracao: list,
                     contexto_lstm: str | None = None):
    """Detector pronto para pontuar, para QUALQUER braço.

    Interface única: ``Callable[[list[DataFrame]], np.ndarray]`` — a mesma que
    `macro_comum` e `weibull_por_modelo` já exigem. É o que permite os dois
    braços passarem pelo mesmo protocolo sem adaptador.

    O denso vem CONGELADO do disco: é o detector da dissertação, e retreiná-lo
    aqui inventaria um modelo que nenhum manifesto registra. O AE-LSTM é
    treinado no bloco de CALIBRAÇÃO — nunca nas janelas de avaliação, senão a
    comparação mediria vazamento.
    """
    import torch

    from src.core.seguranca import carregar_pickle_com_sidecar
    from src.ml.gpvs_principal import carregar_normalizacao_baseline

    if braco is DENSO:
        from src.ml import macro_proposto

        return macro_proposto.construir_scorer(macro_proposto.carregar_detector())

    if braco is LSTM:
        from src.ml import macro_ibrahim

        ckpt = torch.load(
            PASTA_AE / "modelo_autoencoder.pt", map_location="cpu",
            weights_only=False,
        )
        scaler = carregar_pickle_com_sidecar(PASTA_AE / "scaler.pkl")
        colunas = ckpt["colunas_feat"]
        normalizacao = carregar_normalizacao_baseline(PASTA_AE)

        X_cal = macro_ibrahim.features_das_janelas(
            janelas_calibracao, colunas, scaler, normalizacao
        )
        modelo = macro_ibrahim.treinar_detector(X_cal)
        return macro_ibrahim.construir_scorer(
            modelo, X_cal, colunas, scaler, normalizacao,
            contexto=contexto_lstm or macro_ibrahim.CONTEXTO_NORMAL,
        )

    raise KeyError(f"Braço sem construtor de detector: {braco.id}")


def resumo_markdown() -> str:
    """Quadro dos dois braços, para o agente explicar a separação."""
    linhas = ["| Braço | Arquitetura | Resultados em |", "|---|---|---|"]
    for braco in BRACOS:
        relativo = braco.pasta.relative_to(RAIZ).as_posix()
        linhas.append(f"| **{braco.nome}** | {braco.arquitetura} | `{relativo}/` |")
    linhas.append("")
    linhas.append(
        f"Comparações entre os dois moram em "
        f"`{PASTA_COMPARACAO.relative_to(RAIZ).as_posix()}/` e só são geradas "
        f"quando pedidas."
    )
    return "\n".join(linhas)
