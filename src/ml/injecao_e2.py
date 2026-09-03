"""Injeção sintética E2 sobre o sinal saudável, ancorada nos ensaios reais.

POR QUE ESTA ETAPA EXISTE
========================
A E3 responde "cada modelo detecta esta falha?" com dado real, e responde de
forma BINÁRIA: o ensaio tem falha ou não tem. O que ela não dá é o eixo de
severidade — "a partir de que magnitude a detecção começa" —, e é esse eixo
que alimenta `a_det`, a POD e a discussão de manutenção.

Para ter severidade é preciso um botão contínuo, e o GPVS não tem: cada ensaio
foi gravado numa condição fixa. Daí a injeção. Ela NÃO substitui a E3 e não
compete com ela; ela acrescenta uma dimensão que o dado real não carrega.

O QUE `a` SIGNIFICA
===================
`a` é a fração da assinatura nominal, em [0, 1]. **Não é tempo.** `a=0` é a
janela saudável intacta; `a=1` é a assinatura completa, definida para cada item
pelo próprio contrato do GPVS (falha completa de IGBT, erro de 20% no sensor,
ganho do PI reduzido em 20%). Ler `a_det` como vida útil, RUL ou taxa de falha
é erro de categoria — ver `docs/nomenclatura_deteccao.md`.

DOIS MÉTODOS, E A DIFERENÇA IMPORTA
===================================
A FMECA vigente cobre três itens, e eles não são igualmente sintetizáveis:

  IGBT e sensor/realimentação têm assinatura elétrica direta. A falha se
  escreve no sinal, e a física diz como. `injection_method` =
  `electrical_signature`.

  Ganho e constante de tempo do controlador PI mudam a DINÂMICA DE MALHA
  FECHADA. Não há manipulação post-hoc de um sinal já gravado que reproduza
  isso com fidelidade — qualquer tentativa seria caricatura apresentada como
  física. Para esse item, `a` percorre o caminho entre dois estados MEDIDOS,
  o saudável e o ensaio real de controle. `injection_method` =
  `measured_state_interpolation`.

O método viaja em cada linha publicada. Um leitor que veja `a_det` do controle
tem de conseguir saber, sem perguntar, que aquilo não é simulação física.

A ÂNCORA
========
Toda injeção é uma hipótese sobre como a falha se parece. `distancia_ancora`
testa essa hipótese contra a medição: em `a=1`, a janela injetada deveria cair
perto do ensaio real correspondente no espaço das 24 features. A distância sai
em unidades do IQR saudável, então é lida como "tantos sigmas robustos".

Distância grande não invalida a varredura — significa que a assinatura injetada
é uma caricatura da falha real, e isso precisa aparecer no artefato em vez de
ser descoberto na defesa. É a âncora que a implementação anterior nunca teve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.ml.dados_gpvs import (
    CURRENT_COLUMNS,
    FAULT_CONTRACTS,
    FEATURE_COLUMNS,
    PRIMARY_COLUMNS,
    WINDOW_SAMPLES,
    feature_vector,
)

# Grade de severidade. Começa acima de zero porque `a=0` é a janela intacta e
# não pertence à varredura; termina em 1,0 porque é a assinatura nominal
# completa, e extrapolar além dela não teria referência no contrato do GPVS.
GRADE_SEVERIDADE = tuple(np.round(np.linspace(0.02, 1.0, 50), 4).tolist())

# Erro nominal do sensor de realimentação no GPVS: 20%.
ERRO_NOMINAL_SENSOR = 0.20

METODO_ASSINATURA = "electrical_signature"
METODO_INTERPOLACAO = "measured_state_interpolation"


@dataclass(frozen=True)
class EspecificacaoInjecao:
    """Um item da FMECA vigente e como sua severidade é construída."""

    id: str
    nome: str
    fmeca_scope: str
    ensaios_reais: tuple[str, ...]
    metodo: str
    significado_de_a1: str
    fundamentacao: str

    @property
    def exige_ensaio_real(self) -> bool:
        """Interpolação precisa do outro extremo; assinatura não."""
        return self.metodo == METODO_INTERPOLACAO


ESPECIFICACOES = (
    EspecificacaoInjecao(
        id="igbt",
        nome="IGBT",
        fmeca_scope="igbt",
        ensaios_reais=("F1L", "F1M"),
        metodo=METODO_ASSINATURA,
        significado_de_a1="semiciclo positivo de uma fase integralmente suprimido",
        fundamentacao=(
            "Falha completa de um IGBT interrompe a condução de uma perna em "
            "um dos semiciclos. Num inversor conectado à rede quem impõe a "
            "tensão é a rede, então a assinatura aparece na CORRENTE de fase e "
            "não na tensão: por isso a injeção não toca va/vb/vc."
        ),
    ),
    EspecificacaoInjecao(
        id="sensor_realimentacao",
        nome="Sistema de sensor/realimentação",
        fmeca_scope="sensor_feedback_system",
        ensaios_reais=("F2L", "F2M"),
        metodo=METODO_ASSINATURA,
        significado_de_a1=f"erro de ganho de {ERRO_NOMINAL_SENSOR:.0%} nas três fases",
        fundamentacao=(
            "O contrato do GPVS define a falha 2 como erro de 20% no sistema "
            "de sensor/realimentação. Medindo errado, a malha leva a corrente "
            "real a um ponto de operação deslocado pelo mesmo fator. O erro é "
            "do sistema de medição, logo atinge as três fases igualmente: o "
            "desbalanceamento NÃO se move, e é isso que separa esta assinatura "
            "da do IGBT."
        ),
    ),
    EspecificacaoInjecao(
        id="controle",
        nome="Sistema/circuito de controle do inversor",
        fmeca_scope="inverter_control_system",
        ensaios_reais=("F6L", "F6M", "F7L", "F7M"),
        metodo=METODO_INTERPOLACAO,
        significado_de_a1="o próprio ensaio real de anomalia de controle",
        fundamentacao=(
            "Ganho do PI reduzido em 20% e constante de tempo elevada em 20% "
            "alteram a dinâmica de malha fechada. Nenhuma manipulação de um "
            "sinal já gravado reproduz isso: seria caricatura apresentada como "
            "física. Aqui `a` percorre o caminho entre dois estados MEDIDOS, e "
            "a interpolação não alega simular o controlador — alega apenas ser "
            "monótona entre saudável e falho."
        ),
    ),
)

POR_ID = {especificacao.id: especificacao for especificacao in ESPECIFICACOES}


def _validar_janela(janela: pd.DataFrame) -> pd.DataFrame:
    faltando = [coluna for coluna in PRIMARY_COLUMNS if coluna not in janela.columns]
    if faltando:
        raise ValueError(f"Janela sem colunas primárias: {faltando}")
    if len(janela) != WINDOW_SAMPLES:
        raise ValueError(
            f"Janela deve ter {WINDOW_SAMPLES} amostras; recebeu {len(janela)}"
        )
    return janela


def _validar_severidade(a: float) -> float:
    valor = float(a)
    if not 0.0 <= valor <= 1.0:
        raise ValueError(
            f"A severidade `a` é fração da assinatura nominal e vive em [0, 1]; "
            f"recebeu {valor}. Ela não é tempo e não se extrapola."
        )
    return valor


# ── as três injeções ───────────────────────────────────────────────────────

def injetar_igbt(janela: pd.DataFrame, a: float) -> pd.DataFrame:
    """Suprime progressivamente o semiciclo positivo de `ia`.

    Em `a=1` a fase perde metade do ciclo, que é a falha completa. A tensão
    não é tocada: num inversor conectado à rede é a rede que a impõe.
    """
    _validar_janela(janela)
    severidade = _validar_severidade(a)
    resultado = janela.copy()
    # `to_numpy` pode devolver uma VISTA somente-leitura do bloco do DataFrame;
    # sem a cópia, a atribuição mascarada abaixo estoura.
    corrente = np.array(resultado["ia"].to_numpy(dtype=float), copy=True)
    positivo = corrente > 0.0
    corrente[positivo] = corrente[positivo] * (1.0 - severidade)
    resultado["ia"] = corrente
    return resultado


def injetar_sensor_realimentacao(janela: pd.DataFrame, a: float) -> pd.DataFrame:
    """Erro de ganho igual nas três fases, até 20% em `a=1`.

    Por atingir as três igualmente, `i_rms_unbalance` fica onde estava e a THD
    também — ganho é transformação linear. O que se move é RMS e potência.
    """
    _validar_janela(janela)
    severidade = _validar_severidade(a)
    resultado = janela.copy()
    fator = 1.0 + ERRO_NOMINAL_SENSOR * severidade
    for coluna in CURRENT_COLUMNS:
        resultado[coluna] = resultado[coluna].to_numpy(dtype=float) * fator
    return resultado


def injetar_controle(
    janela: pd.DataFrame,
    a: float,
    *,
    janela_falha: pd.DataFrame,
) -> pd.DataFrame:
    """Caminho linear entre a janela saudável e uma janela de falha MEDIDA.

    Em `a=0` devolve a saudável; em `a=1`, a de falha. Não simula o
    controlador: interpola entre duas medições.
    """
    _validar_janela(janela)
    _validar_janela(janela_falha)
    severidade = _validar_severidade(a)
    resultado = janela.copy()
    for coluna in PRIMARY_COLUMNS:
        saudavel = janela[coluna].to_numpy(dtype=float)
        falha = janela_falha[coluna].to_numpy(dtype=float)
        resultado[coluna] = (1.0 - severidade) * saudavel + severidade * falha
    return resultado


def injetor_de(
    especificacao: EspecificacaoInjecao,
    *,
    janela_falha: pd.DataFrame | None = None,
) -> Callable[[pd.DataFrame, float], pd.DataFrame]:
    """Devolve a função de injeção já amarrada ao que ela precisa.

    A interpolação exige o outro extremo. Pedi-lo aqui, e não no meio da
    varredura, faz a ausência estourar antes de qualquer número sair.
    """
    if especificacao.id == "igbt":
        return injetar_igbt
    if especificacao.id == "sensor_realimentacao":
        return injetar_sensor_realimentacao
    if especificacao.id == "controle":
        if janela_falha is None:
            raise ValueError(
                "A injeção de controle é interpolação entre estados medidos e "
                "exige `janela_falha` de um ensaio real F6/F7. Sem ela não há "
                "extremo superior, e `a=1` não significaria nada."
            )
        def _interpolar(janela: pd.DataFrame, a: float) -> pd.DataFrame:
            return injetar_controle(janela, a, janela_falha=janela_falha)

        return _interpolar
    raise KeyError(f"Especificação sem injetor: {especificacao.id}")


# ── a âncora ───────────────────────────────────────────────────────────────

def distancia_ancora(
    janelas_injetadas: list[pd.DataFrame],
    janelas_reais: list[pd.DataFrame],
    escala_saudavel: np.ndarray,
) -> dict:
    """Quão longe a assinatura em `a=1` cai do ensaio real, em IQR saudável.

    Compara as MEDIANAS das duas nuvens no espaço das 24 features. A mediana,
    e não a média, porque uma única janela de transição não pode dominar.

    Uma distância grande é informação, não falha: diz que a assinatura injetada
    é uma caricatura da falha medida. O lugar de descobrir isso é o artefato.
    """
    if not janelas_injetadas or not janelas_reais:
        raise ValueError("A âncora exige janelas dos dois lados")
    escala = np.asarray(escala_saudavel, dtype=float)
    if escala.shape != (len(FEATURE_COLUMNS),):
        raise ValueError(
            f"A escala deve ter {len(FEATURE_COLUMNS)} entradas, uma por feature"
        )
    if not np.all(escala > 0.0):
        raise ValueError("A escala saudável precisa ser estritamente positiva")

    injetado = np.median(
        np.asarray([feature_vector(j) for j in janelas_injetadas], dtype=float), axis=0
    )
    real = np.median(
        np.asarray([feature_vector(j) for j in janelas_reais], dtype=float), axis=0
    )
    por_feature = np.abs(injetado - real) / escala
    return {
        "distancia_euclidiana_iqr": float(np.linalg.norm(por_feature)),
        "distancia_mediana_iqr": float(np.median(por_feature)),
        "distancia_maxima_iqr": float(np.max(por_feature)),
        "feature_mais_distante": FEATURE_COLUMNS[int(np.argmax(por_feature))],
        "n_injetadas": len(janelas_injetadas),
        "n_reais": len(janelas_reais),
        "unidade": "IQR do bloco saudável de treino",
    }


def contrato_da_especificacao(especificacao: EspecificacaoInjecao) -> dict:
    """O que acompanha cada linha publicada da varredura."""
    escopos = {
        FAULT_CONTRACTS[int(ensaio[1])]["fmeca_scope"]
        for ensaio in especificacao.ensaios_reais
    }
    if escopos != {especificacao.fmeca_scope}:
        raise AssertionError(
            f"{especificacao.id} declara escopo {especificacao.fmeca_scope} mas "
            f"seus ensaios reais apontam para {escopos}"
        )
    return {
        "injection_id": especificacao.id,
        "injection_name": especificacao.nome,
        "fmeca_scope": especificacao.fmeca_scope,
        "reference_experiments": list(especificacao.ensaios_reais),
        "injection_method": especificacao.metodo,
        "a_axis": "fraction_of_nominal_signature_not_time",
        "a1_meaning": especificacao.significado_de_a1,
        "rationale": especificacao.fundamentacao,
        "evidence_level": "E2",
        "physical_simulation": especificacao.metodo == METODO_ASSINATURA,
    }


__all__ = [
    "ERRO_NOMINAL_SENSOR",
    "ESPECIFICACOES",
    "EspecificacaoInjecao",
    "GRADE_SEVERIDADE",
    "METODO_ASSINATURA",
    "METODO_INTERPOLACAO",
    "POR_ID",
    "contrato_da_especificacao",
    "distancia_ancora",
    "injetar_controle",
    "injetar_igbt",
    "injetar_sensor_realimentacao",
    "injetor_de",
]
