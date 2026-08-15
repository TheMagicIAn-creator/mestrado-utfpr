"""
macro_weibull.py — Al IAdo PV / MACRO-CÓDIGO 3: detectabilidade POR MODELO

Papel de probabilidade, confiabilidade, densidade/acumulada e intensidade —
as quatro curvas — para **cada** detector comparado, sobre o GPVS-Faults.

POR QUE ESTE MÓDULO EXISTE
==========================
O pesquisador pediu, literalmente: *"a distribuição, que é aquela distribuição
com uma linha reta, com vários pontinhos de dispersão, a curva de
confiabilidade, a curva de falha, curva de taxas de falha, etcétera, pertinente
a cada modelo"*, comparando o AE denso com o AE-LSTM do Ibrahim.

Não saía. A cadeia `rul_weibull_execucao` carrega UM checkpoint fixo de
`resultados/autoencoder/` e itera sobre `FALHAS` — o laço é por COMPONENTE,
nunca por MODELO. Os quatro macro-códigos, do outro lado, paravam em AUC, SMD e
matriz de confusão: grep por "weibull" neles devolvia zero.

`weibull_por_modelo` destravou a capacidade (varredura de magnitude com scorer
plugável). Este módulo é a ORQUESTRAÇÃO que faltava: monta os dois detectores,
varre, ajusta e desenha.

O QUE É COMPARTILHADO E O QUE É POR MODELO
==========================================
Compartilhado, e tem de ser, senão a comparação mede outra coisa:
  - o holdout F0 do GPVS e a divisão calibração/avaliação (`macro_comum`);
  - as janelas que viram trajetória (`selecionar_trajetorias_holdout`);
  - as realizações de ruído da injeção (semente por índice de janela).

Por modelo, e tem de ser:
  - o LIMIAR. O MSE de um autoencoder denso e o de um AE-LSTM vivem em escalas
    diferentes; um limiar único compararia unidades. Cada um é calibrado ao
    MESMO alvo de FP, no MESMO bloco, por `macro_comum.calibrar_limiar`.

CUSTO
=====
A varredura é o passo caro: até `N_STEPS` inferências por trajetória, por falha,
por modelo — com parada antecipada quando a detecção confirma. Por isso este
macro é um ponto de entrada SEPARADO, e não um apêndice automático do
`macro_comparar`. Ajuste com `--n-steps` e `--n-trajetorias` antes de rodar a
grade cheia.

O QUE ELE NÃO FAZ
=================
Não promove nada a E3 e não converte magnitude em tempo. `a_det` é fração da
assinatura nominal; `S_D` é probabilidade de ainda não detectar e `h_D` é
intensidade de primeiro cruzamento. Nenhuma das duas é confiabilidade ou taxa
de falha física do componente.

Uso:
  python -m src.ml.macro_weibull
  python -m src.ml.macro_weibull --n-steps 101 --n-trajetorias 60   # ensaio rápido

Saídas: resultados/macro/weibull/{proposto,ibrahim}/weibull_*.png
        resultados/macro/weibull/detectabilidade_por_modelo.{json,csv,md}
        resultados/macro/weibull/comparacao_confiabilidade.png
Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

try:
    from src.core.logs import adaptar_logger_como_print as _adaptar_log
    from src.core.logs import get_logger as _get_logger
except ModuleNotFoundError:  # execução direta
    import sys as _sys
    from pathlib import Path as _Path
    _raiz = str(_Path(__file__).resolve().parents[2])
    if _raiz not in _sys.path:
        _sys.path.insert(0, _raiz)
    from src.core.logs import adaptar_logger_como_print as _adaptar_log
    from src.core.logs import get_logger as _get_logger

_logger = _get_logger("macro_weibull")
_log = _adaptar_log(_logger)


import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).parent.parent.parent
PASTA_SAIDA = RAIZ / "resultados" / "macro" / "weibull"

def _pasta_do_modelo(nome: str) -> Path:
    """Pasta do BRAÇO, não uma subpasta deste macro.

    Era um dicionário `SLUG` local — a quarta cópia da identidade dos modelos.
    Agora vem de `bracos_modelo`, que é a fonte única: id, rótulo, cor e pasta
    nascem juntos e não podem divergir.

    O nome do arquivo dentro da pasta é o MESMO das figuras do pipeline
    principal (weibull_confiabilidade.png etc.) de propósito: o leitor compara
    figuras homônimas em pastas distintas, sem traduzir nomes.
    """
    from src.ml.bracos_modelo import identificar

    braco = identificar(nome)
    if braco is not None:
        return braco.pasta / "weibull"
    # Braço não registrado (ablação exploratória): pasta própria e estável, para
    # não colidir com os dois oficiais.
    slug = "".join(c if c.isalnum() else "_" for c in nome.lower())[:40]
    return PASTA_SAIDA / slug


# ============================================================
# ETAPA 1 — os dois detectores, sobre o MESMO holdout
# ============================================================

def montar_detectores(janelas_calib: list, bracos=None) -> list[dict]:
    """Constrói os scorers dos braços pedidos — por padrão, os dois.

    Nome, cor e construção do detector vêm todos de `bracos_modelo`. Antes
    estavam repetidos aqui como literais, e essa era a quarta cópia da
    identidade dos modelos.

    Aceitar `bracos` é o que atende ao pedido de não misturar: rodar só o denso
    é `montar_detectores(j_cal, [DENSO])`, e nada do LSTM é treinado nem gravado.
    """
    from src.ml.bracos_modelo import BRACOS, construir_scorer

    return [
        {
            "braco": braco,
            "nome": braco.nome,
            "cor": braco.cor,
            "scorer": construir_scorer(braco, janelas_calib),
        }
        for braco in (bracos if bracos is not None else BRACOS)
    ]


# ============================================================
# ETAPA 2 — as figuras, reusando os plots do pipeline principal
# ============================================================

def _dicts_para_plot(bloco: dict) -> tuple[dict, dict, dict]:
    """Converte o bloco por modelo no formato que `graficos_rul` já consome."""
    a_dets = {fid: np.asarray(d["a_dets"], dtype=float)
              for fid, d in bloco["falhas"].items()}
    eventos = {fid: np.asarray(d["eventos_observados"], dtype=bool)
               for fid, d in bloco["falhas"].items()}
    params = {fid: d["weibull"] for fid, d in bloco["falhas"].items()}
    return a_dets, eventos, params


def desenhar_modelo(bloco: dict) -> dict:
    """As quatro figuras pedidas, para UM modelo.

    Reusa `graficos_rul` sem tocar em nada: as funções já recebem
    ``(a_dets, eventos, params, pasta)`` e escrevem na pasta que receberem.
    Reimplementá-las aqui criaria uma segunda fonte para a mesma curva — que é
    o problema que o projeto já teve com `confiabilidade_fisica_v2`.
    """
    from src.ml.graficos_rul import (
        plotar_confiabilidade,
        plotar_distribuicao_weibull,
        plotar_funcoes_distribuicao_weibull,
        plotar_intensidade_deteccao,
    )

    pasta = _pasta_do_modelo(bloco["modelo"])
    pasta.mkdir(parents=True, exist_ok=True)
    a_dets, eventos, params = _dicts_para_plot(bloco)

    plotar_distribuicao_weibull(a_dets, eventos, params, pasta)
    plotar_confiabilidade(a_dets, eventos, params, pasta)
    plotar_funcoes_distribuicao_weibull(a_dets, eventos, params, pasta)
    plotar_intensidade_deteccao(a_dets, eventos, params, pasta)

    return {
        "papel_weibull": pasta / "weibull_distribuicao.png",
        "confiabilidade": pasta / "weibull_confiabilidade.png",
        "funcoes_distribuicao": pasta / "weibull_funcoes_distribuicao.png",
        "intensidade": pasta / "weibull_intensidade_deteccao.png",
    }


def plotar_comparacao_confiabilidade(blocos: list[dict]) -> Path:
    """Sobrepõe a S_D(a) dos modelos, um painel por falha da FMECA.

    Kaplan-Meier, não a Weibull ajustada, e isso é deliberado: o ajuste 2P vem
    sendo REJEITADO pelo teste de aderência quantizada nas execuções atuais, e
    desenhar a reta paramétrica como se fosse o resultado seria afirmar mais do
    que o dado sustenta. A curva empírica é o que se pode defender na banca; a
    paramétrica entra tracejada, e só quando o próprio ajuste se recomenda.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.ml.confiabilidade import confiabilidade, curva_kaplan_meier
    from src.ml.estilo_graficos import TAM, aplicar_estilo, salvar_figura
    from src.ml.injecao_falhas import FALHAS

    aplicar_estilo()
    fig, axes = plt.subplots(1, len(FALHAS), figsize=TAM["painel_3"],
                             layout="constrained", sharey=True)
    if len(FALHAS) == 1:
        axes = [axes]
    fig.suptitle(
        "Probabilidade de AINDA NÃO detectar por modelo — S_D(a), não "
        "confiabilidade física"
    )

    estilos = ["-", "--", ":", "-."]
    for ax, falha in zip(axes, FALHAS):
        fid = falha["id"]
        for i, bloco in enumerate(blocos):
            dados = bloco["falhas"][fid]
            a_det = np.asarray(dados["a_dets"], dtype=float)
            eventos = np.asarray(dados["eventos_observados"], dtype=bool)
            cor = bloco.get("cor", f"C{i}")

            km_a, km_s = curva_kaplan_meier(a_det, eventos)
            pod = dados["desfechos"]["pod_mon_no_teto"]
            ax.step(km_a, km_s, where="post", color=cor, linewidth=1.9,
                    linestyle=estilos[i % len(estilos)],
                    label=f"{bloco['modelo']} — POD_mon={pod:.2f}")

            p = dados["weibull"]
            if p.get("fit_converged") and p.get("resumo_parametrico_recomendado"):
                grade = np.linspace(max(float(a_det.min()), 1e-6), 1.0, 300)
                ax.plot(grade, confiabilidade(grade, p["beta"], p["eta"]),
                        color=cor, linewidth=1.1, alpha=0.55,
                        label=f"{bloco['modelo']} — Weibull 2P")

        ax.set_title(f"{falha['nome']} (NPR={falha['npr']})", fontsize=10)
        ax.set_xlabel("a — fração da assinatura nominal injetada")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("S_D(a) = P(ainda não detectado)")
    axes[0].legend(fontsize=7.5, loc="lower left")

    caminho = PASTA_SAIDA / "comparacao_confiabilidade.png"
    salvar_figura(
        fig, caminho,
        "E2 sintético (injeção FMECA no sinal). Curva empírica Kaplan-Meier; a "
        "Weibull 2P só aparece quando o teste de aderência a recomenda. Eixo é "
        "MAGNITUDE, não tempo — curva mais baixa = detecta com menos assinatura.",
    )
    return caminho


# ============================================================
# ETAPA 3 — tabela comparável
# ============================================================

def tabela_markdown(comparacao: dict) -> str:
    """Tabela por modelo × falha, nos marcos que decidem manutenção."""
    linhas = [
        "| Modelo | Falha (NPR) | n | detectadas | POD_mon@a=1 | a10 | "
        "a_det mediana | 2P adotada |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in comparacao["linhas"]:
        def _num(valor, casas=3):
            return f"{valor:.{casas}f}" if isinstance(valor, (int, float)) \
                and valor == valor else "—"

        linhas.append(
            f"| {r['modelo']} | {r['falha']} (NPR={r['npr']}) | "
            f"{r['n_trajetorias']} | {r['detectadas']} | "
            f"{r['pod_mon_no_teto']:.2f} | {_num(r.get('a10'))} | "
            f"{_num(r.get('a_det_mediana'))} | "
            f"{'sim' if r.get('resumo_parametrico_recomendado') else 'não'} |"
        )
    return "\n".join(linhas)


def salvar_saidas(blocos: list[dict], comparacao: dict) -> dict:
    import csv

    from src.ml.rul_weibull import _json_seguro

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    arq_json = PASTA_SAIDA / "detectabilidade_por_modelo.json"
    arq_json.write_text(
        json.dumps(_json_seguro({"modelos": blocos, "comparacao": comparacao}),
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    arq_md = PASTA_SAIDA / "detectabilidade_por_modelo.md"
    arq_md.write_text(tabela_markdown(comparacao), encoding="utf-8")

    arq_csv = PASTA_SAIDA / "detectabilidade_por_modelo.csv"
    with arq_csv.open("w", newline="", encoding="utf-8") as fh:
        campos = ["modelo", "falha", "falha_id", "npr", "n_trajetorias",
                  "detectadas", "pod_mon_no_teto", "ajuste_convergiu",
                  "resumo_parametrico_recomendado", "a10", "a_det_mediana",
                  "beta", "eta", "evidence_level"]
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for linha in comparacao["linhas"]:
            w.writerow(linha)

    return {"json": arq_json, "tabela_md": arq_md, "tabela_csv": arq_csv}


# ============================================================
# PROVENIÊNCIA
# ============================================================

def _saidas_weibull() -> list[Path]:
    arquivos = [
        PASTA_SAIDA / "detectabilidade_por_modelo.json",
        PASTA_SAIDA / "detectabilidade_por_modelo.md",
        PASTA_SAIDA / "detectabilidade_por_modelo.csv",
        PASTA_SAIDA / "comparacao_confiabilidade.png",
    ]
    # As curvas moram na pasta do BRAÇO, não numa subpasta deste macro: é a
    # separação que o pesquisador pediu, e ela precisa aparecer no manifesto.
    from src.ml.bracos_modelo import BRACOS

    arquivos += [
        braco.pasta / "weibull" / nome
        for braco in BRACOS
        for nome in ("weibull_distribuicao.png", "weibull_confiabilidade.png",
                     "weibull_funcoes_distribuicao.png",
                     "weibull_intensidade_deteccao.png")
    ]
    return arquivos


def manifesto_atual(n_janelas: int | None = None,
                    n_steps: int | None = None) -> dict:
    """Descreve a varredura por modelo, para o artefato ser citável."""
    from src.ml.gpvs_principal import ARQUIVO_FEATURES, PASTA_GPVS
    from src.ml.injecao_falhas import N_JANELAS_SMD
    from src.ml.macro_comum import FRACAO_AJUSTE_LIMIAR, PURGA
    from src.ml.macro_ibrahim import EPOCHS, SEQ_LEN
    from src.ml.proveniencia import gerar_manifesto
    from src.ml.rul_weibull import (
        A_DET_MAX, A_DET_MIN, N_STEPS, PERSISTENCIA_MAGNITUDE,
    )
    from src.ml.weibull_por_modelo import SEED_TRAJETORIAS

    return gerar_manifesto(
        "macro_weibull",
        Path(__file__),
        {
            "n_janelas": int(n_janelas or N_JANELAS_SMD),
            "n_steps": int(n_steps or N_STEPS),
            "a_det_intervalo": [A_DET_MIN, A_DET_MAX],
            "persistencia_magnitude": PERSISTENCIA_MAGNITUDE,
            "seed_trajetorias": SEED_TRAJETORIAS,
            "purga": PURGA,
            "fracao_ajuste_limiar": FRACAO_AJUSTE_LIMIAR,
            "aelstm_seq_len": SEQ_LEN,
            "aelstm_epochs": EPOCHS,
        },
        {
            "dataset_gpvs_f0l": PASTA_GPVS / "F0L.csv",
            "dataset_gpvs_f0m": PASTA_GPVS / "F0M.csv",
            "features": ARQUIVO_FEATURES,
            "modelo_autoencoder": RAIZ / "resultados/autoencoder/modelo_autoencoder.pt",
            "scaler_autoencoder": RAIZ / "resultados/autoencoder/scaler.pkl",
            "limiar_autoencoder": RAIZ / "resultados/autoencoder/limiar.json",
        },
        _saidas_weibull(),
        code_dependencies={
            nome: RAIZ / caminho
            for nome, caminho in {
                "macro_weibull": "src/ml/macro_weibull.py",
                "weibull_por_modelo": "src/ml/weibull_por_modelo.py",
                "varredura_a_det": "src/ml/varredura_a_det.py",
                "rul_weibull": "src/ml/rul_weibull.py",
                "confiabilidade": "src/ml/confiabilidade.py",
                "graficos_rul": "src/ml/graficos_rul.py",
                "macro_comum": "src/ml/macro_comum.py",
                "macro_proposto": "src/ml/macro_proposto.py",
                "macro_ibrahim": "src/ml/macro_ibrahim.py",
                "injecao_falhas": "src/ml/injecao_falhas.py",
                "gpvs_principal": "src/ml/gpvs_principal.py",
            }.items()
        },
        evidence_level="E2",
    )


def registrar_manifesto(n_janelas: int | None = None,
                        n_steps: int | None = None) -> Path:
    from src.ml.proveniencia import salvar_manifesto

    return salvar_manifesto(manifesto_atual(n_janelas, n_steps))


# ============================================================
# ORQUESTRAÇÃO
# ============================================================

def executar(n_janelas: int | None = None, n_steps: int | None = None,
             n_trajetorias: int | None = None, n_boot: int = 0,
             bracos: list[str] | None = None) -> dict:
    from src.ml.gpvs_principal import preparar_janelas_holdout
    from src.ml.injecao_falhas import FALHAS, N_JANELAS_SMD
    from src.ml.macro_comum import (
        calibrar_limiar, conferir_escala_do_limiar,
        dividir_calibracao_avaliacao,
    )
    from src.ml.bracos_modelo import DENSO, por_id
    from src.ml.rul_weibull import N_STEPS, selecionar_trajetorias_holdout
    from src.ml.weibull_por_modelo import (
        comparar_detectabilidade, detectabilidade_do_modelo,
    )

    n_steps = int(n_steps or N_STEPS)

    _log("=" * 60)
    _log("  MACRO-CÓDIGO 3 — DETECTABILIDADE POR MODELO (E2)")
    _log("=" * 60)

    pasta_ae = RAIZ / "resultados" / "autoencoder"
    arq_modelo = pasta_ae / "modelo_autoencoder.pt"
    if not arq_modelo.exists():
        raise FileNotFoundError(
            "Autoencoder não treinado. Rode antes:\n"
            "  python -m src.ml.exec_etapa_isolada features_gpvs && "
            "python -m src.ml.exec_etapa_isolada autoencoder")

    _log("\n  Carregando holdout F0 do GPVS-Faults (teste isolado)...")
    janelas, _meta = preparar_janelas_holdout(n_max=n_janelas or N_JANELAS_SMD)
    j_cal, j_aval = dividir_calibracao_avaliacao(janelas)
    _log(f"  {len(janelas)} janelas | calibração={len(j_cal)} | "
         f"avaliação={len(j_aval)} (disjuntos)")

    # Sem `bracos`, roda os dois. Com um só, nada do outro e treinado nem
    # gravado -- e o pedido de nao misturar e atendido na raiz.
    selecionados = [por_id(b) for b in bracos] if bracos else None
    detectores = montar_detectores(j_cal, selecionados)

    blocos = []
    for detector in detectores:
        nome, scorer = detector["nome"], detector["scorer"]
        # Limiar POR MODELO, mesmo alvo de FP, mesmo bloco. Ver a docstring de
        # macro_comum.calibrar_limiar: escalas de escore não são comparáveis.
        limiar, percentil = calibrar_limiar(scorer, j_cal)
        _log(f"\n  {nome}")
        _log(f"    limiar = {limiar:.5f} (percentil {percentil:.1f})")
        if detector["braco"] is DENSO:
            alerta = conferir_escala_do_limiar(nome, limiar, pasta_ae)
            if alerta:
                _log(f"    {alerta}")
        n_traj = len(selecionar_trajetorias_holdout(j_aval, n_trajetorias))
        # Teto, não previsão: a parada antecipada corta a varredura no primeiro
        # cruzamento confirmado, então falha detectável custa uma fração disto.
        # O número existe para o pesquisador poder desistir ANTES de esperar.
        teto = n_traj * len(FALHAS) * n_steps
        _log(f"    varrendo magnitude em {n_traj} trajetórias × {len(FALHAS)} "
             f"falhas × {n_steps} passos")
        _log(f"    teto de {teto:,} pontuações de janela (parada antecipada "
             f"reduz muito); use --n-trajetorias para cortar")

        bloco = detectabilidade_do_modelo(
            nome, scorer, limiar, j_aval, n_steps=n_steps,
            n_max_trajetorias=n_trajetorias, n_boot=n_boot,
        )
        bloco["cor"] = detector["cor"]
        bloco["braco_id"] = detector["braco"].id
        bloco["percentil_limiar"] = float(percentil)
        blocos.append(bloco)

        for fid, dados in bloco["falhas"].items():
            d = dados["desfechos"]
            _log(f"      {dados['nome']:<14} POD_mon@a=1 = "
                 f"{d['pod_mon_no_teto']:.2f} "
                 f"({d['n_detectadas']}/{d['n_traj']} detectadas)")

        figuras = desenhar_modelo(bloco)
        for chave, caminho in figuras.items():
            _log(f"      figura {chave}: {Path(caminho).name}")

    comparacao = comparar_detectabilidade(blocos)
    caminho_overlay = plotar_comparacao_confiabilidade(blocos)
    saidas = salvar_saidas(blocos, comparacao)
    caminho_manifesto = registrar_manifesto(n_janelas, n_steps)

    _log("\n" + "=" * 60)
    _log("  DETECTABILIDADE POR MODELO")
    _log("=" * 60)
    _log("\n" + tabela_markdown(comparacao))
    _log(f"\n  Artefatos em {PASTA_SAIDA}:")
    for chave, caminho in {**saidas, "sobreposicao": caminho_overlay,
                           "manifesto": caminho_manifesto}.items():
        _log(f"    {chave}: {Path(caminho).name}")
    _log("\n  Leitura: a10 e a_det mediana MENORES = o modelo confirma a falha")
    _log("  com menos assinatura. POD_mon@a=1 é a fração detectada na magnitude")
    _log("  máxima — é o elo com a curva POD e com o D_mon da FMECA.")
    _log("  O eixo é MAGNITUDE, não tempo: nada aqui é vida do componente.")
    _log("=" * 60)
    return {"modelos": blocos, "comparacao": comparacao,
            "saidas": {**saidas, "sobreposicao": caminho_overlay}}


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--n-janelas", type=int, default=None,
                   help="teto de janelas do holdout (padrão: todas)")
    p.add_argument("--n-steps", type=int, default=None,
                   help="passos da grade de magnitude (padrão: 501)")
    p.add_argument("--n-trajetorias", type=int, default=None,
                   help="teto de trajetórias por falha (padrão: todas)")
    p.add_argument("--n-boot", type=int, default=0,
                   help="reamostragens do IC do ajuste (padrão: 0, sem IC)")
    p.add_argument("--braco", action="append", dest="bracos", default=None,
                   metavar="ID",
                   help="roda SÓ este braço (ae_denso | ae_lstm). Pode repetir. "
                        "Sem a opção, roda os dois e emite a comparação.")
    args = p.parse_args(argv)

    from src.core.logs import habilitar_console

    habilitar_console()
    executar(args.n_janelas, args.n_steps, args.n_trajetorias,
             args.n_boot, args.bracos)


if __name__ == "__main__":
    main()
