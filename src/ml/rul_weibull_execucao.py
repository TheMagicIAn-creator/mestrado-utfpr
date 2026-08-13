"""Orquestracao dos artefatos de detectabilidade Weibull E2.

A matematica e a API publica permanecem em :mod:`src.ml.rul_weibull`. Este
modulo separa a execucao custosa e a serializacao para manter responsabilidades
auditiveis e o limite arquitetural dos modulos de producao.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.rul_weibull import (
    A_DET_UNIDADE,
    FALHAS,
    MAX_CENSURA_RUL_PCT,
    MAX_VARIACAO_RELATIVA_GRADE,
    MIN_EVENTOS_WEIBULL,
    MIN_R2_PAPEL_WEIBULL,
    N_BOOTSTRAP,
    N_BOOTSTRAP_ADERENCIA,
    N_BOOTSTRAP_MODO,
    N_STEPS,
    N_STEPS_SENSIBILIDADE,
    N_TRAJ,
    PASTA_AE,
    PERSISTENCIA_CRUZAMENTO,
    PERSISTENCIA_MAGNITUDE,
    TEMPO_FISICO_CALIBRADO,
    TEMPO_FISICO_NOTA,
    TTF_UNIDADE,
    _json_seguro,
    _log,
    ajustar_weibull,
    carregar_normalizacao_baseline,
    classificar_desfechos,
    gerar_a_det,
    metadados_tempo_rul,
    passos_persistencia,
    preparar_janelas_holdout,
    selecionar_janelas_baseline_normais,
    selecionar_trajetorias_holdout,
)


def executar_rul_weibull() -> bool:
    from src.ml.graficos_rul import (
        plotar_confiabilidade,
        plotar_distribuicao_weibull,
        plotar_funcoes_distribuicao_weibull,
        plotar_intensidade_deteccao,
        plotar_modos_operacao,
        plotar_rul,
        plotar_sensibilidade_grade,
        plotar_ttf_histogramas,
    )

    _log("=" * 60)
    _log("  AL IADO PV — DETECTABILIDADE E2 COM WEIBULL")
    _log("=" * 60)
    _log(f"\n  Teto de trajetórias por falha: {N_TRAJ}")
    _log(f"  Grade de magnitude   : {N_STEPS} pontos (a_inj 0→1,0)")
    _log(f"  Eixo do Weibull      : a_det — fração da assinatura nominal, NÃO tempo")

    # ── 1. Carrega artefatos ─────────────────────────────────
    _log(f"\n📂 Carregando artefatos...")
    for arq in [PASTA_AE/"modelo_autoencoder.pt",
                PASTA_AE/"scaler.pkl",
                PASTA_AE/"limiar.json"]:
        if not arq.exists():
            _log(f"   ❌ {arq.name} não encontrado")
            return False

    import torch
    from src.ml.autoencoder import Autoencoder

    checkpoint = torch.load(PASTA_AE/"modelo_autoencoder.pt",
                            map_location="cpu", weights_only=False)
    from src.core.seguranca import carregar_pickle_com_sidecar

    scaler = carregar_pickle_com_sidecar(PASTA_AE / "scaler.pkl")
    with open(PASTA_AE/"limiar.json", "r") as f:
        info_limiar = json.load(f)

    n_features   = checkpoint["n_features"]
    latente_dim  = checkpoint["latente_dim"]
    colunas_feat = checkpoint["colunas_feat"]
    limiar       = info_limiar["limiar"]   # OPERACIONAL (método escolhido)

    # Escore operacional (o MESMO que definiu o limiar): método + régua.
    # O TTF é o passo em que ESTE escore cruza ESTE limiar. Sem a régua
    # (artefato antigo), cai para MSE.
    from src.ml import escore_anomalia as ea

    metodo_escore = info_limiar.get("metodo_escore", "mse")
    estat_residuo = ea.carregar_estatistica(PASTA_AE)
    normalizacao_baseline = carregar_normalizacao_baseline(PASTA_AE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = Autoencoder(n_features, latente_dim).to(device)
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.eval()
    _log(f"   ✅ Limiar={limiar:.4f} | device={device} | "
          f"escore={ea.descricao_metodo(metodo_escore, info_limiar.get('k_localizado', 5))}")

    # ── 2. Holdout temporal isolado ───────────────────────────
    _log(f"\n📂 Carregando dataset...")
    janelas_holdout, meta_holdout = preparar_janelas_holdout()
    n_janelas_originais = len(janelas_holdout)
    janelas_holdout, erros_baseline, mascara_elegivel = selecionar_janelas_baseline_normais(
        janelas_holdout, modelo, scaler, device, colunas_feat, limiar,
        estat_residuo, metodo_escore, normalizacao_baseline,
    )
    n_excluidas = int((~mascara_elegivel).sum())
    meta_holdout["filtro_baseline_ttf"] = {
        "criterio": "erro_reconstrucao_baseline <= limiar",
        "limiar": float(limiar),
        "n_janelas_antes": n_janelas_originais,
        "n_janelas_elegiveis": len(janelas_holdout),
        "n_janelas_excluidas": n_excluidas,
        "erros_baseline": [float(x) for x in erros_baseline],
    }
    if not janelas_holdout:
        _log("   ❌ Nenhuma janela saudável ficou abaixo do limiar para gerar TTF")
        return False
    janelas_holdout = selecionar_trajetorias_holdout(janelas_holdout, N_TRAJ)
    n_traj_real = len(janelas_holdout)
    contagem_ensaios: dict[str, int] = {}
    for janela in janelas_holdout:
        ensaio = str(janela.attrs.get("ensaio", "sem_ensaio"))
        contagem_ensaios[ensaio] = contagem_ensaios.get(ensaio, 0) + 1
    meta_holdout["amostragem_trajetorias_weibull"] = {
        "metodo": (
            "todas_as_janelas_elegiveis"
            if N_TRAJ is None else
            "amostra_uniforme_estratificada_por_ensaio"
        ),
        "n_max": N_TRAJ,
        "n_selecionadas": n_traj_real,
        "por_ensaio": contagem_ensaios,
    }
    _log(f"   ✅ {n_janelas_originais} janelas não sobrepostas do teste")
    _log(f"   ✅ {len(janelas_holdout)} elegíveis; {n_excluidas} excluídas por anomalia em t=0")
    _log(f"   ✅ {n_traj_real} trajetórias serão usadas: {contagem_ensaios}; "
         "independência temporal não é presumida")

    # ── 3. Gera cruzamentos na grade principal e nas grades de sensibilidade ──
    _log(f"\n⚙️  Gerando trajetórias de magnitude...")

    def gerar_grade(n_steps: int, *, detalhar: bool) -> tuple[dict, dict]:
        a_dets_grade: dict[str, np.ndarray] = {}
        eventos_grade: dict[str, np.ndarray] = {}
        for falha in FALHAS:
            fid, nome = falha["id"], falha["nome"]
            _log(
                f"\n   🔴 {nome} ({n_traj_real} trajetórias × "
                f"{n_steps} pontos)..."
            )
            a_dets, eventos = [], []
            for i, janela_base in enumerate(janelas_holdout):
                a_det, detectou = gerar_a_det(
                    janela_base, modelo, scaler, device,
                    colunas_feat, limiar, fid, n_steps, seed=i,
                    estat_residuo=estat_residuo, metodo=metodo_escore,
                    normalizacao_baseline=normalizacao_baseline,
                )
                a_dets.append(a_det)
                eventos.append(detectou)
                if detalhar and (i + 1) % 20 == 0:
                    _log(
                        f"      [{i+1:>3}/{n_traj_real}] a_det médio: "
                        f"{np.mean(a_dets):.3f}", end="\r"
                    )
            a_arr = np.asarray(a_dets, dtype=float)
            e_arr = np.asarray(eventos, dtype=bool)
            a_dets_grade[fid] = a_arr
            eventos_grade[fid] = e_arr
            if detalhar:
                d = classificar_desfechos(a_arr, e_arr)
                _log(
                    f"\n      a_det: μ={a_arr.mean():.3f} ± {a_arr.std():.3f} | "
                    f"min={a_arr.min():.3f} | max={a_arr.max():.3f}"
                )
                _log(
                    "      POD_mon no teto (a_inj=1,0): "
                    f"{d['pod_mon_no_teto']:.1%} | indetectáveis no teto: "
                    f"{d['n_indetectaveis_no_teto']}/{d['n_traj']}"
                )
        return a_dets_grade, eventos_grade

    resultados_grade: dict[int, tuple[dict, dict]] = {}
    for n_steps in sorted(set(N_STEPS_SENSIBILIDADE + (N_STEPS,))):
        resultados_grade[n_steps] = gerar_grade(
            n_steps, detalhar=n_steps == N_STEPS
        )
    ttfs_dict, eventos_dict = resultados_grade[N_STEPS]

    # ── 4. Ajuste de Weibull ─────────────────────────────────
    _log(f"\n📐 Ajustando distribuição de Weibull...")
    params = {}
    for falha in FALHAS:
        fid = falha["id"]
        p = ajustar_weibull(
            ttfs_dict[fid], eventos_dict[fid], n_boot=N_BOOTSTRAP,
            seed=42 + len(params), passo_grade=1.0 / (N_STEPS - 1),
            n_boot_aderencia=N_BOOTSTRAP_ADERENCIA,
        )
        params[fid] = p
        npm_str = f"NPR={falha['npr']}"
        _log(f"\n   {falha['nome']} ({npm_str})")
        if p["fit_converged"]:
            _log(f"      β={p['beta']:.3f}  η={p['eta']:.3f}  "
                  f"média(a_det)={p['mttf']:.3f}  a10={p['b10']:.3f}")
            _log(f"      Censura={p['censura_pct']:.0f}% | "
                 f"R²(papel)={p['diagnostico_papel_weibull']['r2']:.3f} | "
                 f"bootstrap={p['bootstrap_validos']}/{p['bootstrap_solicitados']}")
        else:
            _log(
                f"      ⚠️ Weibull não estimável: {p['n_eventos']} eventos; "
                f"mínimo configurado={MIN_EVENTOS_WEIBULL}. "
                "Margem restrita por Kaplan-Meier será mantida."
            )

    # ── 4b. Sensibilidade à grade e estratificação por modo operacional ──
    ensaios_trajetorias = np.asarray([
        str(janela.attrs.get("ensaio", "sem_ensaio"))
        for janela in janelas_holdout
    ])
    linhas_sensibilidade: list[dict] = []
    linhas_trajetorias: list[dict] = []
    ajustes_grade: dict[tuple[str, str, int], dict] = {}
    for n_steps, (a_grade, e_grade) in sorted(resultados_grade.items()):
        delta_a = 1.0 / (n_steps - 1)
        for falha in FALHAS:
            fid = falha["id"]
            for indice, (a_det, detectou, ensaio) in enumerate(zip(
                a_grade[fid], e_grade[fid], ensaios_trajetorias, strict=True
            )):
                linhas_trajetorias.append({
                    "trajetoria_id": indice,
                    "ensaio": ensaio,
                    "falha": falha["nome"],
                    "falha_id": fid,
                    "n_steps": n_steps,
                    "delta_a": delta_a,
                    "persistencia_magnitude": PERSISTENCIA_MAGNITUDE,
                    "persistencia_pontos": passos_persistencia(n_steps),
                    "a_det": float(a_det),
                    "detectou": bool(detectou),
                })
            for escopo in ("global", *sorted(contagem_ensaios)):
                mascara = (
                    np.ones(n_traj_real, dtype=bool)
                    if escopo == "global" else ensaios_trajetorias == escopo
                )
                if n_steps == N_STEPS and escopo == "global":
                    ajuste = params[fid]
                else:
                    ajuste = ajustar_weibull(
                        a_grade[fid][mascara], e_grade[fid][mascara],
                        n_boot=N_BOOTSTRAP_MODO
                        if n_steps == N_STEPS else 0,
                        seed=500 + n_steps + len(linhas_sensibilidade),
                        passo_grade=delta_a,
                        n_boot_aderencia=N_BOOTSTRAP_ADERENCIA
                        if n_steps == N_STEPS else 0,
                    )
                ajustes_grade[(fid, escopo, n_steps)] = ajuste
                p_gof = (ajuste.get("teste_aderencia_quantizada") or {}).get(
                    "p_value"
                )
                linhas_sensibilidade.append({
                    "falha": falha["nome"],
                    "falha_id": fid,
                    "escopo": escopo,
                    "n_steps": n_steps,
                    "delta_a": delta_a,
                    "persistencia_magnitude": PERSISTENCIA_MAGNITUDE,
                    "persistencia_pontos": passos_persistencia(n_steps),
                    "n_traj": ajuste["n_traj"],
                    "n_eventos": ajuste["n_eventos"],
                    "n_niveis_distintos": ajuste["n_niveis_distintos"],
                    "taxa_empates": ajuste["taxa_empates"],
                    "mediana_a_det": float(np.median(a_grade[fid][mascara])),
                    "beta": ajuste["beta"],
                    "eta": ajuste["eta"],
                    "r2_papel": ajuste["diagnostico_papel_weibull"]["r2"],
                    "aderencia_p_value": p_gof,
                    "status_aderencia": ajuste["status_aderencia"],
                })

    duas_grades_finas = sorted(resultados_grade)[-2:]
    for falha in FALHAS:
        fid = falha["id"]
        for escopo in ("global", *sorted(contagem_ensaios)):
            grosso = ajustes_grade[(fid, escopo, duas_grades_finas[0])]
            fino = ajustes_grade[(fid, escopo, duas_grades_finas[1])]
            variacoes = {
                nome: float(
                    abs(fino[nome] - grosso[nome])
                    / max(abs(fino[nome]), np.finfo(float).eps)
                )
                for nome in ("beta", "eta")
            }
            estavel = bool(
                fino["fit_converged"] and grosso["fit_converged"]
                and max(variacoes.values()) <= MAX_VARIACAO_RELATIVA_GRADE
            )
            diagnostico_grade = {
                "grades_comparadas": duas_grades_finas,
                "variacao_relativa": variacoes,
                "limite_variacao_relativa": MAX_VARIACAO_RELATIVA_GRADE,
                "estavel": estavel,
            }
            if escopo == "global":
                params[fid]["sensibilidade_grade"] = diagnostico_grade
                params[fid]["resumo_parametrico_recomendado"] = bool(
                    params[fid]["resumo_parametrico_recomendado"] and estavel
                )
            else:
                ajuste_modo = ajustes_grade[(fid, escopo, N_STEPS)]
                ajuste_modo["sensibilidade_grade"] = diagnostico_grade
                ajuste_modo["resumo_parametrico_recomendado"] = bool(
                    ajuste_modo["resumo_parametrico_recomendado"] and estavel
                )
                params[fid].setdefault("ajustes_por_modo", {})[
                    escopo
                ] = ajuste_modo

    arq_sensibilidade = PASTA_AE / "weibull_sensibilidade_grade.csv"
    pd.DataFrame(linhas_sensibilidade).to_csv(
        arq_sensibilidade, index=False
    )
    _log(f"   📋 {arq_sensibilidade.name}")
    arq_trajetorias = PASTA_AE / "weibull_trajetorias_grade.csv"
    pd.DataFrame(linhas_trajetorias).to_csv(arq_trajetorias, index=False)
    _log(f"   📋 {arq_trajetorias.name}")

    # ── 5. Visualizações ─────────────────────────────────────
    _log(f"\n📊 Gerando gráficos...")
    plotar_ttf_histogramas(ttfs_dict, eventos_dict, params, PASTA_AE)
    plotar_confiabilidade(ttfs_dict, eventos_dict, params, PASTA_AE)
    plotar_intensidade_deteccao(ttfs_dict, eventos_dict, params, PASTA_AE)
    plotar_funcoes_distribuicao_weibull(
        ttfs_dict, eventos_dict, params, PASTA_AE
    )
    plotar_distribuicao_weibull(ttfs_dict, eventos_dict, params, PASTA_AE)
    plotar_rul(ttfs_dict, eventos_dict, params, PASTA_AE)
    plotar_sensibilidade_grade(resultados_grade, PASTA_AE)
    plotar_modos_operacao(
        ttfs_dict, eventos_dict, ensaios_trajetorias, params, PASTA_AE
    )

    # ── 6. Salva resultados ──────────────────────────────────
    # A montagem do artefato vive em src/ml/relatorio_weibull.py: este módulo
    # ficou com a matemática, aquele com a serialização. Ver o docstring de lá.
    from src.ml.relatorio_weibull import montar_relatorio

    relatorio, linhas_weibull = montar_relatorio(
        params=params, a_dets_dict=ttfs_dict, eventos_dict=eventos_dict,
        falhas=FALHAS, meta_holdout=meta_holdout,
        metadados_tempo=metadados_tempo_rul(), limiar=float(limiar),
        n_traj_max=N_TRAJ, n_traj_real=n_traj_real, n_steps=N_STEPS,
        a_det_unidade=A_DET_UNIDADE, ttf_unidade=TTF_UNIDADE,
        tempo_fisico_calibrado=TEMPO_FISICO_CALIBRADO,
        tempo_fisico_nota=TEMPO_FISICO_NOTA,
        min_eventos_weibull=MIN_EVENTOS_WEIBULL,
        max_censura_rul_pct=MAX_CENSURA_RUL_PCT,
        min_r2_papel_weibull=MIN_R2_PAPEL_WEIBULL,
        persistencia_cruzamento=PERSISTENCIA_CRUZAMENTO,
        persistencia_magnitude=PERSISTENCIA_MAGNITUDE,
        json_seguro=_json_seguro,
    )

    arq_json = PASTA_AE / "weibull_results.json"
    with open(arq_json, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)
    _log(f"   ✅ {arq_json.name}")

    arq_tabela = PASTA_AE / "weibull_tabela.csv"
    pd.DataFrame(linhas_weibull).to_csv(arq_tabela, index=False)
    _log(f"   📋 {arq_tabela.name}")
    from scripts.relatorio_confiabilidade import main as gerar_relatorio
    gerar_relatorio()

    # ── 7. Resumo final ──────────────────────────────────────
    _log(f"\n{'='*60}")
    _log(f"  ANÁLISE DE DETECTABILIDADE WEIBULL E2 CONCLUÍDA!")
    _log(f"\n  Valores em FRAÇÃO DA ASSINATURA NOMINAL (a_det), não em tempo.")
    _log(f"\n  {'Falha':<28} {'β':>6} {'η':>7} {'média a':>8} {'a10':>8} {'POD@1,0':>8}")
    _log(f"  {'-'*68}")
    for falha in FALHAS:
        fid = falha["id"]
        p   = params[fid]
        media = f"{p['media_a_det_parametrica']:>8.3f}" if p["resumo_parametrico_recomendado"] else f"{'--':>8}"
        a10 = f"{p['a10_parametrico']:>8.3f}" if p["resumo_parametrico_recomendado"] else f"{'--':>8}"
        _log(f"  {falha['nome']:<28} "
              f"{p['beta']:>6.3f} {p['eta']:>7.3f} "
              f"{media} {a10} "
              f"{p['desfechos']['pod_mon_no_teto']:>7.1%}")

    # A leitura do β só vale se o IC95 não cruzar 1 — quem decide isso é
    # confiabilidade.classificar_forma, e a conclusão dela já vem no artefato.
    _log(f"\n  Interpretação do β (válida só quando o IC95 não cruza 1):")
    for falha in FALHAS:
        p = params[falha["id"]]
        interp = p.get("interpretacao") or {}
        if interp.get("leitura"):
            marca = "" if interp.get("conclusivo") else "⚠️  "
            _log(f"  {marca}{falha['nome']}: {interp['leitura']}")
    _log(f"\n  Fase 5 do pipeline de ML concluída!")
    _log(f"  Relatório acadêmico e artefatos integrados atualizados.")
    _log(f"{'='*60}")
    return True


def regenerar_graficos_weibull(pasta: Path = PASTA_AE) -> dict:
    """Regenera as oito figuras a partir dos artefatos tabulares versionados."""
    from src.ml.graficos_rul import (
        plotar_confiabilidade,
        plotar_distribuicao_weibull,
        plotar_funcoes_distribuicao_weibull,
        plotar_intensidade_deteccao,
        plotar_modos_operacao,
        plotar_rul,
        plotar_sensibilidade_grade,
        plotar_ttf_histogramas,
    )

    pasta = Path(pasta)
    dados = json.loads(
        (pasta / "weibull_results.json").read_text(encoding="utf-8")
    )
    trajetorias = pd.read_csv(pasta / "weibull_trajetorias_grade.csv")
    params = {
        fid: falha["weibull"] for fid, falha in dados["falhas"].items()
    }
    a_dets = {
        fid: np.asarray(falha["a_dets"], dtype=float)
        for fid, falha in dados["falhas"].items()
    }
    eventos = {
        fid: np.asarray(falha["eventos_observados"], dtype=bool)
        for fid, falha in dados["falhas"].items()
    }
    resultados_grade: dict[int, tuple[dict, dict]] = {}
    for n_steps, bloco_grade in trajetorias.groupby("n_steps", sort=True):
        a_grade, e_grade = {}, {}
        for fid, bloco_falha in bloco_grade.groupby("falha_id", sort=False):
            ordenado = bloco_falha.sort_values("trajetoria_id")
            a_grade[str(fid)] = ordenado["a_det"].to_numpy(dtype=float)
            e_grade[str(fid)] = (
                ordenado["detectou"].astype(str).str.lower().eq("true")
                .to_numpy(dtype=bool)
            )
        resultados_grade[int(n_steps)] = (a_grade, e_grade)

    principal = trajetorias[
        trajetorias["n_steps"] == dados["parametros_simulacao"]["n_steps"]
    ]
    ensaios = (
        principal[["trajetoria_id", "ensaio"]]
        .drop_duplicates()
        .sort_values("trajetoria_id")["ensaio"]
        .to_numpy(dtype=str)
    )
    plotar_ttf_histogramas(a_dets, eventos, params, pasta)
    plotar_confiabilidade(a_dets, eventos, params, pasta)
    plotar_intensidade_deteccao(a_dets, eventos, params, pasta)
    plotar_funcoes_distribuicao_weibull(a_dets, eventos, params, pasta)
    plotar_distribuicao_weibull(a_dets, eventos, params, pasta)
    plotar_rul(a_dets, eventos, params, pasta)
    plotar_sensibilidade_grade(resultados_grade, pasta)
    plotar_modos_operacao(a_dets, eventos, ensaios, params, pasta)
    return {
        "ok": True,
        "outputs": [str(pasta / nome) for nome in (
            "weibull_ttf.png", "weibull_confiabilidade.png",
            "weibull_intensidade_deteccao.png",
            "weibull_funcoes_distribuicao.png",
            "weibull_distribuicao.png", "weibull_rul.png",
            "weibull_sensibilidade_grade.png",
            "weibull_modos_operacao.png",
        )],
    }
