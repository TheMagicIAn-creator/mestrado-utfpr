"""O scaler é um pickle, e carregá-lo executa código.

POR QUE ESTE TESTE EXISTE
=========================
`save_reference_run` grava `scaler.pkl` — um pickle da instância de
`RobustScaler`. Desserializar um pickle EXECUTA o que estiver dentro dele. O
projeto já tinha a defesa pronta em `src/core/seguranca.py`
(`carregar_pickle_verificado`, `gravar_sidecar_sha256`), e o único caminho que
grava o artefato congelado não a usava: nenhum sidecar era escrito, então quem
carregasse o scaler depois não teria contra o que conferir.

A etapa de detectabilidade precisa exatamente desse carregamento — ela
reutiliza os modelos e limiares congelados pela comparação. Escrever o
carregador sem fechar isso seria abrir o buraco no momento de usá-lo.

`carregar_execucao_congelada` é deliberadamente MAIS estrita que
`carregar_pickle_com_sidecar`: aquela avisa e segue quando o sidecar falta,
por compatibilidade com artefatos pré-hardening. Aqui o artefato é sempre
gerado pelo próprio pipeline, então sidecar ausente significa artefato velho
ou mexido — e nos dois casos o certo é parar, não avisar.

O outro grupo de testes cobre a divergência de features. Um checkpoint cujas
`feature_columns` não batem com o código produz vetor na ordem errada: números
plausíveis, nenhum erro. É o mesmo modo de falha do vetor de zeros.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.core.seguranca import gravar_sidecar_sha256
from src.ml.dados_gpvs import FEATURE_COLUMNS

torch = pytest.importorskip("torch")

from src.ml.treino_comparacao import carregar_execucao_congelada  # noqa: E402


class _ScalerFalso:
    """Objeto simples e picklável — o teste não precisa do sklearn real."""

    def __init__(self, fator: float = 2.0):
        self.fator = fator

    def transform(self, x):
        return np.asarray(x) * self.fator


def _montar(raiz: Path, model_id: str = "ae_denso", *, com_sidecar: bool = True,
            features=None) -> Path:
    from src.ml.modelos_autoencoder import AutoencoderDenso

    pasta = raiz / model_id
    pasta.mkdir(parents=True, exist_ok=True)
    modelo = AutoencoderDenso(len(FEATURE_COLUMNS))
    torch.save(
        {
            "state_dict": modelo.state_dict(),
            "model_id": model_id,
            "seed": 42,
            "n_features": len(FEATURE_COLUMNS),
            "feature_columns": list(
                FEATURE_COLUMNS if features is None else features
            ),
            "dense_hidden": 16,
            "lstm_hidden": None,
            "latent_dim": 8,
            "sequence_length": None,
            "score_top_k": 5,
        },
        pasta / "modelo.pt",
    )
    scaler_path = pasta / "scaler.pkl"
    with scaler_path.open("wb") as stream:
        pickle.dump(_ScalerFalso(), stream)
    if com_sidecar:
        gravar_sidecar_sha256(scaler_path)
    (pasta / "contrato.json").write_text(
        json.dumps({"score_threshold": 5.104478, "score_top_k": 5}),
        encoding="utf-8",
    )
    return pasta


# ── o caminho feliz ────────────────────────────────────────────────────────

@pytest.mark.leve
def test_carrega_pesos_scaler_e_limiar_congelados(tmp_path):
    _montar(tmp_path)

    execucao = carregar_execucao_congelada("ae_denso", raiz=tmp_path)

    assert execucao["model_id"] == "ae_denso"
    assert execucao["limiar"] == pytest.approx(5.104478)
    assert execucao["score_top_k"] == 5
    assert execucao["scaler"].fator == 2.0
    assert execucao["modelo"].training is False, (
        "o modelo tem de vir em eval(): dropout ativo mudaria o escore a cada "
        "chamada e a varredura deixaria de ser determinística"
    )


# ── a integridade do pickle ────────────────────────────────────────────────

@pytest.mark.leve
def test_sem_sidecar_recusa_em_vez_de_avisar(tmp_path):
    """Diferente de `carregar_pickle_com_sidecar`, aqui a ausência é erro."""
    _montar(tmp_path, com_sidecar=False)

    with pytest.raises(FileNotFoundError) as erro:
        carregar_execucao_congelada("ae_denso", raiz=tmp_path)

    mensagem = str(erro.value)
    assert "sha256" in mensagem.lower()
    assert "executa código" in mensagem, (
        "a mensagem tem de dizer POR QUE isso importa, não só que faltou"
    )
    assert "comparacao_autoencoders" in mensagem, "e como sair do erro"


@pytest.mark.leve
def test_pickle_adulterado_apos_o_treino_e_recusado(tmp_path):
    """O caso que o sidecar existe para pegar."""
    pasta = _montar(tmp_path)
    with (pasta / "scaler.pkl").open("wb") as stream:
        pickle.dump(_ScalerFalso(fator=999.0), stream)   # trocado em disco

    with pytest.raises(ValueError, match="Integridade violada"):
        carregar_execucao_congelada("ae_denso", raiz=tmp_path)


@pytest.mark.leve
def test_sidecar_com_hash_errado_e_recusado(tmp_path):
    pasta = _montar(tmp_path)
    (pasta / "scaler.pkl.sha256").write_text("0" * 64, encoding="utf-8")

    with pytest.raises(ValueError, match="Integridade violada"):
        carregar_execucao_congelada("ae_denso", raiz=tmp_path)


@pytest.mark.leve
def test_o_treino_grava_o_sidecar(tmp_path):
    """Guarda estrutural: sem isto o carregador recusaria tudo.

    A escrita e a leitura precisam concordar; se alguém remover a gravação, o
    pipeline inteiro para de carregar — e é melhor descobrir aqui.
    """
    fonte = (
        Path(__file__).resolve().parents[1] / "src/ml/treino_comparacao.py"
    ).read_text(encoding="utf-8")

    assert "gravar_sidecar_sha256(scaler_path)" in fonte
    assert "carregar_pickle_verificado" in fonte


# ── divergência de contrato ────────────────────────────────────────────────

@pytest.mark.leve
def test_features_divergentes_estouram_em_vez_de_pontuar_errado(tmp_path):
    """Ordem trocada é erro silencioso: vetor plausível, modelo errado."""
    invertidas = list(FEATURE_COLUMNS)[::-1]
    _montar(tmp_path, features=invertidas)

    with pytest.raises(ValueError, match="features do checkpoint"):
        carregar_execucao_congelada("ae_denso", raiz=tmp_path)


@pytest.mark.leve
def test_model_id_divergente_estoura(tmp_path):
    _montar(tmp_path, model_id="ae_denso")
    (tmp_path / "ae_lstm").mkdir()
    for nome in ("modelo.pt", "scaler.pkl", "scaler.pkl.sha256", "contrato.json"):
        (tmp_path / "ae_lstm" / nome).write_bytes(
            (tmp_path / "ae_denso" / nome).read_bytes()
        )

    with pytest.raises(ValueError, match="model_id"):
        carregar_execucao_congelada("ae_lstm", raiz=tmp_path)


@pytest.mark.leve
def test_artefato_ausente_diz_o_que_rodar(tmp_path):
    with pytest.raises(FileNotFoundError) as erro:
        carregar_execucao_congelada("ae_denso", raiz=tmp_path)

    assert "comparacao_autoencoders" in str(erro.value)
