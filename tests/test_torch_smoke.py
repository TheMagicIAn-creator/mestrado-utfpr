"""
O Autoencoder REALMENTE constrói, treina e reconstrói — em dados de brinquedo.

POR QUE ESTE TESTE EXISTE
=========================
O CI instalava só `requirements-dev.txt`, sem torch. Consequência: nenhum teste
jamais instanciou o `Autoencoder`. Uma mudança que quebrasse a construção da
rede, o `forward`, o carregamento do checkpoint ou o loop de treino passaria
verde e só apareceria numa execução local de 8 minutos — que é exatamente onde
o custo de descobrir é maior.

**Não é mock.** Mockar torch testaria o mock. Aqui roda o torch de verdade, com
uma rede minúscula (12 features, 6 janelas, 2 épocas): segundos de CPU, sem
dataset e sem GPU. O que se verifica é o CONTRATO — formas, finitude, a ausência
de ativação no gargalo, o ciclo salvar/carregar —, não os números do modelo
treinado, que dependem do dataset e pertencem aos manifestos.

Marcado `integracao`: some com honestidade se torch não estiver instalado, em
vez de reprovar a suíte de quem não tem o ambiente completo.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="requer torch (requirements-ml.txt)")

import numpy as np  # noqa: E402

from src.ml.autoencoder import LATENTE_DIM, Autoencoder  # noqa: E402

pytestmark = pytest.mark.integracao

N_FEATURES = 12
N_JANELAS = 6


def _lote(seed: int = 0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(
        rng.normal(0.0, 1.0, size=(N_JANELAS, N_FEATURES)).astype(np.float32)
    )


# ── construção e forma ─────────────────────────────────────────────────────

def test_forward_preserva_a_forma_do_lote():
    modelo = Autoencoder(N_FEATURES, latente_dim=4)
    saida = modelo(_lote())
    assert saida.shape == (N_JANELAS, N_FEATURES)
    assert torch.isfinite(saida).all()


def test_encode_devolve_o_espaco_latente_pedido():
    modelo = Autoencoder(N_FEATURES, latente_dim=4)
    assert modelo.encode(_lote()).shape == (N_JANELAS, 4)


def test_latente_padrao_acompanha_a_constante_do_modulo():
    """O default do `__init__` já contradisse `LATENTE_DIM` uma vez."""
    assert Autoencoder(N_FEATURES).encode(_lote()).shape[1] == LATENTE_DIM


# ── o gargalo não pode voltar a ter ReLU ───────────────────────────────────

def test_gargalo_admite_valores_negativos():
    """A ÚLTIMA camada do encoder é linear, sem ativação.

    Com ReLU no gargalo o latente fica preso ao ortante não negativo e unidades
    podem morrer em zero permanente. Este teste é comportamental de propósito:
    inspecionar `isinstance` da última camada quebraria numa refatoração
    legítima; o que importa é que exista entrada cujo código seja negativo.
    """
    modelo = Autoencoder(N_FEATURES, latente_dim=4)
    with torch.no_grad():
        codigos = torch.cat([modelo.encode(_lote(s)) for s in range(30)])
    assert (codigos < 0).any(), (
        "nenhum código latente negativo em 180 janelas — sinal de ativação "
        "não negativa no gargalo (ver src/ml/autoencoder.py)"
    )


def test_contagem_de_parametros_bate_com_a_topologia_declarada():
    """`n→16→L→16→n`, com bias em todas as camadas."""
    n, latente = N_FEATURES, 4
    esperado = ((n * 16 + 16) + (16 * latente + latente)
                + (latente * 16 + 16) + (16 * n + n))
    modelo = Autoencoder(n, latente_dim=latente)
    assert sum(p.numel() for p in modelo.parameters()) == esperado


# ── o loop de treino ───────────────────────────────────────────────────────

def test_treino_curto_roda_e_reduz_a_perda():
    from torch.utils.data import DataLoader, TensorDataset

    from src.ml.autoencoder import treinar

    dados = TensorDataset(_lote(1))
    loader = DataLoader(dados, batch_size=3, shuffle=False)
    modelo = Autoencoder(N_FEATURES, latente_dim=4, dropout=0.0)

    hist_t, hist_v, epoca = treinar(
        modelo, loader, loader, epochs=12, lr=1e-2, paciencia=12,
        device=torch.device("cpu"),
    )
    assert len(hist_t) == len(hist_v) == 12
    assert all(np.isfinite(hist_t)), "perda virou NaN/inf no treino"
    assert hist_t[-1] < hist_t[0], "12 épocas não reduziram a perda de treino"
    assert 1 <= epoca <= 12


def test_dropout_zero_torna_a_saida_deterministica_em_eval():
    modelo = Autoencoder(N_FEATURES, latente_dim=4)
    modelo.eval()
    x = _lote(2)
    with torch.no_grad():
        assert torch.allclose(modelo(x), modelo(x))


# ── o ciclo salvar → carregar, que o pipeline faz entre etapas ─────────────

def test_checkpoint_reconstroi_o_modelo_identico(tmp_path):
    """`injecao_falhas`, `validacao` e `rul_weibull` reconstroem o modelo a
    partir de `n_features`/`latente_dim` do checkpoint. Se a assinatura mudar,
    as três etapas quebram DEPOIS do treino ter rodado."""
    original = Autoencoder(N_FEATURES, latente_dim=4)
    original.eval()
    arq = tmp_path / "modelo.pt"
    torch.save({"state_dict": original.state_dict(),
                "n_features": N_FEATURES, "latente_dim": 4}, arq)

    ckpt = torch.load(arq, map_location="cpu", weights_only=False)
    copia = Autoencoder(ckpt["n_features"], ckpt["latente_dim"])
    copia.load_state_dict(ckpt["state_dict"])
    copia.eval()

    x = _lote(3)
    with torch.no_grad():
        assert torch.allclose(original(x), copia(x), atol=1e-6)


# ── o escore operacional, que decide o limiar ──────────────────────────────

def test_escore_localizado_roda_sobre_o_modelo_de_verdade():
    """`escore_anomalia` é a régua que define SMD, POD_mon e o eixo do Weibull.

    Ele já tem testes com resíduos sintéticos; o que faltava era exercitá-lo
    contra um modelo torch real, que é como o pipeline o usa.
    """
    from src.ml import escore_anomalia as ea

    modelo = Autoencoder(N_FEATURES, latente_dim=4)
    modelo.eval()
    x = _lote(4).numpy()

    residuos = ea.residuo_por_feature(modelo, x, torch.device("cpu"))
    assert residuos.shape == (N_JANELAS, N_FEATURES)
    assert np.isfinite(residuos).all()

    escores = ea.pontuar(residuos, None, "mse")
    assert escores.shape == (N_JANELAS,)
    assert (escores >= 0).all(), "MSE não pode ser negativo"
