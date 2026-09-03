"""Os dois braços têm de ser comparáveis no que não é a arquitetura.

POR QUE ESTE TESTE EXISTE
=========================
A comparação Denso × AE-LSTM só sustenta conclusão arquitetural se tudo o que
não é arquitetura for igual. Orçamento de treino, features e sementes já eram.
A regularização não era.

O denso tinha `Dropout(0,2)` nas duas travessias do gargalo. O AE-LSTM não
tinha nenhuma: `nn.LSTM` de camada única IGNORA o argumento `dropout`, então
nem declará-lo ali teria efeito — era preciso um `nn.Dropout` explícito.

Isso abria uma explicação concorrente para o resultado. Com cerca de 16x mais
parâmetros e sem regularização, um autoencoder treinado para reconstruir tende
a reconstruir bem demais, inclusive o que deveria destoar. Atribuir à
arquitetura um efeito que podia ser da regularização é exatamente o erro que a
banca cobraria.

A assimetria que SOBRA — a de capacidade — é inerente às duas arquiteturas e
não se resolve; ela tem de ser publicada. Daí a guarda sobre o contrato.
"""

from __future__ import annotations

import pytest

from src.ml.modelos_autoencoder import DROPOUT

torch = pytest.importorskip("torch")

from src.ml.modelos_autoencoder import (  # noqa: E402
    AutoencoderDenso,
    AutoencoderLSTM,
    SEQUENCE_LENGTH,
    parameter_count,
)


N_FEATURES = 24


@pytest.mark.leve
def test_os_dois_modelos_tem_o_mesmo_dropout_por_padrao():
    assert AutoencoderDenso(N_FEATURES).dropout_p == DROPOUT
    assert AutoencoderLSTM(N_FEATURES).dropout_p == DROPOUT


@pytest.mark.leve
def test_o_lstm_tem_dropout_de_verdade_e_nao_so_declarado():
    """`nn.LSTM` de camada única engole o argumento; o módulo tem de existir."""
    modelo = AutoencoderLSTM(N_FEATURES)
    dropouts = [
        modulo for modulo in modelo.modules() if isinstance(modulo, torch.nn.Dropout)
    ]

    assert len(dropouts) == 2, (
        "o AE-LSTM precisa de dropout explícito nas duas travessias do gargalo"
    )
    assert all(camada.p == DROPOUT for camada in dropouts)


@pytest.mark.leve
def test_o_dropout_age_no_treino_e_some_na_avaliacao():
    """Sem isto, um `nn.Dropout` presente mas inerte passaria no teste acima."""
    torch.manual_seed(0)
    modelo = AutoencoderLSTM(N_FEATURES, dropout=0.9)
    entrada = torch.randn(4, SEQUENCE_LENGTH, N_FEATURES)

    modelo.train()
    torch.manual_seed(1)
    treino_a = modelo(entrada)
    torch.manual_seed(2)
    treino_b = modelo(entrada)
    assert not torch.allclose(treino_a, treino_b), (
        "em treino, duas passagens com dropout ativo não podem coincidir"
    )

    modelo.eval()
    with torch.no_grad():
        assert torch.allclose(modelo(entrada), modelo(entrada))


@pytest.mark.leve
def test_dropout_zero_desliga_o_efeito():
    """A ablação precisa ser possível — é como se mede o que o dropout fez."""
    torch.manual_seed(0)
    modelo = AutoencoderLSTM(N_FEATURES, dropout=0.0)
    entrada = torch.randn(2, SEQUENCE_LENGTH, N_FEATURES)

    modelo.train()
    assert torch.allclose(modelo(entrada), modelo(entrada))


@pytest.mark.leve
def test_a_forma_da_reconstrucao_nao_mudou():
    """O dropout entrou no caminho do gargalo; a saída tem de continuar igual."""
    modelo = AutoencoderLSTM(N_FEATURES).eval()
    entrada = torch.randn(3, SEQUENCE_LENGTH, N_FEATURES)
    with torch.no_grad():
        assert modelo(entrada).shape == entrada.shape


@pytest.mark.leve
def test_a_assimetria_de_capacidade_e_real_e_grande():
    """Não é para ser consertada — é para ser publicada.

    Se algum dia os dois ficarem do mesmo tamanho, a prosa do relatório sobre
    "uma ordem de grandeza" vira falsa, e este teste avisa.
    """
    denso = parameter_count(AutoencoderDenso(N_FEATURES))
    lstm = parameter_count(AutoencoderLSTM(N_FEATURES))

    assert lstm > 10 * denso, (
        f"a assimetria caiu para {lstm / denso:.1f}x; a prosa do relatório fala "
        "em uma ordem de grandeza e precisa ser revista"
    )


@pytest.mark.leve
def test_o_contrato_publica_dropout_e_parametros():
    """Guarda estrutural: a assimetria tem de sair no artefato, não só no código."""
    from pathlib import Path

    fonte = (
        Path(__file__).resolve().parents[1] / "src/ml/publicacao_comparacao.py"
    ).read_text(encoding="utf-8")

    assert '"dropout"' in fonte
    assert '"n_parameters"' in fonte
    assert "Capacidade e regularização" in fonte
