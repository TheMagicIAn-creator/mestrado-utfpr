"""Nucleo matematico do autoencoder denso V2.

O modulo nao conhece arquivos, figuras, falhas ou FMECA. Ele recebe matrizes
normalizadas, treina uma rede compacta e devolve erros de reconstrucao. Essa
fronteira impede que dados de falha entrem acidentalmente na selecao do modelo.
"""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:  # CI leve nao instala torch
    torch = None
    nn = None
    DataLoader = TensorDataset = None


FAMILIAS_FEATURES: dict[str, tuple[str, ...]] = {
    "operacao_cc": (
        "Ipv_median",
        "Ipv_iqr",
        "Vpv_median",
        "Vpv_iqr",
        "Vdc_median",
        "Vdc_iqr",
        "p_dc_median",
        "p_dc_iqr",
    ),
    "corrente_ca": (
        "ia_rms",
        "ib_rms",
        "ic_rms",
        "ia_thd",
        "ib_thd",
        "ic_thd",
        "i_rms_unbalance",
    ),
    "tensao_ca": (
        "va_rms",
        "vb_rms",
        "vc_rms",
        "va_thd",
        "vb_thd",
        "vc_thd",
        "v_rms_unbalance",
    ),
    "potencia_ca": (
        "p_ac_mean",
        "p_ac_std",
    ),
}


@dataclass(frozen=True)
class Arquitetura:
    """Especificacao serializavel de uma rede candidata."""

    id: str
    ocultas: tuple[int, ...]
    latente: int
    dropout: float = 0.0
    weight_decay: float = 1e-5
    inclinacao_leaky_relu: float = 0.1

    def como_dict(self) -> dict:
        dados = asdict(self)
        dados["ocultas"] = list(self.ocultas)
        return dados

    @classmethod
    def de_dict(cls, dados: dict) -> "Arquitetura":
        return cls(
            id=str(dados["id"]),
            ocultas=tuple(int(v) for v in dados["ocultas"]),
            latente=int(dados["latente"]),
            dropout=float(dados.get("dropout", 0.0)),
            weight_decay=float(dados.get("weight_decay", 1e-5)),
            inclinacao_leaky_relu=float(
                dados.get("inclinacao_leaky_relu", 0.1)
            ),
        )


ARQUITETURAS_CANDIDATAS = (
    Arquitetura("compacto_12_4", (12,), 4, dropout=0.0),
    Arquitetura("simetrico_16_8", (16,), 8, dropout=0.1),
    Arquitetura("profundo_16_8_4", (16, 8), 4, dropout=0.1),
)


def exigir_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError(
            "PyTorch e necessario para treinar ou pontuar o autoencoder V2."
        )


def pesos_por_familia(colunas: list[str] | tuple[str, ...]) -> np.ndarray:
    """Atribui o mesmo peso total a cada familia fisica.

    Dentro de uma familia, o peso e dividido igualmente entre as features. Isso
    evita que corrente/tensao trifasicas e potencias derivadas dominem o escore
    apenas porque possuem mais colunas correlacionadas.
    """

    nomes = list(colunas)
    declaradas = [nome for grupo in FAMILIAS_FEATURES.values() for nome in grupo]
    duplicadas = {nome for nome in declaradas if declaradas.count(nome) > 1}
    if duplicadas:
        raise ValueError(f"Features repetidas entre familias: {sorted(duplicadas)}")
    faltando = sorted(set(nomes) - set(declaradas))
    extras = sorted(set(declaradas) - set(nomes))
    if faltando or extras or len(nomes) != len(declaradas):
        raise ValueError(
            f"Contrato de familias diverge das features: faltando={faltando}, "
            f"extras={extras}"
        )

    peso_familia = 1.0 / len(FAMILIAS_FEATURES)
    mapa = {}
    for grupo in FAMILIAS_FEATURES.values():
        peso_feature = peso_familia / len(grupo)
        mapa.update({nome: peso_feature for nome in grupo})
    pesos = np.asarray([mapa[nome] for nome in nomes], dtype=np.float32)
    if not np.isclose(float(pesos.sum()), 1.0):
        raise AssertionError("Pesos das familias nao somam um")
    return pesos


_Base = nn.Module if nn is not None else object


class AutoencoderDenso(_Base):
    """Rede simetrica, com gargalo e saida lineares."""

    def __init__(
        self,
        n_features: int,
        arquitetura: Arquitetura,
    ) -> None:
        exigir_torch()
        super().__init__()
        if n_features <= 0 or arquitetura.latente <= 0:
            raise ValueError("Dimensoes da rede devem ser positivas")
        if any(v <= arquitetura.latente for v in arquitetura.ocultas):
            raise ValueError("Camadas ocultas devem ser maiores que o gargalo")

        self.n_features = int(n_features)
        self.arquitetura = arquitetura
        self.encoder = nn.Sequential(
            *self._bloco(
                (self.n_features, *arquitetura.ocultas, arquitetura.latente),
                ativar_ultima=False,
            )
        )
        self.decoder = nn.Sequential(
            *self._bloco(
                (
                    arquitetura.latente,
                    *reversed(arquitetura.ocultas),
                    self.n_features,
                ),
                ativar_ultima=False,
            )
        )

    def _bloco(self, dimensoes: tuple[int, ...], *, ativar_ultima: bool) -> list:
        camadas = []
        ultimo_indice = len(dimensoes) - 2
        for indice, (entrada, saida) in enumerate(
            zip(dimensoes[:-1], dimensoes[1:], strict=True)
        ):
            camadas.append(nn.Linear(entrada, saida))
            if indice < ultimo_indice or ativar_ultima:
                camadas.append(
                    nn.LeakyReLU(self.arquitetura.inclinacao_leaky_relu)
                )
                if self.arquitetura.dropout > 0:
                    camadas.append(nn.Dropout(self.arquitetura.dropout))
        return camadas

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    @property
    def n_parametros(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


@dataclass
class ResultadoTreino:
    historico_treino: list[float]
    historico_validacao: list[float]
    melhor_epoca: int
    melhor_validacao: float
    state_dict: dict


def configurar_seed(seed: int) -> None:
    """Configura todas as fontes de aleatoriedade antes de criar a rede."""

    exigir_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def erro_balanceado_torch(reconstrucao, alvo, pesos):
    """Media no lote da soma ponderada dos residuos quadraticos."""

    return (((reconstrucao - alvo) ** 2) * pesos).sum(dim=1).mean()


def treinar(
    modelo: AutoencoderDenso,
    treino: np.ndarray,
    validacao: np.ndarray,
    pesos: np.ndarray,
    *,
    seed: int,
    epochs: int = 250,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    paciencia: int = 30,
    min_delta: float = 1e-6,
    device=None,
) -> ResultadoTreino:
    """Treina com early stopping definido somente pela validacao saudavel."""

    exigir_torch()
    configurar_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(device)
    x_treino = torch.as_tensor(treino, dtype=torch.float32)
    x_validacao = torch.as_tensor(validacao, dtype=torch.float32, device=device)
    pesos_t = torch.as_tensor(pesos, dtype=torch.float32, device=device)
    gerador = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_treino),
        batch_size=batch_size,
        shuffle=True,
        generator=gerador,
    )
    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=learning_rate,
        weight_decay=modelo.arquitetura.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, factor=0.5, patience=10
    )

    historico_treino: list[float] = []
    historico_validacao: list[float] = []
    melhor = float("inf")
    melhor_epoca = 0
    melhor_estado = None
    sem_melhora = 0

    for epoca in range(1, epochs + 1):
        modelo.train()
        soma = 0.0
        for (lote_cpu,) in loader:
            lote = lote_cpu.to(device)
            otimizador.zero_grad(set_to_none=True)
            perda = erro_balanceado_torch(modelo(lote), lote, pesos_t)
            perda.backward()
            otimizador.step()
            soma += float(perda.detach().cpu()) * len(lote)
        perda_treino = soma / len(x_treino)

        modelo.eval()
        with torch.no_grad():
            perda_validacao = float(
                erro_balanceado_torch(
                    modelo(x_validacao), x_validacao, pesos_t
                ).cpu()
            )
        historico_treino.append(perda_treino)
        historico_validacao.append(perda_validacao)
        scheduler.step(perda_validacao)

        if perda_validacao < melhor - min_delta:
            melhor = perda_validacao
            melhor_epoca = epoca
            melhor_estado = copy.deepcopy(modelo.state_dict())
            sem_melhora = 0
        else:
            sem_melhora += 1
        if sem_melhora >= paciencia:
            break

    if melhor_estado is None:
        raise RuntimeError("Treino terminou sem estado valido")
    modelo.load_state_dict(melhor_estado)
    return ResultadoTreino(
        historico_treino=historico_treino,
        historico_validacao=historico_validacao,
        melhor_epoca=melhor_epoca,
        melhor_validacao=melhor,
        state_dict=copy.deepcopy(melhor_estado),
    )


def residuos_quadraticos(
    modelo: AutoencoderDenso,
    matriz: np.ndarray,
    *,
    device=None,
    batch_size: int = 256,
) -> np.ndarray:
    exigir_torch()
    device = device or next(modelo.parameters()).device
    modelo.eval()
    x = torch.as_tensor(matriz, dtype=torch.float32)
    partes = []
    with torch.no_grad():
        for inicio in range(0, len(x), batch_size):
            lote = x[inicio : inicio + batch_size].to(device)
            partes.append(((modelo(lote) - lote) ** 2).cpu().numpy())
    return np.concatenate(partes, axis=0)


def pontuar_residuos(residuos: np.ndarray, pesos: np.ndarray) -> np.ndarray:
    matriz = np.asarray(residuos, dtype=float)
    vetor = np.asarray(pesos, dtype=float)
    if matriz.ndim != 2 or matriz.shape[1] != len(vetor):
        raise ValueError("Residuos e pesos possuem dimensoes incompativeis")
    return matriz @ vetor
