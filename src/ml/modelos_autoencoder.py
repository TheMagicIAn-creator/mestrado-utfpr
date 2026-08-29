"""Modelos canônicos para a comparação Autoencoder Denso versus AE-LSTM.

Os dois modelos recebem as mesmas 24 features normalizadas do GPVS-Faults.
Treino e validação são sempre blocos distintos; calibração do limiar e teste
ficam fora deste módulo para impedir reutilização acidental de amostras.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:  # CI leve importa o contrato sem instalar torch.
    torch = None
    nn = None
    DataLoader = TensorDataset = None


DENSE_HIDDEN = 16
LATENT_DIM = 8
LSTM_HIDDEN = 32
SEQUENCE_LENGTH = 8
SCORE_TOP_K = 5
LEARNING_RATE = 1e-3
MAX_EPOCHS = 150
PATIENCE = 20
BATCH_SIZE = 32
DROPOUT = 0.2


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch é necessário para treinar os autoencoders. "
            "Instale requirements-ml.txt no ambiente virtual."
        )


if nn is not None:

    class AutoencoderDenso(nn.Module):
        """Arquitetura simétrica 24-16-8-16-24 com saída linear."""

        def __init__(self, n_features: int, dropout: float = DROPOUT):
            super().__init__()
            self.n_features = int(n_features)
            self.encoder = nn.Sequential(
                nn.Linear(self.n_features, DENSE_HIDDEN),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(DENSE_HIDDEN, LATENT_DIM),
            )
            self.decoder = nn.Sequential(
                nn.Linear(LATENT_DIM, DENSE_HIDDEN),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(DENSE_HIDDEN, self.n_features),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))


    class AutoencoderLSTM(nn.Module):
        """Autoencoder recorrente que reconstrói sequências no eixo temporal."""

        def __init__(
            self,
            n_features: int,
            hidden_size: int = LSTM_HIDDEN,
            latent_dim: int = LATENT_DIM,
        ):
            super().__init__()
            self.n_features = int(n_features)
            self.hidden_size = int(hidden_size)
            self.latent_dim = int(latent_dim)
            self.encoder = nn.LSTM(
                self.n_features, self.hidden_size, batch_first=True
            )
            self.to_latent = nn.Linear(self.hidden_size, self.latent_dim)
            self.from_latent = nn.Linear(self.latent_dim, self.hidden_size)
            self.decoder = nn.LSTM(
                self.hidden_size, self.hidden_size, batch_first=True
            )
            self.output = nn.Linear(self.hidden_size, self.n_features)

        def forward(self, x):
            _, (hidden, _) = self.encoder(x)
            latent = self.to_latent(hidden[-1])
            decoder_input = self.from_latent(latent).unsqueeze(1).repeat(
                1, x.size(1), 1
            )
            decoded, _ = self.decoder(decoder_input)
            return self.output(decoded)

else:

    class AutoencoderDenso:  # pragma: no cover - mensagem exercitada no runtime
        def __init__(self, *_args, **_kwargs):
            _require_torch()


    class AutoencoderLSTM:  # pragma: no cover - mensagem exercitada no runtime
        def __init__(self, *_args, **_kwargs):
            _require_torch()


@dataclass(frozen=True)
class TrainingHistory:
    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    best_epoch: int
    stopped_epoch: int

    @property
    def best_validation_loss(self) -> float:
        return float(min(self.validation_loss))


def set_deterministic_seed(seed: int) -> None:
    """Fixa as fontes de aleatoriedade usadas pelo treino em CPU/GPU."""

    _require_torch()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        pass


def _train_autoencoder(
    model,
    train_values: np.ndarray,
    validation_values: np.ndarray,
    *,
    seed: int,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    device=None,
) -> TrainingHistory:
    _require_torch()
    set_deterministic_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_tensor = torch.as_tensor(train_values, dtype=torch.float32)
    validation_tensor = torch.as_tensor(validation_values, dtype=torch.float32)
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=min(int(batch_size), len(train_tensor)),
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loss_function = nn.MSELoss()
    history_train: list[float] = []
    history_validation: list[float] = []
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        batch_losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch), batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            validation_batch = validation_tensor.to(device)
            validation_loss = float(
                loss_function(model(validation_batch), validation_batch).cpu()
            )
        history_train.append(float(np.mean(batch_losses)))
        history_validation.append(validation_loss)

        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break

    model.load_state_dict(best_state)
    model.eval()
    return TrainingHistory(
        train_loss=tuple(history_train),
        validation_loss=tuple(history_validation),
        best_epoch=int(best_epoch),
        stopped_epoch=len(history_train),
    )


def train_dense(
    train_values: np.ndarray,
    validation_values: np.ndarray,
    *,
    seed: int,
    **kwargs,
):
    """Treina o AE denso usando apenas os blocos saudável de treino/validação."""

    _require_torch()
    set_deterministic_seed(seed)
    model = AutoencoderDenso(int(np.asarray(train_values).shape[1]))
    history = _train_autoencoder(
        model, train_values, validation_values, seed=seed, **kwargs
    )
    return model, history


def train_lstm(
    train_sequences: np.ndarray,
    validation_sequences: np.ndarray,
    *,
    seed: int,
    **kwargs,
):
    """Treina o AE-LSTM sem reutilizar sequências de calibração."""

    _require_torch()
    set_deterministic_seed(seed)
    model = AutoencoderLSTM(int(np.asarray(train_sequences).shape[2]))
    history = _train_autoencoder(
        model, train_sequences, validation_sequences, seed=seed, **kwargs
    )
    return model, history


def _top_k_feature_mse(reconstructed, target, *, top_k: int):
    """Média dos ``k`` maiores erros quadráticos no eixo de features."""

    if reconstructed.shape != target.shape:
        raise ValueError("Reconstrução e alvo devem possuir o mesmo shape")
    n_features = int(target.shape[-1])
    if isinstance(top_k, bool) or int(top_k) != top_k:
        raise ValueError("top_k deve ser um número inteiro")
    normalized_k = int(top_k)
    if not 1 <= normalized_k <= n_features:
        raise ValueError(f"top_k deve estar entre 1 e {n_features}")
    squared_error = (reconstructed - target) ** 2
    largest = torch.topk(
        squared_error,
        k=normalized_k,
        dim=-1,
        largest=True,
        sorted=False,
    ).values
    return largest.mean(dim=-1)


def score_dense(
    model,
    values: np.ndarray,
    *,
    top_k: int = SCORE_TOP_K,
    device=None,
) -> np.ndarray:
    """Escore localizado nas ``top_k`` features de cada janela."""

    _require_torch()
    device = device or next(model.parameters()).device
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        reconstructed = model(tensor)
        return _top_k_feature_mse(
            reconstructed,
            tensor,
            top_k=top_k,
        ).cpu().numpy()


def score_lstm(
    model,
    sequences: np.ndarray,
    *,
    top_k: int = SCORE_TOP_K,
    device=None,
) -> np.ndarray:
    """Top-k no último passo, condicionado ao histórico temporal da sequência."""

    _require_torch()
    device = device or next(model.parameters()).device
    tensor = torch.as_tensor(sequences, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        reconstructed = model(tensor)
        return _top_k_feature_mse(
            reconstructed[:, -1, :],
            tensor[:, -1, :],
            top_k=top_k,
        ).cpu().numpy()


def sequences_from_flow(values: np.ndarray, length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Produz uma sequência por janela, com padding apenas no início do bloco."""

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) == 0:
        raise ValueError("O fluxo deve ter shape (n_janelas, n_features) e não ser vazio")
    length = int(length)
    if length < 2:
        raise ValueError("A sequência temporal deve ter ao menos dois passos")
    padded = np.vstack([np.repeat(matrix[:1], length - 1, axis=0), matrix])
    return np.stack([padded[i : i + length] for i in range(len(matrix))])


def sequences_for_blocks(
    values: np.ndarray,
    blocks: list[np.ndarray] | tuple[np.ndarray, ...],
    length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Cria sequências sem atravessar ensaios nem fronteiras de papéis."""

    matrix = np.asarray(values, dtype=np.float32)
    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for block in blocks:
        indices = np.asarray(block, dtype=int)
        if not len(indices):
            continue
        if np.any(np.diff(indices) != 1):
            raise ValueError("Cada bloco temporal deve conter índices contíguos")
        sequences.append(sequences_from_flow(matrix[indices], length))
        targets.append(indices)
    if not sequences:
        raise ValueError("Nenhum bloco temporal válido foi fornecido")
    return np.concatenate(sequences), np.concatenate(targets)


def sequences_with_current_values(
    healthy_context: np.ndarray,
    current_values: np.ndarray,
    groups: np.ndarray | list[str],
    length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """Substitui apenas o último passo e preserva contexto saudável por ensaio."""

    context = np.asarray(healthy_context, dtype=np.float32)
    current = np.asarray(current_values, dtype=np.float32)
    labels = np.asarray(groups).astype(str)
    if context.shape != current.shape or len(labels) != len(context):
        raise ValueError("Contexto, itens atuais e grupos devem estar alinhados")
    output = np.empty((len(context), int(length), context.shape[1]), dtype=np.float32)
    for label in dict.fromkeys(labels.tolist()):
        indices = np.flatnonzero(labels == label)
        base_sequences = sequences_from_flow(context[indices], length)
        base_sequences[:, -1, :] = current[indices]
        output[indices] = base_sequences
    return output


def parameter_count(model) -> int:
    _require_torch()
    return int(sum(parameter.numel() for parameter in model.parameters()))


__all__ = [
    "AutoencoderDenso",
    "AutoencoderLSTM",
    "BATCH_SIZE",
    "DENSE_HIDDEN",
    "DROPOUT",
    "LATENT_DIM",
    "LEARNING_RATE",
    "LSTM_HIDDEN",
    "MAX_EPOCHS",
    "PATIENCE",
    "SEQUENCE_LENGTH",
    "SCORE_TOP_K",
    "TrainingHistory",
    "parameter_count",
    "score_dense",
    "score_lstm",
    "sequences_for_blocks",
    "sequences_from_flow",
    "sequences_with_current_values",
    "set_deterministic_seed",
    "train_dense",
    "train_lstm",
]
