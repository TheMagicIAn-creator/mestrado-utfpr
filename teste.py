import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def plot_weibull_analysis(beta: float, eta: float, t_max: float = 2000):
    """
    Gera o gráfico de análise de confiabilidade Weibull para componentes elétricos/mecânicos.

    Parâmetros:
    -----------
    beta : float
        Parâmetro de forma (shape). β > 1 indica envelhecimento/desgaste.
    eta : float
        Parâmetro de escala (scale / vida característica em horas).
    t_max : float
        Tempo máximo para o eixo X do gráfico.
    """
    # 1. Métricas da Distribuição
    mttf = eta * gamma(1 + 1 / beta)
    b10 = eta * (-np.log(0.90)) ** (1 / beta)

    # 2. Vetor de Tempo
    t = np.linspace(1, t_max, 500)

    # 3. Equações Fundamentais de Weibull
    F_t = 1 - np.exp(-(t / eta) ** beta)  # Inconfiabilidade (CDF)
    R_t = np.exp(-(t / eta) ** beta)      # Confiabilidade R(t)

    # 4. Plotagem da Curva de Confiabilidade / Inconfiabilidade
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=120)

    # Eixo Esquerdo: R(t)
    ax1.plot(t, R_t * 100, color='#1f77b4', linewidth=2.5, label=r'Confiabilidade $R(t)$')
    ax1.set_xlabel('Tempo de Operação $t$ (horas)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Confiabilidade $R(t)$ (%)', color='#1f77b4', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Eixo Direito: F(t)
    ax2 = ax1.twinx()
    ax2.plot(t, F_t * 100, color='#d62728', linestyle='--', linewidth=2, label=r'Inconfiabilidade $F(t)$')
    ax2.set_ylabel('Probabilidade Acumulada de Falha $F(t)$ (%)', color='#d62728', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#d62728')

    # Destaques visuais: B10 e MTTF
    ax1.axvline(b10, color='#ff7f0e', linestyle=':', linewidth=1.8, label=f'B10 = {b10:.1f} h')
    ax1.axvline(mttf, color='#2ca02c', linestyle=':', linewidth=1.8, label=f'MTTF = {mttf:.1f} h')

    # Consolidação de Legendas
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', framealpha=0.9)

    plt.title(f'Análise de Confiabilidade de Weibull ($\\beta = {beta}$, $\\eta = {eta:.0f}$ h)',
              fontsize=13, fontweight='bold', pad=12)
    fig.tight_layout()
    plt.show()

def plot_weibull_linearized(beta: float, eta: float, num_samples: int = 50):
    """
    Gera o gráfico de probabilidade de Weibull no espaço linearizado:
    Y = ln(-ln(1 - F(t)))  vs  X = ln(t)
    """
    # Simulação de dados Amostrais baseados nos parâmetros
    np.random.seed(42)
    ttf_samples = np.sort(eta * np.random.weibull(beta, num_samples))

    # Estimador de Median Ranks (Aproximação de Benard)
    i = np.arange(1, num_samples + 1)
    F_i = (i - 0.3) / (num_samples + 0.4)

    # Transformação Linear
    x_linear = np.log(ttf_samples)
    y_linear = np.log(-np.log(1 - F_i))

    # Regressão Teórica
    x_grid = np.linspace(min(x_linear), max(x_linear), 100)
    y_grid = beta * x_grid - beta * np.log(eta)

    plt.figure(figsize=(8, 5), dpi=120)
    plt.scatter(x_linear, y_linear, color='#1f77b4', edgecolor='k', alpha=0.7, label='Amostras (Median Ranks)')
    plt.plot(x_grid, y_grid, color='#d62728', linewidth=2, label=f'Ajuste Teórico (Ajuste $\\beta={beta}$)')

    plt.xlabel('$\\ln(t)$ [Tempo em escala logarítmica]', fontsize=11, fontweight='bold')
    plt.ylabel('$\\ln(-\\ln(1 - F(t)))$ [Inconfiabilidade Transformada]', fontsize=11, fontweight='bold')
    plt.title('Gráfico de Probabilidade de Weibull (Linearizado)', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

# --- Execução Exemplo ---
if __name__ == "__main__":
    # Parâmetros típicos de falha por desgaste em componente elétrico CA
    BETA_EXEMPLO = 2.5   # β > 1 (Taxa de falhas crescente por desgaste)
    ETA_EXEMPLO = 1200.0 # η = 1200h (Vida característica)

    # 1. Gráfico Principal de Confiabilidade / Inconfiabilidade
    plot_weibull_analysis(beta=BETA_EXEMPLO, eta=ETA_EXEMPLO)

    # 2. Gráfico Linearizado (Papel Weibull)
    plot_weibull_linearized(beta=BETA_EXEMPLO, eta=ETA_EXEMPLO)