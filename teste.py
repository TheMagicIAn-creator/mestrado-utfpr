import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import weibull_min, lognorm

# ==============================================================================
# FUNÇÕES DE AJUSTE MLE PARA DADOS CENSURADOS À DIREITA
# ==============================================================================

def neg_log_likelihood_weibull(params, ttf, event):
    """
    Log-verossimilhança negativa da distribuição Weibull com censura à direita.
    params: [beta (forma), eta (escala)]
    ttf: array com tempos até falha ou tempo de censura
    event: array binário (1 = falha observada, 0 = censurado à direita)
    """
    beta, eta = params
    if beta <= 0 or eta <= 0:
        return 1e10

    # Eventos observados: f(t)
    # Eventos censurados: S(t) = 1 - F(t) = exp(-(t/eta)^beta)
    log_f = np.log(beta / eta) + (beta - 1) * np.log(ttf / eta) - (ttf / eta)**beta
    log_s = - (ttf / eta)**beta

    log_lik = np.sum(event * log_f + (1 - event) * log_s)
    return -log_lik

def fit_censored_weibull(ttf, event):
    """Ajusta Weibull via MLE considerando censura."""
    n_failures = np.sum(event)
    if n_failures < 3:
        return None  # Amostra insuficiente

    # Chute inicial razoável baseado nos eventos observados
    obs_ttf = ttf[event == 1]
    eta_init = np.median(obs_ttf)
    beta_init = 2.0

    res = minimize(
        neg_log_likelihood_weibull,
        x0=[beta_init, eta_init],
        args=(ttf, event),
        bounds=[(1e-3, None), (1e-3, None)],
        method='L-BFGS-B'
    )

    if res.success:
        beta_fit, eta_fit = res.x
        # B10: tempo para 10% de falhas acumuladas
        b10 = eta_fit * (-np.log(0.90))**(1.0 / beta_fit)
        return {"beta": beta_fit, "eta": eta_fit, "b10": b10, "n_fail": n_failures}
    return None

# ==============================================================================
# SCRIPT DE PLOTAGEM CORRIGIDO
# ==============================================================================

def plot_ttf_distributions(data_dict, title_prefix="Análise de Confiabilidade CA"):
    """
    Gera o painel de distribuição TTF corrigido.

    data_dict: dicionário no formato:
      {
        "Componente": {"ttf": np.array([...]), "event": np.array([...])},
        ...
      }
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
    colors = {'Contator AC': '#1f77b4', 'IGBT': '#2ca02c', 'Fusível AC': '#d62728'}

    for ax, (comp_name, comp_data) in zip(axes, data_dict.items()):
        ttf = comp_data['ttf']
        event = comp_data['event']

        ttf_obs = ttf[event == 1]
        n_obs = len(ttf_obs)
        n_cens = np.sum(event == 0)
        c_color = colors.get(comp_name, '#333333')

        # 1. Plot do Histograma (Apenas Falhas Observadas)
        if n_obs > 0:
            count, bins, _ = ax.hist(
                ttf_obs,
                bins='auto',
                density=True,
                alpha=0.4,
                color=c_color,
                edgecolor='black',
                label=f'Falhas Obs. (n={n_obs})'
            )
        else:
            ax.text(0.5, 0.5, "Sem falhas observadas", ha='center', va='center', transform=ax.transAxes)

        # 2. Ajuste MLE e Plotagem da PDF Ajustada
        fit_res = fit_censored_weibull(ttf, event)

        if fit_res is not None:
            beta = fit_res['beta']
            eta = fit_res['eta']
            b10 = fit_res['b10']

            # Eixo x para a curva teórica contínua
            t_max = np.max(ttf) * 1.5
            t_grid = np.linspace(0.5, t_max, 300)

            # PDF Weibull: f(t) = (beta/eta)*(t/eta)^(beta-1) * exp(-(t/eta)^beta)
            pdf_grid = weibull_min.pdf(t_grid, c=beta, scale=eta)

            ax.plot(t_grid, pdf_grid, color=c_color, lw=2.5,
                    label=f'Weibull MLE\n($\\beta$={beta:.2f}, $\\eta$={eta:.1f}h)')

            # Marcação do B10
            ax.axvline(b10, color='black', linestyle='--', alpha=0.7,
                       label=f'$B_{{10}}$ = {b10:.1f}h')

            title_text = f"{comp_name}\n(\\beta={beta:.2f}, \\eta={eta:.1f}h, B10={b10:.1f}h)"
        else:
            title_text = f"{comp_name}\n(Dados Insuficientes / Alta Censura)"
            ax.text(0.5, 0.3, f"Censurados: {n_cens}\nFalhas: {n_obs}",
                    ha='center', va='center', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_title(title_text, fontsize=11, fontweight='bold')
        ax.set_xlabel("Tempo até Falha - TTF (horas)")
        ax.set_ylabel("Densidade de Probabilidade $f(t)$")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=8)

        # Indica o ponto de encerramento do teste (censura)
        t_cens_max = np.max(ttf[event == 0]) if n_cens > 0 else None
        if t_cens_max:
            ax.axvline(t_cens_max, color='red', linestyle=':', alpha=0.5, label='Fim do Ensaio')

    plt.suptitle(title_prefix, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig, axes

# ==============================================================================
# EXEMPLO DE USO / TESTE SINTÉTICO COM O SEU PERFIL DE DADOS
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)

    # Exemplo simulando o cenário real corrigido (janela de ensaio = 120h)
    T_MAX = 120.0

    # 1. Contator AC (Moderado tempo de falha)
    ttf_contator_raw = np.random.weibull(a=2.1, size=30) * 80
    event_contator = (ttf_contator_raw <= T_MAX).astype(int)
    ttf_contator = np.minimum(ttf_contator_raw, T_MAX)

    # 2. IGBT (Falhas por degradação térmica)
    ttf_igbt_raw = np.random.weibull(a=3.5, size=30) * 110
    event_igbt = (ttf_igbt_raw <= T_MAX).astype(int)
    ttf_igbt = np.minimum(ttf_igbt_raw, T_MAX)

    # 3. Fusível AC (Alta confiabilidade, quase sem falhas em 120h)
    ttf_fusivel_raw = np.random.weibull(a=1.2, size=30) * 800
    event_fusivel = (ttf_fusivel_raw <= T_MAX).astype(int)
    ttf_fusivel = np.minimum(ttf_fusivel_raw, T_MAX)

    dados_simulados = {
        "Contator AC": {"ttf": ttf_contator, "event": event_contator},
        "IGBT": {"ttf": ttf_igbt, "event": event_igbt},
        "Fusível AC": {"ttf": ttf_fusivel, "event": event_fusivel}
    }

    fig, _ = plot_ttf_distributions(dados_simulados)
    plt.savefig("weibull_ttf_corrigido.png", dpi=100, bbox_inches='tight')
    plt.show()