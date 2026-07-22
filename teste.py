def calcular_corrente_pico_pv(
    i_mp_stc: float,
    irradiancia: float,
    temperatura_celula: float,
    alpha_imp_pct: float = 0.05,
    g_stc: float = 1000.0,
    t_stc: float = 25.0,
) -> float:
    """Calcula a corrente de pico / máxima potência (I_mp) de um painel fotovoltaico

    para condições operacionais de irradiância e temperatura.

    Parâmetros:
    -----------
    i_mp_stc : float
        Corrente no ponto de máxima potência em STC [A] (do datasheet).
    irradiancia : float
        Irradiância solar incidente no plano do painel [W/m²].
    temperatura_celula : float
        Temperatura de operação da célula fotovoltaica [°C].
    alpha_imp_pct : float, opcional
        Coeficiente de temperatura da corrente [%/°C ou %/K]. Padrão médio: +0.05%/°C.
    g_stc : float, opcional
        Irradiância de referência em STC [W/m²]. Padrão: 1000.0.
    t_stc : float, opcional
        Temperatura de referência em STC [°C]. Padrão: 25.0.

    Retorno:
    --------
    float
        Corrente de pico calculada (I_mp) [A].
    """
    # Converte o coeficiente percentual para escala decimal por °C
    alpha_decimal = alpha_imp_pct / 100.0

    # Correção térmica da corrente
    delta_t = temperatura_celula - t_stc
    i_mp_corrigida_temp = i_mp_stc * (1.0 + alpha_decimal * delta_t)

    # Correção pela irradiância solar
    i_mp_operacional = i_mp_corrigida_temp * (irradiancia / g_stc)

    return i_mp_operacional


# ==============================================================================
# Exemplo Prático de Uso (Painel de ~550Wp comercial)
# ==============================================================================
if __name__ == "__main__":
    # Dados do Datasheet (STC)
    I_MP_STC = 13.20  # Amperes
    ALPHA_IMP = 0.045  # %/°C

    # Condições de Campo
    G_campo = 850.0  # W/m²
    T_celula = 45.0  # °C

    i_pico = calcular_corrente_pico_pv(
        i_mp_stc=I_MP_STC,
        irradiancia=G_campo,
        temperatura_celula=T_celula,
        alpha_imp_pct=ALPHA_IMP,
    )

    print(f"--- Parâmetros de Entrada ---")
    print(f"I_mp (STC): {I_MP_STC} A | G: {G_campo} W/m² | T_célula: {T_celula} °C")
    print(f"\n--- Resultado ---")
    print(f"Corrente de Pico Calculada (I_mp): {i_pico:.2f} A")