import sympy as sp


def integral_simbolica_definida(expressao_str, variavel_str, a, b):
    """Calcula a integral definida exata usando expressão em string.

    Exemplo de expressao_str: 'x**2 + 3*x'
    """
    x = sp.Symbol(variavel_str)
    expressao = sp.sympify(expressao_str)

    # Integral exata
    resultado_exato = sp.integrate(expressao, (x, a, b))
    return resultado_exato, float(resultado_exato)


# --- EXEMPLO DE USO ---
expr = "x**2"
lim_inf, lim_sup = 0, 2

exato, numerico = integral_simbolica_definida(expr, "x", lim_inf, lim_sup)

print(f"Resultado Exato: {exato}")  # Retorna 8/3
print(f"Resultado Decimal: {numerico:.6f}")  # Retorna 2.666667