import sys
import sympy as sp


def menu():
    print("=" * 55)
    print("      CALCULADORA INTERATIVA DE INTEGRAIS (SymPy)     ")
    print("=" * 55)
    print("1 - Integral Indefinida (Simples / Primitiva)")
    print("2 - Integral Definida (com limites a e b)")
    print("0 - Sair")
    print("-" * 55)


def ler_expressao(var_sym):
    """Lê a expressão do usuário e converte para objeto do SymPy."""
    while True:
        raw_expr = input("\nDigite a função f(x) (ex: x**2 + 3*x, sin(x), exp(-x)): ").strip()
        try:
            # Transforma a string em expressão matemática reconhecida pelo SymPy
            expr = sp.sympify(raw_expr)
            return expr
        except Exception as e:
            print(f"❌ Erro ao interpretar a expressão: {e}")
            print("Dica: Use '*' para multiplicação (ex: 2*x) e '**' para potência (ex: x**2).")


def resolver_indefinida(x):
    expr = ler_expressao(x)
    print("\nCalculando...")
    resultado = sp.integrate(expr, x)

    print("\n" + "—" * 40)
    print(f"📥 Função original : f(x) = {expr}")
    print(f"📤 Integral∫f(x)dx : {resultado} + C")
    print("—" * 40)


def resolver_definida(x):
    expr = ler_expressao(x)

    try:
        lim_inf_str = input("Digite o limite INFERIOR (a): ").strip()
        lim_sup_str = input("Digite o limite SUPERIOR (b): ").strip()

        # Converte os limites (aceita frações como 1/2, pi como sp.pi, etc.)
        lim_inf = sp.sympify(lim_inf_str)
        lim_sup = sp.sympify(lim_sup_str)

        print("\nCalculando...")
        res_exato = sp.integrate(expr, (x, lim_inf, lim_sup))

        print("\n" + "—" * 40)
        print(f"📥 Função original : f(x) = {expr}")
        print(f"📍 Intervalo       : [{lim_inf}, {lim_sup}]")
        print(f"📤 Resultado exato : {res_exato}")

        # Se o resultado puder ser convertido para decimal
        try:
            res_num = float(res_exato.evalf())
            print(f"🔢 Valor decimal   : {res_num:.6f}")
        except Exception:
            pass
        print("—" * 40)

    except Exception as e:
        print(f"❌ Erro no cálculo dos limites: {e}")


def main():
    # Define a variável simbólica 'x'
    x = sp.Symbol('x')

    while True:
        menu()
        opcao = input("Escolha uma opção (0, 1 ou 2): ").strip()

        if opcao == '1':
            resolver_indefinida(x)
        elif opcao == '2':
            resolver_definida(x)
        elif opcao == '0':
            print("\nSaindo da calculadora. Até logo!")
            sys.exit(0)
        else:
            print("\n❌ Opção inválida! Tente novamente.")

        input("\nPressione ENTER para continuar...")


if __name__ == "__main__":
    main()