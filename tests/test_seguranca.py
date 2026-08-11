"""
Reestruturação — núcleo de cibersegurança (src/core/seguranca.py).

Testes leves (CI torch-free):
- máscara de segredos cobre os formatos de chave usados no projeto;
- anti path-traversal bloqueia escapes e aceita caminhos legítimos;
- nome de arquivo de upload é sanitizado;
- pickle só carrega com SHA-256 conferido (e recusa adulteração);
- ambiente de subprocesso não herda chaves de API.
"""

from __future__ import annotations

import pickle

import pytest

from src.core.seguranca import (
    GUARDA_ANTI_INJECAO,
    RAIZ_PROJETO,
    caminho_dentro_do_projeto,
    carregar_pickle_com_sidecar,
    carregar_pickle_verificado,
    env_minimo_subprocesso,
    gravar_sidecar_sha256,
    mascarar_segredos,
    nome_arquivo_seguro,
    sha256_de_arquivo,
)


# ── máscara de segredos ──────────────────────────────────────────────────────

# Fixtures FAKE montadas por concatenação em runtime: o valor final exercita
# os padrões de mascaramento, mas o literal completo nunca aparece no fonte —
# senão a varredura de segredos do CI (grep) o flagra como falso positivo.
@pytest.mark.parametrize("segredo", [
    "gsk_" + "AbCdEf123456789012345678",
    "AIza" + "SyD-1234567890abcdefghijklmnopqrstu",
    "sk-" + "proj-abcdefghijklmnop1234",
    "hf_" + "abcdefghijklmnop1234",
    "ghp_" + "abcdefghijklmnopqrst1234",
])
def test_mascara_chaves_conhecidas(segredo):
    texto = f"erro ao chamar API com {segredo} no header"
    mascarado = mascarar_segredos(texto)
    assert segredo not in mascarado
    assert "***" in mascarado


def test_mascara_bearer_e_query_string():
    t1 = mascarar_segredos("Authorization: Bearer abcdefghijklmnopqrstuvwxyz123")
    assert "abcdefghijklmnopqrstuvwxyz123" not in t1
    t2 = mascarar_segredos("GET https://api.x.com/v1?key=supersecretvalue123&q=a")
    assert "supersecretvalue123" not in t2
    assert "key=" in t2  # nome do parâmetro preservado para diagnóstico


def test_mascara_valor_de_env_sensivel(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "valor_secreto_da_env_12345")
    saida = mascarar_segredos("falhou com valor_secreto_da_env_12345 na URL")
    assert "valor_secreto_da_env_12345" not in saida


def test_mascara_preserva_texto_normal():
    texto = "AUC=0.93 no conjunto de teste; ver resultados/validacao.json"
    assert mascarar_segredos(texto) == texto


# ── anti path-traversal ──────────────────────────────────────────────────────

def test_caminho_relativo_legitimo_resolve():
    p = caminho_dentro_do_projeto("resultados/experimentos")
    assert str(p).startswith(str(RAIZ_PROJETO))


def test_caminho_traversal_bloqueado():
    with pytest.raises(ValueError):
        caminho_dentro_do_projeto("../../../Windows/System32/config")


def test_caminho_absoluto_externo_bloqueado(tmp_path):
    with pytest.raises(ValueError):
        caminho_dentro_do_projeto(tmp_path / "fora.txt")


def test_caminho_com_base_customizada(tmp_path):
    ok = caminho_dentro_do_projeto("sub/arq.txt", base=tmp_path)
    assert str(ok).startswith(str(tmp_path.resolve()))
    with pytest.raises(ValueError):
        caminho_dentro_do_projeto("../escape.txt", base=tmp_path)


# ── nome de arquivo seguro ───────────────────────────────────────────────────

@pytest.mark.parametrize("entrada,proibido", [
    ("../../etc/passwd", ".."),
    ("..\\..\\windows\\system32\\x.pdf", "\\"),
    ("artigo<script>.pdf", "<"),
    ("con|trole?.pdf", "|"),
])
def test_nome_arquivo_remove_perigosos(entrada, proibido):
    saida = nome_arquivo_seguro(entrada)
    assert proibido not in saida
    assert saida  # nunca vazio


def test_nome_arquivo_normal_preservado():
    assert nome_arquivo_seguro("Stender (2020) Dataset.pdf") == \
        "Stender (2020) Dataset.pdf"


def test_nome_arquivo_vazio_vira_padrao():
    assert nome_arquivo_seguro("") == "arquivo"
    assert nome_arquivo_seguro("...") == "arquivo"


# ── pickle verificado ────────────────────────────────────────────────────────

def test_pickle_carrega_com_hash_correto(tmp_path):
    alvo = tmp_path / "modelo.pkl"
    alvo.write_bytes(pickle.dumps({"w": [1, 2, 3]}))
    h = sha256_de_arquivo(alvo)
    obj = carregar_pickle_verificado(alvo, h)
    assert obj == {"w": [1, 2, 3]}


def test_pickle_recusa_arquivo_adulterado(tmp_path):
    alvo = tmp_path / "modelo.pkl"
    alvo.write_bytes(pickle.dumps({"w": [1]}))
    h = sha256_de_arquivo(alvo)
    alvo.write_bytes(pickle.dumps({"w": ["ADULTERADO"]}))  # troca pós-hash
    with pytest.raises(ValueError, match="Integridade"):
        carregar_pickle_verificado(alvo, h)


def test_pickle_exige_hash():
    with pytest.raises(ValueError):
        carregar_pickle_verificado("qualquer.pkl", "")


def test_mensagem_de_integridade_ensina_a_sair_do_erro(tmp_path):
    """A mensagem antiga mandava "retreinar ou restaurar" e nada mais.

    O pesquisador seguiu e retreinou — mas o pickle dele estava intacto: o que
    tinha divergido era o SIDECAR, chegado por `git pull` da máquina do outro
    agente. Meia hora perdida por uma mensagem que só nomeava a causa rara.

    Adulteração é possível, e a guarda existe para ela. Par dessincronizado é
    muito mais provável — a mensagem precisa dizer como distinguir, e dar o
    comando de saída quando o artefato é reconhecidamente do próprio usuário.
    """
    alvo = tmp_path / "scaler.pkl"
    alvo.write_bytes(pickle.dumps({"w": [1]}))

    with pytest.raises(ValueError) as exc:
        carregar_pickle_verificado(alvo, "0" * 64)
    msg = str(exc.value)

    assert "execuções diferentes" in msg, "não nomeia a causa provável"
    assert "gravar_sidecar_sha256" in msg, "não dá o comando de saída"
    assert "não regenere" in msg, (
        "precisa dizer o que fazer quando o artefato NÃO é reconhecido — "
        "regenerar às cegas anula a proteção contra pickle adulterado"
    )
    assert str(alvo) in msg, "o caminho tem de estar pronto para copiar"


def test_sidecar_grava_e_verifica(tmp_path):
    alvo = tmp_path / "scaler.pkl"
    alvo.write_bytes(pickle.dumps([1.0, 2.0]))
    side = gravar_sidecar_sha256(alvo)
    assert side.name == "scaler.pkl.sha256"
    assert carregar_pickle_com_sidecar(alvo) == [1.0, 2.0]
    # adulteração pós-sidecar → recusa
    alvo.write_bytes(pickle.dumps(["MAU"]))
    with pytest.raises(ValueError, match="Integridade"):
        carregar_pickle_com_sidecar(alvo)


def test_sidecar_ausente_carrega_com_aviso(tmp_path):
    alvo = tmp_path / "legado.pkl"
    alvo.write_bytes(pickle.dumps({"ok": True}))
    # sem sidecar: caminho legado funciona (com warning no log)
    assert carregar_pickle_com_sidecar(alvo) == {"ok": True}


# ── env mínimo de subprocesso ────────────────────────────────────────────────

def test_env_minimo_remove_chaves(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x" * 20)
    monkeypatch.setenv("GOOGLE_API_KEY", "y" * 20)
    monkeypatch.setenv("PATH_QUALQUER", "ok")
    env = env_minimo_subprocesso()
    assert "GROQ_API_KEY" not in env
    assert "GOOGLE_API_KEY" not in env
    assert env.get("PATH_QUALQUER") == "ok"
    assert env.get("KMP_DUPLICATE_LIB_OK") == "TRUE"


def test_env_minimo_extras_sobrepoe():
    env = env_minimo_subprocesso(extras={"AL_IADO_EXP_CHILD": "1"})
    assert env["AL_IADO_EXP_CHILD"] == "1"


# ── guarda anti-injeção ──────────────────────────────────────────────────────

def test_guarda_anti_injecao_existe_e_e_imperativa():
    assert "DADO" in GUARDA_ANTI_INJECAO
    assert "instrução" in GUARDA_ANTI_INJECAO


# ── Sidecar de integridade nunca pode viajar sem o artefato ────────────────

def test_nenhum_sidecar_sha256_de_artefato_ignorado_esta_rastreado():
    """Um checksum versionado de um arquivo NÃO versionado quebra quem der pull.

    Aconteceu de fato em 11/08/2026: `resultados/autoencoder/scaler.pkl` é
    ignorado (`.gitignore: *.pkl`), mas `scaler.pkl.sha256` estava rastreado.
    O `git pull` entregou ao pesquisador o hash gerado na máquina do outro
    agente sobre o pickle gerado na dele, e `carregar_pickle_com_sidecar`
    abortou com "Integridade violada" — que soa como adulteração, quando era
    só o par ter sido separado pelo versionamento.

    A regra `*.pkl.sha256` já existia no .gitignore, mas gitignore não afeta
    arquivo já rastreado: a regra estava inerte. Este teste é o que torna a
    inércia visível.

    O sidecar é POR MÁQUINA, por construção — ele descreve o artefato local.
    Versioná-lo é sempre erro.
    """
    import subprocess

    from src.core.config import RAIZ_PROJETO

    rastreados = subprocess.run(
        ["git", "ls-files", "*.sha256"],
        cwd=RAIZ_PROJETO, capture_output=True, text=True,
    ).stdout.split()
    if not rastreados:
        return

    problemas = []
    for sidecar in rastreados:
        artefato = sidecar[: -len(".sha256")]
        ignorado = subprocess.run(
            ["git", "check-ignore", "-q", artefato],
            cwd=RAIZ_PROJETO, capture_output=True,
        ).returncode == 0
        if ignorado:
            problemas.append(f"{sidecar} (o artefato {artefato} é ignorado)")

    assert not problemas, (
        "sidecar de integridade rastreado para artefato NÃO versionado — "
        "quem der pull vai receber 'Integridade violada':\n  "
        + "\n  ".join(problemas)
        + "\nCorreção: git rm --cached <sidecar>"
    )
