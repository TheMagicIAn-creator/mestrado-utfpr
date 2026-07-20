from __future__ import annotations

import json

import pytest

from src.conhecimento.memoria_persistente import (
    MemoriaCorrompida,
    MemoriaInvalida,
    MemoriaPersistente,
)


def _candidato(**extras):
    dados = {
        "tipo": "preferencia",
        "escopo": "conversa",
        "conteudo": "Prefere respostas diretas em portugues.",
        "evidencia_usuario": "Prefiro respostas diretas em portugues.",
    }
    dados.update(extras)
    return dados


def test_memoria_registra_deduplica_e_recupera(tmp_path):
    memoria = MemoriaPersistente(tmp_path / "memoria.json")

    primeiro = memoria.registrar(
        _candidato(),
        origem="teste",
        validado_por="Groq",
        confianca=0.93,
    )
    repetido = memoria.registrar(
        _candidato(),
        origem="teste",
        validado_por="Groq",
        confianca=0.96,
    )

    assert primeiro.criado is True
    assert repetido.criado is False
    assert memoria.contar() == 1
    assert memoria.recuperar("Como devo formatar a resposta?")[0]["id"] == primeiro.item["id"]
    assert "id=" in memoria.formatar_para_prompt("resposta em portugues")


def test_memoria_supera_item_anterior(tmp_path):
    memoria = MemoriaPersistente(tmp_path / "memoria.json")
    anterior = memoria.registrar(
        _candidato(), origem="teste", validado_por="Groq", confianca=0.9
    ).item
    novo = memoria.registrar(
        _candidato(
            conteudo="Prefere respostas detalhadas em portugues.",
            evidencia_usuario="Agora prefiro respostas detalhadas em portugues.",
            substitui_id=anterior["id"],
        ),
        origem="teste",
        validado_por="Groq",
        confianca=0.94,
    )

    assert novo.criado is True
    assert memoria.contar() == 1
    todos = memoria.listar(somente_ativas=False)
    assert {item["status"] for item in todos} == {"ativo", "superado"}


@pytest.mark.parametrize(
    "candidato,confianca",
    [
        (_candidato(conteudo="Minha API_KEY=AIza" + "A" * 35), 0.99),
        (
            _candidato(
                tipo="contexto_projeto",
                conteudo="O AUC atual e 0.97 no teste local.",
                evidencia_usuario="O AUC atual e 0.97 no teste local.",
            ),
            0.99,
        ),
        (_candidato(), 0.4),
    ],
)
def test_memoria_rejeita_segredos_metricas_e_baixa_confianca(
    tmp_path, candidato, confianca
):
    memoria = MemoriaPersistente(tmp_path / "memoria.json")
    with pytest.raises(MemoriaInvalida):
        memoria.registrar(
            candidato,
            origem="teste",
            validado_por="Groq",
            confianca=confianca,
        )


def test_memoria_corrompida_nao_e_sobrescrita(tmp_path):
    caminho = tmp_path / "memoria.json"
    caminho.write_text("{invalido", encoding="utf-8")
    memoria = MemoriaPersistente(caminho)

    assert memoria.listar() == []
    with pytest.raises(MemoriaCorrompida):
        memoria.registrar(
            _candidato(), origem="teste", validado_por="Groq", confianca=0.9
        )
    assert caminho.read_text(encoding="utf-8") == "{invalido"


def test_snapshot_tem_schema_explicito(tmp_path):
    caminho = tmp_path / "memoria.json"
    memoria = MemoriaPersistente(caminho)
    memoria.registrar(
        _candidato(), origem="teste", validado_por="Groq", confianca=0.9
    )
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    assert dados["schema_version"] == 1
    assert dados["itens"][0]["validado_por"] == "Groq"
