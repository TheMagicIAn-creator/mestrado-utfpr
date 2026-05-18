"""
watcher.py — Al IAdo PV
Monitora a pasta novos_pdfs/ em tempo real.
Quando um PDF novo é detectado, processa automaticamente:
  → Renomeia no padrão autor_titulo_ano.pdf
  → Classifica o tema
  → Copia para literatura/<tema>/
  → Indexa no ChromaDB
  → Gera nota no Obsidian

Como usar:
  python watcher.py

Deixe rodando em segundo plano — ele monitora continuamente.
Para parar: Ctrl+C

Autor: Rodolfo Torres (UTFPR)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from watchdog.observers import Observer
from watchdog.events    import FileSystemEventHandler
from sentence_transformers import SentenceTransformer
from src.agente    import MODELO_EMBEDDINGS, PASTA_CHROMADB
from src.processador_pdf import processar_pdf_unico


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PASTA_MONITORADA = Path(__file__).parent / "novos_pdfs"


# ============================================================
# HANDLER DE EVENTOS
# ============================================================

class HandlerPDF(FileSystemEventHandler):
    """
    Detecta quando um PDF novo aparece na pasta monitorada
    e dispara o pipeline de processamento automaticamente.
    """

    def __init__(self, modelo_embeddings):
        self.modelo = modelo_embeddings
        # Controla arquivos já processados para evitar duplo processamento
        self.processados = set()

    def on_created(self, event):
        """Chamado quando um arquivo novo é criado na pasta."""

        if event.is_directory:
            return

        caminho = Path(event.src_path)

        # Só processa PDFs
        if caminho.suffix.lower() != ".pdf":
            return

        # Evita processar o mesmo arquivo duas vezes
        if str(caminho) in self.processados:
            return

        self.processados.add(str(caminho))

        print(f"\n{'='*60}")
        print(f"📄 PDF detectado: {caminho.name}")
        print(f"{'='*60}")

        # Aguarda 2 segundos para garantir que o arquivo foi completamente copiado
        time.sleep(2)

        # Verifica se o arquivo ainda existe (pode ter sido movido)
        if not caminho.exists():
            print(f"⚠️  Arquivo não encontrado após espera: {caminho.name}")
            return

        # Processa o PDF
        resultado = processar_pdf_unico(
            caminho_pdf      = caminho,
            modelo_embeddings= self.modelo,
            pasta_chromadb   = PASTA_CHROMADB,
            gerar_obsidian   = True
        )

        if resultado["sucesso"]:
            print(f"\n✅ Processado com sucesso!")
            print(f"   Autor  : {resultado['autor']}")
            print(f"   Título : {resultado['titulo'][:60]}")
            print(f"   Ano    : {resultado['ano']}")
            print(f"   Tema   : {resultado['tema']}")
            print(f"   Arquivo: {resultado['arquivo_final']}")
            print(f"   Chunks : {resultado['n_chunks']}")
            if resultado["nota_obsidian"]:
                print(f"   Obsidian: ✅ nota gerada")

            # Remove o PDF original da pasta de entrada após processar
            try:
                caminho.unlink()
                print(f"   Limpeza: arquivo original removido de novos_pdfs/")
            except Exception as e:
                print(f"   ⚠️  Não foi possível remover o original: {e}")

        else:
            print(f"\n❌ Erro ao processar: {resultado['erro']}")
            print(f"   O arquivo permanece em novos_pdfs/ para nova tentativa.")
            # Remove da lista de processados para permitir nova tentativa
            self.processados.discard(str(caminho))

        print(f"\n👁️  Monitorando novos_pdfs/ ...")


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():

    # Cria pasta se não existir
    PASTA_MONITORADA.mkdir(exist_ok=True)

    print("=" * 60)
    print("  AL IADO PV — MONITOR DE PDFs")
    print("=" * 60)
    print(f"\n📂 Monitorando: {PASTA_MONITORADA}")
    print(f"\n🔄 Carregando modelo de embeddings...")

    modelo = SentenceTransformer(MODELO_EMBEDDINGS)

    print(f"   ✅ Modelo pronto!")
    print(f"\n👁️  Aguardando PDFs em novos_pdfs/ ...")
    print(f"   Coloque PDFs nessa pasta e eles serão processados automaticamente.")
    print(f"   Para parar: Ctrl+C")
    print(f"\n{'-'*60}")

    # Configura o observador
    handler  = HandlerPDF(modelo)
    observer = Observer()
    observer.schedule(handler, str(PASTA_MONITORADA), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print(f"\n\n⏹️  Monitor encerrado. Até logo!")

    observer.join()


# ============================================================
# PONTO DE ENTRADA
# ============================================================

def iniciar_em_background(modelo_embeddings) -> Observer:
    """
    Inicia o watcher de arquivos E o agendador de consolidação
    em threads de background.
    """
    import schedule
    import threading

    PASTA_MONITORADA.mkdir(exist_ok=True)

    # ── Watcher de arquivos ──────────────────────────────────
    handler  = HandlerPDF(modelo_embeddings)
    observer = Observer()
    observer.schedule(handler, str(PASTA_MONITORADA), recursive=False)
    observer.daemon = True
    observer.start()

    # ── Agendador de consolidação ────────────────────────────
    def job_consolidacao():
        """Roda toda segunda-feira às 8h."""
        try:
            from src.consolidar_memoria import consolidar
            print("\n⏰ Agendador: iniciando consolidação de memória...")
            consolidar()
        except Exception as e:
            print(f"\n⚠️  Erro na consolidação agendada: {e}")

    # Agenda para toda segunda-feira às 08:00
    schedule.every().monday.at("08:00").do(job_consolidacao)

    def loop_agendador():
        while True:
            schedule.run_pending()
            time.sleep(60)  # verifica a cada minuto

    thread_agendador        = threading.Thread(target=loop_agendador)
    thread_agendador.daemon = True
    thread_agendador.start()

    return observer

if __name__ == "__main__":
    main()