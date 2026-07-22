@echo off
REM ============================================================
REM  Al IAdo PV - Atualizar vault (baixa do GitHub para o PC)
REM ------------------------------------------------------------
REM  O app na web (Streamlit Cloud) commita sessoes, memorias
REM  consolidadas e trechos salvos no GitHub. Este script traz
REM  esses commits para o seu PC, para o Obsidian mostrar tudo.
REM  Basta dar dois cliques neste arquivo antes de abrir o vault.
REM ============================================================

cd /d "%~dp0"

echo.
echo === Atualizando o vault a partir do GitHub ===
echo.

REM Driver de merge que mantem seus resultados/ locais (roda uma vez; idempotente).
git config merge.ours.driver true

git pull origin main
if %errorlevel% neq 0 goto :falhou

echo.
echo [OK] Vault atualizado. Pode abrir o Obsidian - as novas sessoes e
echo      memorias consolidadas estarao la.
goto :fim

:falhou
echo.
echo [!] O pull encontrou um problema - provavelmente um conflito de
echo     notas editadas nos dois lados. Rode  git status  para ver,
echo     ou peca ajuda ao Al IAdo PV.

:fim
echo.
pause
