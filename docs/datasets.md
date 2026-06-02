# Datasets — Al IAdo PV

Os dados brutos **não são versionados** (ficam em `dados/brutos/`, ignorados no
git). Valide-os localmente com `python scripts/verificar_datasets.py` (gera
`dados/dataset_manifest.json` com SHA-256, linhas e classes).

## Paderborn — domínio **CA** (eixo principal)
- Arquivo: `dados/brutos/Inverter_Data_Set.csv`
- Inversor IGBT trifásico **saudável**, ~235 mil amostras, 10 kHz.
- **Sem rótulos** reais de falha → adequado a **modelagem de normalidade**.
- Uso: treinar o Autoencoder; a validação de anomalia usa **falhas sintéticas**
  injetadas (ground truth, E2).
- Ref.: Stender, Wallscheid & Böcker (2020).

## PV Farms — domínio **CC** (eixo complementar)
- Arquivos: `dados/brutos/train_data.csv`, `test_data.csv` (separador `;`).
- Dados **rotulados**: Normal, F1 (string), F2 (string-terra), F3 (string-string).
- Predominância de **falhas CC**; uso como **benchmark supervisionado**.
- Ref.: Ghoneim, Rashed & Elkalashy (2021).

## Regra de separação de domínio
Os dois datasets **não se fundem**. O classificador PV Farms (CC) **não**
diagnostica falhas CA do inversor, e suas métricas **não** se transferem ao
pipeline CA. O uso combinado é conceitual/arquitetural — ver `consultar_datasets`
e `comparar_abordagens_ml` no chat.
