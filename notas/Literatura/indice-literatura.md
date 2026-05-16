# 📚 Índice da Literatura — Mestrado UTFPR

## 📊 Todos os Artigos

```dataview
TABLE autor, ano, tema, subtema, relevancia, lido
FROM "Literatura"
SORT ano DESC
```


## 🔴 Não Lidos

```dataview
TABLE autor, ano, tema
FROM "Literatura"
WHERE lido = false
```

## ✅ Lidos

```dataview
TABLE autor, ano, subtema
FROM "Literatura"
WHERE relevancia = "Alta"
```

## ⭐ Alta Relev
```dataview
TABLE autor, ano, subt
SORT ano DESC
```
