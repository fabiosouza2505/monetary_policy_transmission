# data/raw — insumos de mercado

Todos os arquivos vêm de fontes públicas. Eles estão versionados porque os endereços antigos de download da B3 foram desativados. Assim, guardar a matéria-prima é o que garante a reprodução das surpresas monetárias.

| Arquivo | Conteúdo | Fonte |
|---|---|---|
| `pre_terms.csv` | taxas spot da curva DI × PRE (% a.a., base 252) nos prazos até as vigências v0–v4, para t−1, t0 e t+1 de cada reunião; interpolação flat-forward | B3, taxas referenciais DI × PRE |
| `pre_terms_part1.csv`, `pre_terms_part2.csv` | as mesmas taxas em duas partes, no formato da coleta original (data `AAMMDD`) | B3 |
| `copom_decisoes.csv` | reuniões do Copom desde a 80ª (jan/2003): data, início de vigência, meta Selic, viés, reuniões extraordinárias e decisões monocráticas | BCB, histórico das taxas de juros |
| `ativos_janela.csv` | retorno do Ibovespa e do USD/BRL na janela de cada reunião (%) | B3 e BCB (PTAX) |

Amostra: janeiro de 2003 em diante, 198 reuniões.
