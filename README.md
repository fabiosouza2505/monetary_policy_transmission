# Poder de mercado e transmissão da política monetária no Brasil

Este trabalho replica a arquitetura empírica de Ottonello e Winberry (2020, *Econometrica*) na economia brasileira, trocando a variável de heterogeneidade: além da posição financeira, entra o **markup da firma**, estimado pela abordagem de razão de custo de De Loecker e Warzynski (2012). A pergunta é se o poder de mercado condiciona a transmissão da política monetária mesmo depois de controlar pelo canal de crédito.

Autor: Fabio de Medeiros Souza (Departamento de Economia, UnB)

## Estrutura

```
src/               scripts do pipeline, numerados na ordem de execução
data/raw/          insumos públicos de mercado (B3, BCB) — ver data/raw/README.md
data/interim/      calendário de eventos do Copom
data/processed/    surpresas monetárias, agregação trimestral, despesa de pessoal
output/figures/    figuras dos protocolos de validação e das projeções locais
output/tables/     checagens de validação e tabelas de resultados
```

## Pipeline

| Etapa | Script | Faz |
|---|---|---|
| Bloco B | `01_build_surprises.py` | surpresa de nível e fatores de trajetória a partir da taxa a termo entre reuniões do Copom (curva DI × PRE) |
| Bloco B | `02_validacao_bloco_b.py` | checagens de validação das surpresas |
| — | `03_agrega_trimestral.py` | agregação dos choques ao trimestre (ponderação da eq. 2 de Ottonello e Winberry) |
| Bloco A | `validacao_bloco_a.py` | validação da estimação de markups |
| — | `pessoal_dva.py` | despesa de pessoal trimestral a partir da DVA (dados abertos da CVM) |
| — | `05a_estoques_demonstrativos.py` | variáveis de estoque dos demonstrativos financeiros |
| — | `05_painel_regressao.py` | painel firma × trimestre para a seção 6 |
| Bloco C | `05b_painel_replicacao.py` | painel da replicação de Ottonello e Winberry |
| — | `06_tabela1.py` | projeções locais (markup, alavancagem e as duas juntas) |
| Bloco C | `07_tabela_ow.py` | replicação no formato da Tabela III do artigo de referência |
| — | `08_fig_horse_race.py` | figura das projeções locais conjuntas |

## Reprodutibilidade e dados

- **Surpresas monetárias (Bloco B):** dá para reproduzir integralmente a partir deste repositório. Todos os insumos estão em `data/raw/`.
- **Markups:** são estimados no repositório irmão [`markup-estimation`](https://github.com/fabiosouza2505/markup-estimation). Os scripts que dependem dele esperam encontrá-lo em `../Projeto Markup/GMM - ACF/markup-estimation/`, ou recebem o caminho como argumento.
- **Demonstrativos de firma:** vêm da **Economatica**, cuja licença não permite redistribuição. Os arquivos derivados (`data/interim/estoques*`, `data/processed/painel_regressao.csv`, `data/processed/painel_replicacao.csv`) não estão versionados. Para reproduzir as etapas 05 a 08 é preciso ter acesso próprio à base.

Dependências: Python 3 com `pandas`, `numpy`, `statsmodels`, `linearmodels`, `matplotlib`.

## Licença

Código sob licença MIT (ver `LICENSE`).
