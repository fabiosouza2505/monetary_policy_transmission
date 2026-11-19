# -*- coding: utf-8 -*-
"""
Serie trimestral de despesa de pessoal, a partir da Demonstracao do Valor Adicionado
dos arquivos estruturados da CVM.

A DVA e obrigatoria para companhias abertas desde a Lei 11.638/2007 e traz a conta fixa
7.08.01 (Pessoal), desdobrada em remuneracao direta, beneficios e FGTS. A DRE brasileira
e apresentada por funcao e nao abre pessoal; a DVA e a unica fonte estruturada.

Tres cuidados que o dado exige:

1. ACUMULACAO. Os valores sao acumulados no exercicio social: Q1 traz 3 meses, Q2 traz 6,
   Q3 traz 9 e o anual traz 12. O fluxo trimestral e a diferenca entre janelas consecutivas.
   A cadeia precisa ser montada juntando ITR e DFP, porque o valor de 9 meses esta no ITR
   do terceiro trimestre e o de 12 meses so existe no DFP.

2. ESCALA. A coluna ESCALA_MOEDA alterna entre MIL e UNIDADE dentro do mesmo arquivo.
   Ignorar isso erra por mil vezes em cerca de um decimo das linhas.

3. EXERCICIO IRREGULAR. Parte das companhias tem exercicio social deslocado, e a janela
   nao cai em 3/6/9/12. O fluxo so e emitido como trimestral quando a diferenca entre
   janelas consecutivas fica entre 2 e 4 meses; caso contrario a observacao e marcada.

A validacao decisiva e o fechamento: os quatro fluxos trimestrais de um exercicio
devem somar o valor anual de 12 meses declarado no DFP. O script testa isso.

Saida: data/processed/pessoal_trimestral.csv. A coluna cnpj e texto com zeros a
esquerda; ao reler, use dtype={"cnpj": str}, sob pena de perder o zero inicial.

Uso:
    python3 pessoal_dva.py [dir_CVM] [painel.csv]
"""
import sys, glob, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT/"data"/"processed"; TAB = ROOT/"output"/"tables"
for p in (OUT, TAB): p.mkdir(parents=True, exist_ok=True)

CVM = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT.parent/"Projeto Markup"/"GMM - ACF"/"markup-estimation"/"data"/"raw"/"CVM")
PAN = Path(sys.argv[2]) if len(sys.argv) > 2 else (
    ROOT.parent/"Projeto Markup"/"GMM - ACF"/"markup-estimation"/"data"/"processed"/
    "quarterly-v1"/"final_data.csv")

CONTA   = "7.08.01"          # Pessoal (conta fixa da DVA)
BASE    = "ind"              # individual, que casa com a extracao "Nao Consolidado"
ESCALA  = {"MIL": 1_000.0, "UNIDADE": 1.0, "MILHAO": 1_000_000.0}
COLS    = ["CNPJ_CIA","DT_REFER","VERSAO","DENOM_CIA","ESCALA_MOEDA","MOEDA",
           "ORDEM_EXERC","DT_INI_EXERC","DT_FIM_EXERC","CD_CONTA","VL_CONTA"]

digitos = lambda s: s.astype(str).str.replace(r"\D","",regex=True).str.zfill(14)

# ---------------------------------------------------------------- leitura
def carrega():
    arq = sorted(glob.glob(str(CVM/("*_cia_aberta_DVA_%s_*.csv"%BASE))))
    if not arq: sys.exit("nenhum arquivo DVA_%s em %s" % (BASE, CVM))
    out = []
    for f in arq:
        d = pd.read_csv(f, sep=";", encoding="latin-1", low_memory=False, usecols=COLS)
        d = d[(d.ORDEM_EXERC == "ÚLTIMO") & (d.CD_CONTA == CONTA)]
        d["doc"] = "dfp" if Path(f).name.startswith("dfp") else "itr"
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    d["cnpj"] = digitos(d.CNPJ_CIA)
    for c in ("DT_REFER","DT_INI_EXERC","DT_FIM_EXERC"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    d = d[d.MOEDA.eq("REAL") & d.DT_FIM_EXERC.notna()]
    d["fator"] = d.ESCALA_MOEDA.map(ESCALA)
    nao_mapeada = d.fator.isna().sum()
    d["acum"] = pd.to_numeric(d.VL_CONTA, errors="coerce") * d.fator
    # a mesma janela pode reaparecer em versoes diferentes do documento
    d = (d.sort_values(["cnpj","DT_INI_EXERC","DT_FIM_EXERC","VERSAO"])
           .drop_duplicates(["cnpj","DT_INI_EXERC","DT_FIM_EXERC"], keep="last"))
    return d, len(arq), nao_mapeada

# ---------------------------------------------------------------- desacumulacao
def desacumula(d):
    d = d.sort_values(["cnpj","DT_INI_EXERC","DT_FIM_EXERC"]).copy()
    d["meses"] = ((d.DT_FIM_EXERC - d.DT_INI_EXERC).dt.days/30.44).round()
    g = d.groupby(["cnpj","DT_INI_EXERC"], sort=False)
    d["acum_ant"]  = g.acum.shift()
    d["meses_ant"] = g.meses.shift().fillna(0)
    d["pessoal"]   = d.acum - d.acum_ant.fillna(0)
    d["janela"]    = d.meses - d.meses_ant
    d["ano"]       = d.DT_FIM_EXERC.dt.year
    d["trimestre"] = d.DT_FIM_EXERC.dt.quarter
    # um elo faltante na cadeia produz fluxo de 6 ou 9 meses rotulado como trimestre
    d["ok"] = d.janela.between(2, 4)
    return d

# ---------------------------------------------------------------- execucao
d, n_arq, sem_escala = carrega()
d = desacumula(d)

pan = pd.read_csv(PAN, sep=";", encoding="utf-8", low_memory=False)
pan["cnpj"] = digitos(pan.cnpj)
mapa = (pan[pan.nome_setor.notna()][["empresa","cnpj","nome_setor"]]
          .drop_duplicates("cnpj"))

s = d.merge(mapa, on="cnpj", how="inner")
s = s[s.ok & s.pessoal.notna()]
s = s[s.pessoal > 0]                     # zero e negativo nao sao despesa de pessoal
s["ln_pessoal"] = np.log(s.pessoal)

col = ["empresa","cnpj","nome_setor","ano","trimestre","pessoal","ln_pessoal",
       "janela","doc","DT_REFER","DT_INI_EXERC","DT_FIM_EXERC","DENOM_CIA"]
s = s[col].sort_values(["empresa","ano","trimestre"])
s.to_csv(OUT/"pessoal_trimestral.csv", index=False)

# ---------------------------------------------------------------- diagnostico
res=[]
def add(c,v,cr,ok,o=""): res.append({"checagem":c,"valor":v,"criterio":cr,
                                     "veredito":"OK" if ok else "ATENCAO","obs":o})
add("arquivos lidos", "%d" % n_arq, "ITR e DFP, base %s" % BASE, n_arq>20)
add("escala nao mapeada", "%d linhas" % sem_escala, "zero", sem_escala==0,
    "ESCALA_MOEDA fora de %s" % list(ESCALA))
esc = d.ESCALA_MOEDA.value_counts().to_dict()
add("escalas encontradas", str(esc), "normalizadas para reais", True)
mau = int((~d.ok).sum())
add("elos de cadeia quebrados", "%d de %d" % (mau, len(d)), "menos de 5%",
    mau < .05*len(d), "janela fora de 2 a 4 meses; observacao descartada")
add("fluxos nao positivos", "%d" % int((d.ok & (d.pessoal<=0)).sum()), "informativo", True,
    "zeros e negativos descartados")
add("serie final", "%d obs, %d empresas, %d-%d" %
    (len(s), s.empresa.nunique(), s.ano.min(), s.ano.max()), "painel casado por CNPJ", len(s)>0)
q = s.trimestre.value_counts().reindex([1,2,3,4], fill_value=0)
add("equilibrio entre trimestres", "Q1 %d | Q2 %d | Q3 %d | Q4 %d" % tuple(q),
    "sem trimestre ausente", (q>0).all())
# fechamento: os quatro trimestres de um exercicio devem somar o anual do DFP
anual = (d[(d.meses == 12) & (d.doc == "dfp")]
           .assign(ano=lambda x: x.DT_FIM_EXERC.dt.year)
           .groupby(["cnpj","ano"]).acum.first().rename("anual").reset_index())
soma = (s.groupby(["cnpj","ano"]).pessoal.agg(["sum","size"])
          .rename(columns={"sum":"soma","size":"n"}).reset_index())
fech = soma[soma.n == 4].merge(anual, on=["cnpj","ano"], how="inner")
fech = fech[fech.anual > 0]
if len(fech):
    err = ((fech.soma - fech.anual)/fech.anual).abs()
    add("fechamento anual", "%.1f%% dentro de 0,1%%" % (100*(err < .001).mean()),
        "acima de 90%", (err < .001).mean() > .90,
        "%d exercicios completos; erro mediano %.1e; %d acima de 1%%"
        % (len(fech), err.median(), int((err > .01).sum())))
    fech.assign(erro=err).sort_values("erro", ascending=False).head(50).to_csv(
        TAB/"pessoal_fechamento_divergente.csv", index=False)
else:
    add("fechamento anual", "sem exercicios completos", "acima de 90%", False)

# sanidade: pessoal deve ser fracao plausivel e estavel da receita
sz = (s.groupby(["ano","trimestre"]).pessoal.sum())
cv = sz.pct_change().abs().median()
add("estabilidade agregada", "variacao mediana %.1f%%" % (100*cv), "abaixo de 25%", cv<.25,
    "soma trimestral de pessoal no painel")

t = pd.DataFrame(res); t.to_csv(TAB/"pessoal_checagens.csv", index=False)
pd.set_option("display.width",220); pd.set_option("display.max_colwidth",46)
print(t.to_string(index=False)); print()
print(s.groupby("ano").agg(obs=("pessoal","size"), empresas=("empresa","nunique")).to_string())
print("\nserie: data/processed/pessoal_trimestral.csv")
