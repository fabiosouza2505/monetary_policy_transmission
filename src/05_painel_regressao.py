# -*- coding: utf-8 -*-
"""
Painel de regressao da Secao 6 -- firma x trimestre.

Junta quatro fontes:
  markups      results/python-gmm-acf/baseline/markups_empresa.csv  (medida corrigida)
  estoques     data/interim/estoques_demonstrativos.csv             (05a)
  pessoal      data/processed/pessoal_trimestral.csv                (pessoal_dva.py)
  choques      data/processed/surpresas_trimestrais.csv             (03_agrega_trimestral.py)

Variaveis de heterogeneidade, medidas em t-1:
  mu_q    quantil do markup dentro da celula setor x trimestre (0 a 1)
  lev_z   alavancagem (divida CP + LP sobre ativo total), padronizada na celula

Respostas, y(t+h) - y(t-1), h = 0..8:
  lnk     log do imobilizado -- a medida de investimento de Ottonello-Winberry
  lnr     log da receita liquida
  lnl     log da despesa de pessoal (DVA; de 2011 em diante)
E a resposta de um periodo do artigo original: dlnk_ow = lnk(t+1) - lnk(t).

Choques em t, em desvios-padrao, positivo = contracionista. Sob essa convencao,
Ottonello-Winberry e Kimball preveem o MESMO sinal para a interacao: positivo. Firmas
mais alavancadas (O-W) e de markup mais alto (Kimball) respondem menos a um aperto.

Regras herdadas da V10: energia excluida; 2024Q4 descartado (extracao incompleta);
quantil so e calculado em celulas com ao menos 12 firmas.

Uso:
    python3 05_painel_regressao.py [markups_empresa.csv]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MK = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT.parent/"Projeto Markup"/"GMM - ACF"/"markup-estimation"/"results"/
    "python-gmm-acf"/"baseline"/"markups_empresa.csv")
EST = ROOT/"data"/"interim"/"estoques_demonstrativos.csv"
PES = ROOT/"data"/"processed"/"pessoal_trimestral.csv"
SUR = ROOT/"data"/"processed"/"surpresas_trimestrais.csv"
OUT = ROOT/"data"/"processed"/"painel_regressao.csv"

MIN_FIRMAS, H = 12, 8
tq = lambda a, q: a*4 + q - 1                          # indice inteiro de trimestre

def winsor(x, p=.01):
    lo, hi = x.quantile([p, 1-p]); return x.clip(lo, hi)

# ------------------------------------------------------------------ markups
mk = pd.read_csv(MK, usecols=["empresa","ano","trimestre","nome_setor","markup","omega","receita"])
mk = mk[(mk.nome_setor != "Energia Elétrica")]
mk = mk[~((mk.ano == 2024) & (mk.trimestre == 4))]
mk["t"] = tq(mk.ano, mk.trimestre)
setor = mk.groupby("empresa").nome_setor.agg(lambda s: s.mode().iat[0])

ok = mk[mk.markup.notna()].copy()
ok["n"] = ok.groupby(["nome_setor","t"]).empresa.transform("nunique")
ok = ok[ok.n >= MIN_FIRMAS]
ok["mu_q"] = ok.groupby(["nome_setor","t"]).markup.rank(pct=True)

# ------------------------------------------------------------------ estoques
e = pd.read_csv(EST); e["t"] = tq(e.ano, e.trimestre)
e = e.sort_values(["empresa","t"]).drop_duplicates(["empresa","t"], keep="last")
div = e[["fin_cp","fin_lp","deb_cp","deb_lp"]]
e["lev"] = np.where(div.notna().any(axis=1) & (e.ativo > 0),
                    div.fillna(0).sum(axis=1)/e.ativo, np.nan)
e["lnk"]  = np.log(e.imob.where(e.imob > 0))
e["lnat"] = np.log(e.ativo.where(e.ativo > 0))
e["liq"]  = (e.ac/e.pc).where(e.pc > 0)
e = e[e.empresa.isin(setor.index)].copy()
e["nome_setor"] = e.empresa.map(setor)
e["lev"] = winsor(e.lev); e["liq"] = winsor(e.liq)
cel = e.groupby(["nome_setor","t"]).lev
e["lev_z"] = (e.lev - cel.transform("mean"))/cel.transform("std")
# construcao de Ottonello-Winberry: desvio em relacao a media da propria firma,
# padronizado na amostra inteira -- identifica com a firma mais endividada que o usual
dm = e.lev - e.groupby("empresa").lev.transform("mean")
e["lev_dm"] = dm/dm.std()
e["ac_at"] = (e.ac/e.ativo).where(e.ativo > 0)

# ------------------------------------------------------------------ pessoal, receita
p = pd.read_csv(PES, dtype={"cnpj": str}); p["t"] = tq(p.ano, p.trimestre)
p = p.groupby(["empresa","t"]).ln_pessoal.first().rename("lnl").reset_index()
r = mk[["empresa","t","receita","omega"]].copy()
r["lnr"] = np.log(r.receita.where(r.receita > 0))

# ------------------------------------------------------------------ base firma x t
base = (e[["empresa","t","nome_setor","lev","lev_z","lev_dm","ac_at","lnk","lnat","liq"]]
          .merge(r[["empresa","t","lnr","omega"]], on=["empresa","t"], how="outer")
          .merge(p, on=["empresa","t"], how="left")
          .merge(ok[["empresa","t","mu_q","markup","n"]], on=["empresa","t"], how="left"))
base["nome_setor"] = base.empresa.map(setor)
base = base.dropna(subset=["nome_setor"])

def em(var, k):
    """valor de var em t+k para a mesma firma (k negativo = defasagem)"""
    x = base[["empresa","t",var]].copy(); x["t"] = x.t - k
    return base[["empresa","t"]].merge(x, on=["empresa","t"], how="left")[var].values

for v in ["mu_q","lev_z","lev_dm","lev","lnat","liq","ac_at","omega","markup"]:
    base[v+"_l1"] = em(v, -1)
for v in ["mu_q","lev_z","lev_dm"]:
    base[v+"_l2"] = em(v, -2)
# crescimento das vendas em t-1, controle de Ottonello-Winberry
base["dlnr_l1"] = em("lnr", -1) - em("lnr", -2)
for v in ["lnk","lnr","lnl"]:
    base[v+"_l1"] = em(v, -1)
    for h in range(H+1):
        base["d%d_%s" % (h, v)] = winsor(pd.Series(em(v, h)) - base[v+"_l1"].values).values
base["dlnk_ow"] = winsor(pd.Series(em("lnk", 1)) - base.lnk.values).values

# ------------------------------------------------------------------ choques
s = pd.read_csv(SUR); s = s[s.completo == True].copy()
s["t"] = tq(s.trimestre.str[:4].astype(int), s.trimestre.str[-1].astype(int))
sd_n, sd_p = s.fator_nivel.std(), s.fator_trajetoria.std()
s["s_nivel"] = s.fator_nivel/sd_n; s["s_traj"] = s.fator_trajetoria/sd_p
sm = s.set_index("t")
base["s_nivel"] = base.t.map(sm.s_nivel); base["s_traj"] = base.t.map(sm.s_traj)
base["s_nivel_l1"] = (base.t - 1).map(sm.s_nivel)

base["ano"] = base.t//4; base["trimestre"] = base.t % 4 + 1
base["cel"] = base.nome_setor + "_" + base.t.astype(str)
base["sem_problema"] = ~base.nome_setor.str.startswith(("Máquinas","Agropecuaria"))
base = base[base.s_nivel.notna()]                       # janela do choque: 2003Q2 em diante
base.to_csv(OUT, index=False)

# ------------------------------------------------------------------ resumo
u = base.dropna(subset=["mu_q_l1"])
print("painel: %d linhas firma-trimestre com choque | %d com quantil em t-1" % (len(base), len(u)))
print("periodo: %dQ%d a %dQ%d | empresas com quantil: %d"
      % (u.ano.min(), u[u.ano==u.ano.min()].trimestre.min(),
         u.ano.max(), u[u.ano==u.ano.max()].trimestre.max(), u.empresa.nunique()))
print("dp dos choques (bp): nivel %.2f | trajetoria %.2f -> divididos para 1 dp" % (sd_n, sd_p))
cob = {c: "%.0f%%" % (100*u[c].notna().mean()) for c in
       ["lev_z_l1","lnat_l1","liq_l1","omega_l1","d1_lnk","d1_lnr","d1_lnl","dlnk_ow"]}
print("cobertura (entre as obs com quantil):", cob)
print("saida:", OUT)
