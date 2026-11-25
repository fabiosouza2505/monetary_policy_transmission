# -*- coding: utf-8 -*-
"""
Painel da replicacao de Ottonello-Winberry (Bloco C) -- sem exigencia de markup.

A replicacao com alavancagem nao depende da estimacao de markups, de modo que nao ha
razao para restringi-la as empresas do painel de markups. Este painel usa toda empresa
com demonstrativos e setor conhecido, para medir quanto poder se ganha com a amostra
maior.

Setor: o do pipeline de markups quando existe; senao, o Setor_B3 do indice da B3.

Uso:
    python3 05b_painel_replicacao.py
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ME = ROOT.parent/"Projeto Markup"/"GMM - ACF"/"markup-estimation"/"data"
EST = ROOT/"data"/"interim"/"estoques_demonstrativos.csv"
SUR = ROOT/"data"/"processed"/"surpresas_trimestrais.csv"
OUT = ROOT/"data"/"processed"/"painel_replicacao.csv"
tq = lambda a, q: a*4 + q - 1
def winsor(x, p=.01):
    lo, hi = x.quantile([p, 1-p]); return x.clip(lo, hi)

f = pd.read_csv(ME/"processed"/"quarterly-v1"/"final_data.csv", sep=";", low_memory=False)
pip = f[f.nome_setor.notna()].groupby("empresa").nome_setor.agg(lambda s: s.mode().iat[0])
idx = pd.read_csv(ME/"raw"/"index"/"empresas_b3_enriquecido_nao_financeiras.csv",
                  sep=";", encoding="utf-8-sig", dtype=str)
b3 = idx.dropna(subset=["Setor_B3"]).drop_duplicates("Ticker_Raiz").set_index("Ticker_Raiz").Setor_B3

e = pd.read_csv(EST); e["t"] = tq(e.ano, e.trimestre)
e = e.sort_values(["empresa","t"]).drop_duplicates(["empresa","t"], keep="last")
e = e[~((e.ano == 2024) & (e.trimestre == 4))]
e["setor"] = e.empresa.map(pip).fillna(e.empresa.map(b3).map(lambda s: "B3: " + s if isinstance(s, str) else s))
e["fonte_setor"] = np.where(e.empresa.isin(pip.index), "pipeline", np.where(e.setor.notna(), "B3", ""))
e = e[e.setor.notna()].copy()
e["energia"] = e.setor.isin(["Energia Elétrica", "B3: Utilidade Pública"])
e["painel_markup"] = e.empresa.isin(pip.index) & ~e.energia

div = e[["fin_cp","fin_lp","deb_cp","deb_lp"]]
e["lev"] = winsor(pd.Series(np.where(div.notna().any(axis=1) & (e.ativo > 0),
                                     div.fillna(0).sum(axis=1)/e.ativo, np.nan), index=e.index))
dm = e.lev - e.groupby("empresa").lev.transform("mean"); e["lev_dm"] = dm/dm.std()
e["lnk"] = np.log(e.imob.where(e.imob > 0)); e["lnat"] = np.log(e.ativo.where(e.ativo > 0))
e["ac_at"] = (e.ac/e.ativo).where(e.ativo > 0)

def em(v, k):
    x = e[["empresa","t",v]].copy(); x["t"] = x.t - k
    return e[["empresa","t"]].merge(x, on=["empresa","t"], how="left")[v].values
for v in ["lev_dm","lnat","ac_at"]: e[v+"_l1"] = em(v, -1)
e["dlnk_ow"] = winsor(pd.Series(em("lnk", 1) - e.lnk.values)).values

s = pd.read_csv(SUR); s = s[s.completo == True].copy()
s["t"] = tq(s.trimestre.str[:4].astype(int), s.trimestre.str[-1].astype(int))
e["s_nivel_bp"] = e.t.map(s.set_index("t").fator_nivel)
e = e[e.s_nivel_bp.notna()]
e["ano"] = e.t//4; e["cel"] = e.setor + "_" + e.t.astype(str)
cols = ["empresa","t","ano","setor","fonte_setor","energia","painel_markup","cel","lev","lev_dm_l1",
        "lnat_l1","ac_at_l1","dlnk_ow","s_nivel_bp"]
e[cols].to_csv(OUT, index=False)
u = e.dropna(subset=["dlnk_ow","lev_dm_l1"])
print("painel de replicacao: %d obs, %d empresas, %d setores" % (len(u), u.empresa.nunique(), u.setor.nunique()))
print("  sem energia: %d obs, %d empresas" % ((~u.energia).sum(), u[~u.energia].empresa.nunique()))
print("  fonte do setor:", u.groupby("fonte_setor").empresa.nunique().to_dict())
print("saida:", OUT)
