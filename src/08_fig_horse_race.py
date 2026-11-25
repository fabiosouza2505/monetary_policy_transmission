# -*- coding: utf-8 -*-
"""Figura do horse race (Tabela 6.4, alavancagem na construcao de O-W): 2 x 3."""
import sys; from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"output"/"tables"/"tabela1_resultados.csv"
PAIN = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT/"data"/"processed"/"painel_regressao.csv"
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT/"output"/"figures"/"fig_lp_64_horse_race_ow.png"
R = pd.read_csv(RES, dtype={"tabela": str})
P = pd.read_csv(PAIN).dropna(subset=["mu_q_l1","lnat_l1","liq_l1","omega_l1","s_nivel"])
sdmu = P.mu_q_l1.std()
x = R[(R.tabela == "6.4") & (R.spec == "lev O-W") & (R.amostra == "completa")]
RESP = [("lnk","Investimento (Δ ln K)"), ("lnr","Receita (ln R)"), ("lnl","Pessoal (ln L)")]
TERM = [("i_mu","markup × choque", sdmu), ("i_levdm","alavancagem × choque", 1.0)]
f, ax = plt.subplots(2, 3, figsize=(13, 6.6), sharex=True, sharey="row")
for i, (k, rot, esc) in enumerate(TERM):
    for j, (y, ry) in enumerate(RESP):
        a = ax[i, j]; z = x[(x.resposta == y) & (x.termo == k)].sort_values("h")
        c, e = z.coef*esc, z.ep*esc
        a.axhline(0, lw=.8, color="#333333")
        a.fill_between(z.h, c-1.96*e, c+1.96*e, alpha=.15, color="#2b6cb0", lw=0, label="IC 95%")
        a.fill_between(z.h, c-1.645*e, c+1.645*e, alpha=.30, color="#2b6cb0", lw=0, label="IC 90%")
        a.plot(z.h, c, marker="o", ms=4, color="#2b6cb0", label="estimativa")
        a.spines[["top","right"]].set_visible(False)
        if i == 0: a.set_title(ry)
        if i == 1: a.set_xlabel("trimestres após o choque (h)")
        if j == 0: a.set_ylabel("%s\np.p. por 1 dp × 1 dp" % rot)
ax[0, 2].legend(frameon=False, fontsize=8, loc="upper right")
f.tight_layout(); f.savefig(OUT, dpi=150, bbox_inches="tight"); print(OUT)
