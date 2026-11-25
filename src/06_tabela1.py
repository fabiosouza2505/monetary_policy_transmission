# -*- coding: utf-8 -*-
"""
Tabela 1 da proposta (Secao 6) -- primeira rodada de estimacoes, versao inicial.

  6.1  Replicacao de Ottonello-Winberry (h = 1): dlnk_ow sobre choque x alavancagem
  6.2  Markup isolado, projecoes locais h = 0..8
  6.3  Alavancagem isolada, projecoes locais
  6.4  Modelo conjunto (horse race): as duas interacoes
  6.5  Nivel vs trajetoria: as interacoes com os dois fatores do choque

Especificacao, seguindo a Secao 6:
  y(t+h) - y(t-1) = b_h (s_t x mu_q_{t-1}) + g_h (s_t x lev_z_{t-1}) + d' X_{t-1}
                    + a_i + l_{setor,t} + e
  X = log do ativo, liquidez corrente, produtividade omega, e as variaveis de
  heterogeneidade em nivel; mais as defasagens do termo de interacao
  s_{t-1} x mu_q_{t-2} e s_{t-1} x lev_z_{t-2} (V9, Secao 5.3).
  Efeitos fixos de firma e de setor x trimestre. O efeito medio do choque e absorvido
  pelo efeito fixo de setor x trimestre; o que se identifica e a resposta diferencial.

Convencoes:
  s em desvios-padrao, positivo = contracionista.
  Respostas em pontos percentuais (log x 100).
  mu_q vai de 0 a 1: o coeficiente e a diferenca de resposta entre o topo e a base da
  distribuicao intrassetorial; do p25 ao p75 e metade disso.
  lev_z em desvios-padrao dentro da celula.
  Predicao de sinal, para os dois canais: POSITIVO (firmas de markup alto e firmas
  alavancadas respondem menos a um aperto).

Inferencia: calculam-se tres erros-padrao -- agrupado por firma e trimestre (CRV1
bidimensional), so por firma, so por trimestre -- e reporta-se o MAIOR. O bidimensional
subtrai a intersecao e, com 57 a 127 firmas e 56 a 87 trimestres, pode sair menor que
ambos os unidimensionais, o que fabrica significancia. Valor-p pela t de Student com
min(firmas, trimestres) - 1 graus de liberdade. O bootstrap em bloco com reestimacao da
primeira etapa fica para a versao seguinte.

Uso:
    python3 06_tabela1.py [painel_regressao.csv] [dir_saida]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import pyfixest as pf
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PAINEL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"data"/"processed"/"painel_regressao.csv"
SAIDA  = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT/"output"
TAB, FIG = SAIDA/"tables", SAIDA/"figures"
for p in (TAB, FIG): p.mkdir(parents=True, exist_ok=True)

H = 8
RESP = {"lnk": "Investimento ($\\Delta\\ln K$)", "lnr": "Receita ($\\ln R$)", "lnl": "Pessoal ($\\ln L$)"}
VCOV = {"CRV1": "empresa+t"}
FE = "empresa + cel"

d = pd.read_csv(PAINEL)
d = d.dropna(subset=["mu_q_l1", "lnat_l1", "liq_l1", "omega_l1", "s_nivel"])
d["empresa"] = d.empresa.astype(str); d["cel"] = d.cel.astype(str)
for c in [c for c in d.columns if c.startswith("d") and c.split("_")[-1] in RESP] + ["dlnk_ow"]:
    d[c] = 100*d[c]

# termos de interacao explicitos
d["i_mu"]      = d.s_nivel*d.mu_q_l1
d["i_lev"]     = d.s_nivel*d.lev_z_l1
d["i_mu_tr"]   = d.s_traj*d.mu_q_l1
d["i_lev_tr"]  = d.s_traj*d.lev_z_l1
d["il_mu"]     = d.s_nivel_l1*d.mu_q_l2
d["il_lev"]    = d.s_nivel_l1*d.lev_z_l2
# alavancagem na construcao de Ottonello-Winberry (desvio da media da firma)
d["i_levdm"]   = d.s_nivel*d.lev_dm_l1
d["il_levdm"]  = d.s_nivel_l1*d.lev_dm_l2
CTRL = "lnat_l1 + liq_l1 + omega_l1"
CTRL_OW = "lev_dm_l1 + dlnr_l1 + lnat_l1 + ac_at_l1"    # controles Z do artigo

def roda(f, dados, termos):
    """estima e devolve linhas tidy para os termos de interesse"""
    import re
    usadas = [v for v in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", f)) if v in dados.columns]
    dados = dados.dropna(subset=usadas)
    m = pf.feols(f, data=dados, vcov=VCOV)
    tb2 = m.tidy().copy()
    tbf = m.vcov({"CRV1": "empresa"}).tidy().copy()
    tbt = m.vcov({"CRV1": "t"}).tidy().copy()
    G = min(dados.empresa.nunique(), dados.t.nunique())
    out = []
    for k in termos:
        if k not in tb2.index: continue
        c = tb2.loc[k, "Estimate"]
        e2, ef, et = tb2.loc[k, "Std. Error"], tbf.loc[k, "Std. Error"], tbt.loc[k, "Std. Error"]
        # regra conservadora: o CRV1 bidimensional subtrai a intersecao e, com poucos
        # clusters, pode produzir variancia quase nula; reporta-se o maior dos tres
        ep = np.nanmax([e2, ef, et])
        p = 2*stats.t.sf(abs(c)/ep, df=G-1)
        out.append({"termo": k, "coef": c, "ep": ep, "p": p,
                    "ep_2vias": e2, "ep_firma": ef, "ep_trim": et, "n": m._N,
                    "empresas": dados.empresa.nunique(), "trimestres": dados.t.nunique()})
    return out

linhas = []
def guarda(tabela, y, h, amostra, spec, res):
    for r in res:
        r.update({"tabela": tabela, "resposta": y, "h": h, "amostra": amostra, "spec": spec})
        linhas.append(r)

AMOSTRAS = {"completa": d, "sem Maquinas e Agro": d[d.sem_problema == True],
            "2011 em diante": d[d.ano >= 2011]}

# ------------------------------------------------------------------ 6.1
for nome, a in AMOSTRAS.items():
    f = "dlnk_ow ~ i_lev + lev_z_l1 + lnat_l1 + liq_l1 | " + FE
    guarda("6.1", "lnk", 1, nome, "O-W", roda(f, a, ["i_lev"]))
f = "dlnk_ow ~ i_lev + lev_z_l1 + lnat_l1 + liq_l1 + il_lev | " + FE
guarda("6.1", "lnk", 1, "completa", "O-W + defasagem", roda(f, d, ["i_lev"]))
# versao fiel ao artigo: alavancagem desviada da media da firma, controles Z de O-W
for nome, a in AMOSTRAS.items():
    f = "dlnk_ow ~ i_levdm + %s | %s" % (CTRL_OW, FE)
    guarda("6.1", "lnk", 1, nome, "O-W fiel", roda(f, a, ["i_levdm"]))

# ------------------------------------------------------------------ 6.2 a 6.4
SPECS = {
    "6.2": ("i_mu + mu_q_l1 + il_mu", ["i_mu"]),
    "6.3": ("i_lev + lev_z_l1 + il_lev", ["i_lev"]),
    "6.4": ("i_mu + i_lev + mu_q_l1 + lev_z_l1 + il_mu + il_lev", ["i_mu", "i_lev"]),
}
for tab, (rhs, termos) in SPECS.items():
    for y in RESP:
        for h in range(H+1):
            f = "d%d_%s ~ %s + %s | %s" % (h, y, rhs, CTRL, FE)
            guarda(tab, y, h, "completa", "base", roda(f, d, termos))
            if tab == "6.4":
                guarda(tab, y, h, "sem Maquinas e Agro", "base",
                       roda(f, AMOSTRAS["sem Maquinas e Agro"], termos))
                # variante: alavancagem na construcao de O-W
                f2 = ("d%d_%s ~ i_mu + i_levdm + mu_q_l1 + lev_dm_l1 + il_mu + il_levdm + %s | %s"
                      % (h, y, CTRL, FE))
                for nome in ("completa", "sem Maquinas e Agro"):
                    guarda(tab, y, h, nome, "lev O-W", roda(f2, AMOSTRAS[nome], ["i_mu", "i_levdm"]))

# ------------------------------------------------------------------ 6.5
rhs = "i_mu + i_lev + i_mu_tr + i_lev_tr + mu_q_l1 + lev_z_l1 + il_mu + il_lev"
for y in RESP:
    for h in (1, 4, 8):
        f = "d%d_%s ~ %s + %s | %s" % (h, y, rhs, CTRL, FE)
        guarda("6.5", y, h, "completa", "base",
               roda(f, d, ["i_mu", "i_lev", "i_mu_tr", "i_lev_tr"]))

R = pd.DataFrame(linhas)
R.to_csv(TAB/"tabela1_resultados.csv", index=False)

# ------------------------------------------------------------------ tabelas legiveis
def est(c, p):
    return "%.3f%s" % (c, "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else "")

def bloco(tab, termos, amostra="completa", hs=(0, 1, 2, 4, 8), spec="base"):
    x = R[(R.tabela == tab) & (R.amostra == amostra) & (R.spec == spec)]
    rows = []
    for y in RESP:
        for k in termos:
            z = x[(x.resposta == y) & (x.termo == k)].set_index("h")
            rows.append([RESP[y], k] + [est(z.loc[h, "coef"], z.loc[h, "p"]) + " (%.3f)" % z.loc[h, "ep"]
                                        if h in z.index else "" for h in hs]
                        + [int(z.loc[hs[1], "n"]) if hs[1] in z.index else ""])
    return pd.DataFrame(rows, columns=["resposta", "termo"] + ["h=%d" % h for h in hs] + ["N (h=1)"])

with open(TAB/"tabela1_resumo.txt", "w", encoding="utf-8") as fh:
    t61 = R[R.tabela == "6.1"][["amostra", "spec", "coef", "ep", "p", "n", "empresas", "trimestres"]]
    fh.write("TABELA 6.1 -- Replicacao O-W, dlnk(t+1), interacao choque x alavancagem\n")
    fh.write(t61.round(4).to_string(index=False) + "\n\n")
    for tab, termos, nome in [("6.2", ["i_mu"], "Markup isolado"),
                              ("6.3", ["i_lev"], "Alavancagem isolada"),
                              ("6.4", ["i_mu", "i_lev"], "Horse race"),
                              ("6.4", ["i_mu", "i_lev"], "Horse race, sem Maquinas e Agro")]:
        am = "sem Maquinas e Agro" if "sem" in nome else "completa"
        fh.write("TABELA %s -- %s\n" % (tab, nome))
        fh.write(bloco(tab, termos, am).to_string(index=False) + "\n\n")
    for am in ("completa", "sem Maquinas e Agro"):
        fh.write("TABELA 6.4 -- Horse race, alavancagem na construcao de O-W, amostra %s\n" % am)
        fh.write(bloco("6.4", ["i_mu", "i_levdm"], am, spec="lev O-W").to_string(index=False) + "\n\n")
    x = R[R.tabela == "6.5"].pivot_table(index=["resposta", "termo"], columns="h", values="coef")
    fh.write("TABELA 6.5 -- Nivel vs trajetoria (coeficientes)\n" + x.round(3).to_string() + "\n")

# ------------------------------------------------------------------ figuras de resposta
def fig(tab, termo, rotulo, arq, amostra="completa", spec="base"):
    # eixo y compartilhado: o mesmo coeficiente e comparado entre as tres respostas
    f, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True, sharey=True)
    for a, y in zip(ax, RESP):
        z = R[(R.tabela == tab) & (R.termo == termo) & (R.resposta == y)
              & (R.amostra == amostra) & (R.spec == spec)].sort_values("h")
        if not len(z): continue
        a.axhline(0, lw=.8, color="#333333")
        a.fill_between(z.h, z.coef - 1.96*z.ep, z.coef + 1.96*z.ep, alpha=.15, color="#2b6cb0",
                       lw=0, label="IC 95%")
        a.fill_between(z.h, z.coef - 1.645*z.ep, z.coef + 1.645*z.ep, alpha=.30, color="#2b6cb0",
                       lw=0, label="IC 90%")
        a.plot(z.h, z.coef, marker="o", ms=4, color="#2b6cb0", label="estimativa")
        a.set_title(RESP[y].replace("$", "").replace("\\Delta", "Δ").replace("\\ln", "ln"))
        a.set_xlabel("trimestres após o choque (h)")
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("p.p. por 1 dp de choque contracionista")
    ax[-1].legend(frameon=False, fontsize=8, loc="upper right")
    f.suptitle(rotulo, y=1.03)
    f.tight_layout(); f.savefig(FIG/arq, dpi=150, bbox_inches="tight"); plt.close(f)

fig("6.2", "i_mu", "Tabela 6.2 — choque × quantil de markup (topo vs base)", "fig_lp_62_markup.png")
fig("6.3", "i_lev", "Tabela 6.3 — choque × alavancagem (por 1 dp)", "fig_lp_63_alavancagem.png")
fig("6.4", "i_mu", "Tabela 6.4 — horse race: termo de markup", "fig_lp_64_markup.png")
fig("6.4", "i_lev", "Tabela 6.4 — horse race: termo de alavancagem", "fig_lp_64_alavancagem.png")
fig("6.4", "i_levdm", "Tabela 6.4 — horse race, alavancagem na construção de O-W", "fig_lp_64_alavancagem_ow.png",
    spec="lev O-W")

print(open(TAB/"tabela1_resumo.txt", encoding="utf-8").read())
print("regressoes estimadas:", R.groupby(["tabela", "resposta", "h", "amostra", "spec"]).ngroups)
