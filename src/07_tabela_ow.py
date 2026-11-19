# -*- coding: utf-8 -*-
"""
Tabela no formato da Tabela III de Ottonello e Winberry (2020).

Resposta: Delta log k(t+1), em pontos percentuais. Para comparacao direta com o artigo,
esta tabela adota a normalizacao de O-W: choque em pontos percentuais, positivo = CORTE
de juros. (O restante do trabalho usa positivo = aperto, em desvios-padrao.)
Heterogeneidade em desvios-padrao: alavancagem desviada da media da firma e padronizada
na amostra (exatamente como O-W); quantil de markup padronizado na amostra.

Colunas:
  (1) alavancagem, sem controles                  EF firma + setor x trimestre
  (2) alavancagem, com controles Z de O-W
  (3) markup, com controles
  (4) alavancagem e markup
  (5) alavancagem e markup, sem EF setor x trimestre: EF firma + setor x trimestre do
      ano e efeito medio do choque (O-W incluem ainda defasagens de PIB, inflacao e
      desemprego; aqui omitidas, versao inicial)
  (6) alavancagem, painel ampliado sem energia (empresas sem markup estimado)
  (7) alavancagem, painel ampliado com energia
Nas colunas (6) e (7) o controle de crescimento das vendas sai, porque a receita so
existe para as empresas do pipeline de markups.

Inferencia: maior dos tres erros-padrao (firma e trimestre; firma; trimestre).

Uso:
    python3 07_tabela_ow.py [painel_regressao.csv] [painel_replicacao.csv] [dir_saida]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import pyfixest as pf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"data"/"processed"/"painel_regressao.csv"
PA = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT/"data"/"processed"/"painel_replicacao.csv"
SAIDA = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT/"output"
TAB = SAIDA/"tables"; TAB.mkdir(parents=True, exist_ok=True)
SD_BP = 9.43                         # dp do fator de nivel trimestral, em pontos-base

def prepara(d, s_bp):
    d = d.copy()
    d["empresa"] = d.empresa.astype(str); d["cel"] = d.cel.astype(str)
    d["s_ow"] = -s_bp/100.0                                # p.p., positivo = corte
    d["dlnk"] = 100*d.dlnk_ow
    d["i_lev"] = d.s_ow*d.lev_dm_l1
    d["i_corte"] = d.s_ow.clip(lower=0)*d.lev_dm_l1      # so cortes
    d["i_aperto"] = d.s_ow.clip(upper=0)*d.lev_dm_l1     # so apertos
    d["qoy"] = d.t % 4 + 1
    return d

R = prepara(pd.read_csv(PR).dropna(subset=["mu_q_l1"]), pd.read_csv(PR).dropna(subset=["mu_q_l1"]).s_nivel*SD_BP)
R["setor"] = R.nome_setor
R["mu_z"] = (R.mu_q_l1 - R.mu_q_l1.mean())/R.mu_q_l1.std()
R["i_mu"] = R.s_ow*R.mu_z
R["setor_q"] = R.setor + "_" + R.qoy.astype(str)
A = pd.read_csv(PA); A = prepara(A, A.s_nivel_bp)

def roda(f, d, termos):
    import re
    usadas = [v for v in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", f)) if v in d.columns]
    d = d.dropna(subset=usadas)
    m = pf.feols(f, data=d, vcov={"CRV1": "empresa+t"})
    t2 = m.tidy().copy(); tf = m.vcov({"CRV1": "empresa"}).tidy().copy(); tt = m.vcov({"CRV1": "t"}).tidy().copy()
    G = min(d.empresa.nunique(), d.t.nunique()); out = {}
    for k in termos:
        c = t2.loc[k, "Estimate"]; ep = np.nanmax([t2.loc[k, "Std. Error"], tf.loc[k, "Std. Error"], tt.loc[k, "Std. Error"]])
        out[k] = (c, ep, 2*stats.t.sf(abs(c)/ep, df=G-1))
    return out, m._N, m._r2, d.empresa.nunique()

Z  = "lev_dm_l1 + dlnr_l1 + lnat_l1 + ac_at_l1"
Za = "lev_dm_l1 + lnat_l1 + ac_at_l1"
COLS = [
    ("(1)", "dlnk ~ i_lev | empresa + cel", R, ["i_lev"], "não", "sim"),
    ("(2)", "dlnk ~ i_lev + %s | empresa + cel" % Z, R, ["i_lev"], "sim", "sim"),
    ("(3)", "dlnk ~ i_mu + mu_z + %s | empresa + cel" % Z, R, ["i_mu"], "sim", "sim"),
    ("(4)", "dlnk ~ i_lev + i_mu + mu_z + %s | empresa + cel" % Z, R, ["i_lev", "i_mu"], "sim", "sim"),
    ("(5)", "dlnk ~ i_lev + i_mu + s_ow + mu_z + %s | empresa + setor_q" % Z, R, ["i_lev", "i_mu", "s_ow"], "sim", "não"),
    ("(6)", "dlnk ~ i_lev + %s | empresa + cel" % Za, A[~A.energia], ["i_lev"], "sim", "sim"),
    ("(7)", "dlnk ~ i_lev + %s | empresa + cel" % Za, A, ["i_lev"], "sim", "sim"),
]
res = {}
for nome, f, d, termos, ctrl, fe in COLS:
    res[nome] = roda(f, d, termos) + (ctrl, fe)

# assimetria: o proprio O-W reporta que a heterogeneidade vem sobretudo dos cortes
assim = {}
for nome, f, d in [("painel de markups", "dlnk ~ i_corte + i_aperto + %s | empresa + cel" % Z, R),
                   ("ampliado sem energia", "dlnk ~ i_corte + i_aperto + %s | empresa + cel" % Za, A[~A.energia]),
                   ("ampliado com energia", "dlnk ~ i_corte + i_aperto + %s | empresa + cel" % Za, A)]:
    assim[nome] = roda(f, d, ["i_corte", "i_aperto"])

# ------------------------------------------------------------------ saida legivel
def cel(t):
    c, ep, p = t
    return "%.2f%s" % (c, "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""), "(%.2f)" % ep

linhas = []
for rot, k in [("alavancagem × choque", "i_lev"), ("markup × choque", "i_mu"), ("choque", "s_ow")]:
    a, b = [rot], [""]
    for nome in res:
        if k in res[nome][0]: x, y = cel(res[nome][0][k]); a.append(x); b.append(y)
        else: a.append(""); b.append("")
    a.append({"i_lev": "−0,57", "i_mu": "", "s_ow": ""}[k]); b.append({"i_lev": "(0,27)", "i_mu": "", "s_ow": ""}[k])
    linhas += [a, b]
linhas.append(["observações"] + ["{:,}".format(res[n][1]).replace(",", ".") for n in res] + ["219.402"])
linhas.append(["empresas"] + [str(res[n][3]) for n in res] + [""])
linhas.append(["R²"] + ["%.3f" % res[n][2] for n in res] + ["0,124"])
linhas.append(["controles de firma"] + [res[n][4] for n in res] + ["sim"])
linhas.append(["EF setor × trimestre"] + [res[n][5] for n in res] + ["sim"])
T = pd.DataFrame(linhas, columns=[""] + list(res) + ["O-W (2)"])
T.to_csv(TAB/"tabela_ow_formato_III.csv", index=False)

with open(TAB/"tabela_ow_formato_III.txt", "w", encoding="utf-8") as fh:
    fh.write(T.to_string(index=False) + "\n\nASSIMETRIA -- alavancagem x choque, separando cortes e apertos (coluna 2/6/7)\n")
    for nome, (o, n, r2, ne) in assim.items():
        fh.write("  %-22s cortes %s %s | apertos %s %s | N=%d\n" % ((nome,) + cel(o["i_corte"]) + cel(o["i_aperto"]) + (n,)))
print(open(TAB/"tabela_ow_formato_III.txt", encoding="utf-8").read())

# ------------------------------------------------------------------ LaTeX
def tex(s): return s.replace("×", "$\\times$").replace("−", "$-$").replace("R²", "$R^2$")
with open(TAB/"tabela_ow_formato_III.tex", "w", encoding="utf-8") as fh:
    fh.write("\\begin{tabular}{l" + "c"*len(res) + "|c}\n\\toprule\n")
    fh.write(" & " + " & ".join(res) + " & O--W (2) \\\\\n\\midrule\n")
    for i, row in T.iterrows():
        if row.iloc[0] == "observações": fh.write("\\midrule\n")
        fh.write(tex(" & ".join(str(v).replace(".", "{,}") if v and v[0].isdigit() or (v and v[0] in "-−") else str(v) for v in row)) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
print("tex:", TAB/"tabela_ow_formato_III.tex")
