# -*- coding: utf-8 -*-
"""
Agregacao trimestral das surpresas do Copom — equacao (2) de Ottonello-Winberry (2020).

Cada choque datado reparte-se entre o trimestre em que ocorre e o seguinte:
    w_a = (tau_n - tau_d)/tau_n   fracao do trimestre que resta apos o choque
    w_b =  tau_d         /tau_n   fracao ja decorrida, que transborda para o seguinte
Os pesos somam 1 por choque: nenhum choque e criado nem destruido.

Baseline em dias de calendario, fiel a O-W; variante em dias uteis (ANBIMA) como
sensibilidade.

A ponderacao induz correlacao serial por construcao, pois cada choque aparece em dois
trimestres adjacentes. A checagem de rho(1) e feita contra a distribuicao nula gerada
por permutacao dos choques entre as datas observadas, e nao contra um limiar arbitrario.
"""
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.home()/"mnt"/"Macro"
PROC, TAB, FIG = ROOT/"data"/"processed", ROOT/"output"/"tables", ROOT/"output"/"figures"
for p in (TAB, FIG): p.mkdir(parents=True, exist_ok=True)

SERIES = ["surpresa_nivel", "fator_nivel", "fator_trajetoria"]
NSIM, SEED = 5000, 20260918

try:
    from bizdays import Calendar
    CAL, TEM_BIZ = Calendar.load("ANBIMA"), True
except Exception as e:
    print("aviso: calendario ANBIMA indisponivel (%s)" % e); TEM_BIZ = False

d = pd.read_csv(PROC/"surpresas_copom.csv", parse_dates=["data_reuniao"])
q = d.data_reuniao.dt.to_period("Q")
QS = pd.period_range(q.min(), q.max()+1, freq="Q")
IA = pd.Index(QS).get_indexer(q.values)
IB = pd.Index(QS).get_indexer(q.values + 1)

def pesos(modo):
    ini, fim = q.dt.start_time, q.dt.end_time.dt.normalize()
    if modo == "calendario":
        tn = (fim - ini).dt.days + 1
        td = (d.data_reuniao - ini).dt.days + 1
    else:
        f = lambda a, b: np.array([CAL.bizdays(x.date(), y.date()) for x, y in zip(a, b)]) + 1
        tn, td = pd.Series(f(ini, fim)), pd.Series(f(ini, d.data_reuniao))
    return ((tn - td)/tn).values, (td/tn).values, tn.values, td.values

def soma_trim(x, wa, wb):
    o = np.zeros(len(QS))
    np.add.at(o, IA, x*wa); np.add.at(o, IB, x*wb)
    return o

def ac1(v):
    v = np.asarray(v, float); v = v - v.mean()
    return float((v[:-1]*v[1:]).sum()/(v*v).sum())

WA, WB, TN, TD = pesos("calendario")
nucleo = slice(1, len(QS)-1)          # pontas incompletas: 1o sem carry-in, ultimo sem reunioes

out = pd.DataFrame(index=QS); out.index.name = "trimestre"
for s in SERIES:
    out[s] = soma_trim(d[s].values, WA, WB)
    out[s+"_soma"] = pd.Series(d[s].values).groupby(q.values).sum().reindex(QS, fill_value=0.).values
out["n_reunioes"] = q.value_counts().reindex(QS, fill_value=0).astype(int).values
out["completo"] = True; out.iloc[0, out.columns.get_loc("completo")] = False
out.iloc[-1, out.columns.get_loc("completo")] = False
out.to_csv(PROC/"surpresas_trimestrais.csv")
v = out[out.completo]

res = []
def add(chk, val, crit, vered, obs=""):
    res.append({"checagem": chk, "valor": val, "criterio": crit, "veredito": vered, "obs": obs})
ok = lambda c: "OK" if c else "ATENCAO"

# ---- construcao -----------------------------------------------------------------
sw = WA + WB
add("soma dos pesos por choque", "min=%.6f max=%.6f" % (sw.min(), sw.max()),
    "identicamente 1", ok(np.allclose(sw, 1.0)), "n=%d" % len(sw))
add("intervalo de w_a", "[%.3f, %.3f]" % (WA.min(), WA.max()), "dentro de [0,1]",
    ok(((WA >= 0) & (WA <= 1)).all()), "w_a=0 no ultimo dia do trimestre")
add("conservacao do choque total", "%.4f vs %.4f bp" % (out.fator_nivel.sum(), d.fator_nivel.sum()),
    "somas iguais", ok(np.isclose(out.fator_nivel.sum(), d.fator_nivel.sum())))
vaz = int((v.n_reunioes == 0).sum())
add("trimestres sem reuniao", "%d" % vaz, "esperado 0", ok(vaz == 0),
    "%d trimestres uteis, %d reunioes" % (len(v), int(out.n_reunioes.sum())))

# ---- propriedades ---------------------------------------------------------------
rng = np.random.default_rng(SEED)
for s in ["fator_nivel", "fator_trajetoria"]:
    x = v[s]
    add("media de %s" % s, "%.2f bp (dp=%.2f)" % (x.mean(), x.std()), "media ~ 0",
        ok(abs(x.mean()) < .5*x.std()))
    obs = ac1(x.values)
    sim = np.array([ac1(soma_trim(rng.permutation(d[s].values), WA, WB)[nucleo]) for _ in range(NSIM)])
    lo, md, hi = np.percentile(sim, [2.5, 50, 97.5])
    dentro = lo <= obs <= hi
    add("rho(1) de %s" % s, "%+.3f" % obs,
        "nulo mecanico: mediana %+.3f, IC95%% [%+.3f, %+.3f]" % (md, lo, hi),
        "OK" if dentro else "RESSALVA",
        "soma ingenua %+.3f; excesso sobre a mediana %+.3f" % (ac1(v[s+"_soma"].values), obs-md))

r = np.corrcoef(v.fator_nivel, v.fator_trajetoria)[0,1]
add("ortogonalidade apos agregacao", "%+.4f" % r, "|corr| < 0,05", ok(abs(r) < .05),
    "era %+.4f no nivel do evento" % np.corrcoef(d.fator_nivel, d.fator_trajetoria)[0,1])
rs = np.corrcoef(v.fator_nivel, v.fator_nivel_soma)[0,1]
add("corr(ponderado, soma ingenua)", "%.3f" % rs, "alta mas < 1", ok(.6 < rs < .999),
    "dp %.2f vs %.2f bp" % (v.fator_nivel.std(), v.fator_nivel_soma.std()))

if TEM_BIZ:
    wa2, wb2, _, _ = pesos("uteis")
    b = pd.DataFrame({s: soma_trim(d[s].values, wa2, wb2) for s in SERIES}, index=QS)
    b["n_reunioes"] = out.n_reunioes.values; b["completo"] = out.completo.values
    b.to_csv(PROC/"surpresas_trimestrais_diasuteis.csv")
    m = b[b.completo]
    rb = np.corrcoef(v.fator_nivel, m.fator_nivel)[0,1]
    add("corr(calendario, dias uteis)", "%.4f" % rb, "> 0,99", ok(rb > .99),
        "max |dif| = %.2f bp" % (v.fator_nivel - m.fator_nivel).abs().max())

tb = pd.DataFrame(res); tb.to_csv(TAB/"agregacao_checagens.csv", index=False)
pd.set_option("display.width", 230); pd.set_option("display.max_colwidth", 52)
print(tb.to_string(index=False)); print()
print("trimestres %s a %s | %d uteis | reunioes/trimestre: mediana %.0f, max %d"
      % (QS[0], QS[-1], len(v), v.n_reunioes.median(), v.n_reunioes.max()))
print(v[["fator_nivel","fator_trajetoria"]].describe().T[["mean","std","min","max"]].round(2).to_string())
print()
print("ACF trimestral do fator de nivel (lags 1-6):",
      [round(v.fator_nivel.autocorr(l), 3) for l in range(1, 7)])

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
t = v.index.to_timestamp()
for a, s, cor, lab in [(ax[0], "fator_nivel", "#2b6cb0", "nivel (bp)"),
                       (ax[1], "fator_trajetoria", "#c05621", "trajetoria (bp)")]:
    a.axhline(0, lw=.8, color="k"); a.bar(t, v[s], width=70, color=cor); a.set_ylabel(lab)
ax[0].set_title("Surpresas do Copom agregadas ao trimestre — ponderacao de dias restantes")
fig.tight_layout(); fig.savefig(FIG/"fig_trimestral.png", dpi=150); plt.close(fig)
print("\nserie: data/processed/surpresas_trimestrais.csv | figura: output/figures/fig_trimestral.png")
