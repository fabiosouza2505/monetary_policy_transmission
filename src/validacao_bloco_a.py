# -*- coding: utf-8 -*-
"""
Bloco A do protocolo de validacao empirica — estimacao de markups.

Submete a medida de markup as checagens previstas na secao 3 da nota metodologica,
na amostra que entra no paper de transmissao: 1994-2024 (pos Plano Real), setor de
energia eletrica excluido.

Escopo deste script (decisao de 18/11/2026):
  A.0  procedencia — confirma que o run carregado e a medida CORRIGIDA (DLW)
  A.1  nivel dos markups
  A.2  elasticidades e retornos de escala, com IC bootstrap
  A.5  estabilidade temporal
  A.3 (margem contabil) e A.4 (robustez a tecnologia) foram transferidas ao
  Projeto Markup; aparecem na tabela como FORA DO ESCOPO para que o protocolo
  permaneca completo e a transferencia fique registrada.

Uso:
    python3 validacao_bloco_a.py [caminho_para_results/python-gmm-acf/<run>]
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT/"output"/"tables", ROOT/"output"/"figures"
for p in (TAB, FIG): p.mkdir(parents=True, exist_ok=True)

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT.parent/"Projeto Markup"/"GMM - ACF"/"markup-estimation"/
    "results"/"python-gmm-acf"/"baseline")
if not RUN.exists():
    sys.exit("run nao encontrado: %s\nPasse o caminho como argumento." % RUN)

ANO_INI, ANO_FIM = 1994, 2024          # pos Plano Real
SETOR_FORA = "Energia Elétrica"        # precos regulados nas duas pontas
LIM_NIVEL  = 2/3                       # "maioria substancial" com mu > 1
LIM_SALTO  = 4.0                       # desvios-padrao na 1a diferenca
MAX_SALTOS = 2                         # saltos tolerados por setor
MAX_IC     = 0.30                      # largura maxima do IC de retornos de escala
MIN_FIRMAS = 12                        # firmas por celula setor x trimestre
DROP_ULT   = True                      # descarta o ultimo trimestre, incompleto

res = []
def add(bloco, chk, val, crit, vered, obs=""):
    res.append({"bloco": bloco, "checagem": chk, "valor": val,
                "criterio": crit, "veredito": vered, "obs": obs})
ok = lambda c: "OK" if c else "ATENCAO"

# ---------------------------------------------------------------- A.0 procedencia
cfg = json.loads((RUN/"config.json").read_text(encoding="utf-8"))
corr = bool(cfg.get("markup_correction", False))
add("A.0 procedencia", "medida em uso",
    "corrigida (DLW)" if corr else "NAO corrigida",
    "corrigida: alpha^V depurada por exp(eps)",
    ok(corr),
    "decisao de 18/11/2026; se virar False, o regressor muda e as tabelas do "
    "Apendice A perdem validade")

d = pd.read_csv(RUN/"markups_empresa.csv",
                usecols=["empresa","ano","trimestre","nome_setor","receita",
                         "theta_v","theta_k","share_v","share_v_corr","eps",
                         "markup","log_v","log_k","motivo_exclusao"])
bruto = len(d)
d = d[(d.nome_setor != SETOR_FORA) & d.ano.between(ANO_INI, ANO_FIM)]
d = d[d.markup.notna()]
SET = sorted(d.nome_setor.unique())

add("A.0 procedencia", "amostra",
    "%d obs, %d empresas, %d setores" % (len(d), d.empresa.nunique(), len(SET)),
    "%d-%d, energia excluida" % (ANO_INI, ANO_FIM),
    ok(len(SET) == 4 and d.ano.min() >= ANO_INI),
    "%d obs no arquivo; %d removidas por setor, ano ou exclusao" % (bruto, bruto-len(d)))

# o ultimo trimestre da extracao costuma vir incompleto e imita saida de firmas
cont = (d.groupby(["ano","trimestre"]).empresa.nunique()
          .rename("n").reset_index().sort_values(["ano","trimestre"]))
ult, ref = cont.n.iloc[-1], cont.n.iloc[-5:-1].median()
add("A.0 procedencia", "completude do ultimo trimestre",
    "%dQ%d com %d firmas" % (cont.ano.iloc[-1], cont.trimestre.iloc[-1], ult),
    ">= 70%% da mediana dos 4 anteriores (%.0f)" % ref, ok(ult >= .7*ref),
    "queda acentuada indica extracao parcial, nao saida; trimestre deve ser descartado")

if DROP_ULT and ult < .7*ref:
    d = d[~((d.ano == cont.ano.iloc[-1]) & (d.trimestre == cont.trimestre.iloc[-1]))]
    add("A.0 procedencia", "amostra apos descarte",
        "%d obs, %d empresas" % (len(d), d.empresa.nunique()),
        "trimestre incompleto removido", "OK",
        "ultimo trimestre retido: %dQ%d" % (d.ano.max(),
            d.loc[d.ano == d.ano.max(), "trimestre"].max()))

# confere que a coluna 'markup' e mesmo theta_v/share_v_corr
dif = (d.markup - d.theta_v/d.share_v_corr).abs().max()
add("A.0 procedencia", "identidade mu = theta_v / alpha^V_corr", "%.2e" % dif,
    "diferenca numerica desprezivel", ok(dif < 1e-9))

# ---------------------------------------------------------------- A.1 nivel
for s in SET:
    x = d.loc[d.nome_setor == s, "markup"]
    frac = (x > 1).mean()
    add("A.1 nivel", s[:46], "%.1f%% com mu > 1" % (100*frac),
        "maioria substancial (>= %.0f%%)" % (100*LIM_NIVEL), ok(frac >= LIM_NIVEL),
        "mediana %.3f, p25 %.3f, p75 %.3f" % (x.median(), x.quantile(.25), x.quantile(.75)))

# ---------------------------------------------------------------- A.2 elasticidades
par = pd.read_csv(RUN/"translog_parametros.csv").set_index("setor")
bs  = pd.read_csv(RUN/"bootstrap_replicacoes.csv")

def soma_theta(b, vbar, kbar):
    """theta_v + theta_k avaliados nas medias amostrais do setor."""
    return (b["beta_v"] + b["beta_k"] + 2*b["beta_vv"]*vbar
            + 2*b["beta_kk"]*kbar + b["beta_vk"]*(vbar + kbar))

for s in SET:
    sub = d[d.nome_setor == s]
    vbar, kbar = sub.log_v.mean(), sub.log_k.mean()
    p = par.loc[s]
    pt = soma_theta(p, vbar, kbar)
    rep = bs[bs.setor == s]
    if len(rep):
        dist = soma_theta(rep, vbar, kbar)
        lo, hi = np.percentile(dist, [2.5, 97.5])
        dentro, larg = lo <= 1.0 <= hi, hi - lo
        add("A.2 elasticidades", "retornos de escala — %s" % s[:34],
            "%.3f" % pt, "1 dentro do IC95%% [%.3f, %.3f]" % (lo, hi), ok(dentro),
            "ep bootstrap %.3f, n=%d" % (dist.std(), len(rep)))
        # um IC largo faz o teste acima passar por falta de poder, nao por acerto
        add("A.2 elasticidades", "precisao do teste — %s" % s[:37],
            "largura do IC = %.3f" % larg, "< %.2f para o teste ter conteudo" % MAX_IC,
            ok(larg < MAX_IC),
            "IC largo aprova retornos de escala sem conseguir rejeitar nada")
    else:
        add("A.2 elasticidades", "retornos de escala — %s" % s[:34],
            "%.3f" % pt, "sem bootstrap disponivel", "ATENCAO")
    # ressalva de degenerescencia: capital sem contribuicao
    tk = sub.theta_k.mean()
    add("A.2 elasticidades", "contribuicao do capital — %s" % s[:32],
        "theta_k medio = %.3f" % tk, "theta_k >= 0,05", ok(tk >= .05),
        "theta_v medio = %.3f; com theta_k ~ 0 a tecnologia pouco acrescenta a parcela, e o que distingue a medida e a correcao por exp(eps)" % sub.theta_v.mean())

# ---------------------------------------------------------------- A.5 estabilidade
serie = (d.groupby(["nome_setor","ano","trimestre"])
           .agg(mu=("markup","median"), n=("empresa","nunique"))
           .reset_index().sort_values(["nome_setor","ano","trimestre"]))
serie["t"] = pd.PeriodIndex(year=serie.ano, quarter=serie.trimestre, freq="Q")

saltos_todos = []
for s in SET:
    x = serie[serie.nome_setor == s].copy()
    x["dmu"] = x.mu.diff()
    x["dn"]  = x.n.diff()
    sd = x.dmu.std()
    x["z"] = x.dmu/sd
    sal = x[x.z.abs() > LIM_SALTO]
    saltos_todos.append(sal.assign(setor=s))
    add("A.5 estabilidade", "saltos — %s" % s[:44], "%d" % len(sal),
        "no maximo %d (|z| > %.0f)" % (MAX_SALTOS, LIM_SALTO), ok(len(sal) <= MAX_SALTOS),
        "; ".join("%s (n=%d)" % (a, b) for a, b in zip(sal.t.astype(str), sal.n))
        if len(sal) else "dp da 1a diferenca = %.4f" % sd)

    # artefato de decumulacao: o pipeline converte fluxos acumulados no ano fiscal
    q1, outros = x[x.trimestre == 1].dmu.dropna(), x[x.trimestre != 1].dmu.dropna()
    if len(q1) > 2 and len(outros) > 2:
        t = (q1.mean() - outros.mean())/np.sqrt(q1.var()/len(q1) + outros.var()/len(outros))
        add("A.5 estabilidade", "efeito 1o trimestre — %s" % s[:36],
            "t = %+.2f" % t, "|t| < 2 (sem artefato de decumulacao)", ok(abs(t) < 2),
            "media da 1a dif: Q1 %+.4f vs demais %+.4f" % (q1.mean(), outros.mean()))

    # composicao: saltos que acompanham entrada/saida de firmas
    xx = x.dropna(subset=["dmu","dn"])
    if len(xx) > 10:
        r = np.corrcoef(xx.dmu, xx.dn)[0,1]
        add("A.5 estabilidade", "composicao do painel — %s" % s[:34],
            "corr = %+.3f" % r, "|corr| < 0,30", ok(abs(r) < .30),
            "variacao do markup mediano contra variacao do numero de firmas")

# densidade das celulas: o quantil intrassetorial exige firmas suficientes
for s in SET:
    x = serie[serie.nome_setor == s]
    rasas = (x.n < MIN_FIRMAS).mean()
    add("A.5 estabilidade", "celulas rasas — %s" % s[:40],
        "%.1f%% com < %d firmas" % (100*rasas, MIN_FIRMAS),
        "menos de 20% das celulas", ok(rasas < .20),
        "mediana %d firmas/celula; min %d em %s" %
        (x.n.median(), x.n.min(), x.loc[x.n.idxmin(), "t"]))

# custo de impor densidade minima: o quantil intrassetorial precisa de firmas
cel = serie.set_index(["nome_setor","ano","trimestre"]).n
obs = d.groupby(["nome_setor","ano","trimestre"]).size()
perde = obs[cel.reindex(obs.index) < MIN_FIRMAS].groupby(level=0).sum().reindex(SET, fill_value=0)
for s in SET:
    x = serie[serie.nome_setor == s]
    viavel = x[x.n >= MIN_FIRMAS]
    ini = viavel.t.min() if len(viavel) else None
    add("A.5 estabilidade", "custo da regra de %d firmas — %s" % (MIN_FIRMAS, s[:26]),
        "%d obs (%.1f%% do setor)" % (perde[s], 100*perde[s]/len(d[d.nome_setor == s])),
        "regra aplicada na amostra de regressao",
        "INFORMATIVO", "primeiro trimestre viavel: %s" % ini)

if saltos_todos:
    pd.concat(saltos_todos)[["setor","t","mu","dmu","z","n"]].to_csv(
        TAB/"bloco_a_saltos.csv", index=False)

# ---------------------------------------------------------------- fora do escopo
add("A.3 margem", "correlacao com margem contabil", "—",
    "requer CPV e EBITDA isolados no pipeline", "FORA DO ESCOPO",
    "transferida ao Projeto Markup em 18/11/2026; criterio precisa ser reformulado, "
    "pois a medida nao corrigida passaria trivialmente")
add("A.4 tecnologia", "ordenacao sob Cobb-Douglas", "—",
    "requer estimacao CD alternativa", "FORA DO ESCOPO",
    "transferida ao Projeto Markup em 18/11/2026")

# ---------------------------------------------------------------- saida
out = pd.DataFrame(res)
out.to_csv(TAB/"bloco_a_checagens.csv", index=False)
serie.drop(columns="t").to_csv(TAB/"bloco_a_serie_setorial.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 48)
print(out[["bloco","checagem","valor","criterio","veredito"]].to_string(index=False))
print()
v = out.veredito.value_counts().to_dict()
print("vereditos:", v, "| run:", RUN.name, "| amostra %d-%d" % (ANO_INI, ANO_FIM))

fig, ax = plt.subplots(figsize=(10, 5))
for s in SET:
    x = serie[serie.nome_setor == s]
    ax.plot(x.t.dt.to_timestamp(), x.mu, lw=1.3, label=s[:38])
ax.axhline(1, lw=.8, color="k", ls="--")
ax.set_ylabel("markup mediano"); ax.legend(fontsize=7)
ax.set_title("Markup mediano por setor — medida corrigida, %d-%d" % (ANO_INI, ANO_FIM))
fig.tight_layout(); fig.savefig(FIG/"fig_bloco_a_series.png", dpi=150); plt.close(fig)
print("tabela: output/tables/bloco_a_checagens.csv | figura: output/figures/fig_bloco_a_series.png")
