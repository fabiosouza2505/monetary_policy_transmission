"""
Bloco B — protocolo de validacao das surpresas monetarias (desenho de taxa a termo).

Saidas
  output/tables/bloco_b_checagens.csv
  output/tables/bloco_b_episodios.csv
  output/figures/fig_fatores.png
  output/figures/fig_cargas.png
  output/figures/fig_nivel_vs_decisao.png
  output/figures/fig_forward_vs_meta.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT = Path(__file__).resolve().parents[1]
PROC, RAW = ROOT/"data"/"processed", ROOT/"data"/"raw"
TAB, FIG = ROOT/"output"/"tables", ROOT/"output"/"figures"
for p in (TAB, FIG): p.mkdir(parents=True, exist_ok=True)
KS = [1, 2, 3, 4]
res = []


def add(bloco, chk, valor, crit, ver, obs=""):
    res.append(dict(bloco=bloco, checagem=chk, valor=valor, criterio=crit, veredito=ver, obs=obs))


def main():
    df = pd.read_csv(PROC/"surpresas_copom.csv", parse_dates=["data_reuniao","tm1","t0","t1"])
    load = pd.read_csv(PROC/"cargas_fatores.csv", index_col=0)
    d = df.dropna(subset=["fator_nivel"]).reset_index(drop=True)

    # ---- B.0 a taxa a termo mede o que deveria -----------------------------------
    # f1 cobre exatamente o intervalo governado pela meta decidida nesta reuniao.
    # Em t1 a decisao e conhecida: f1_t1 deve seguir a meta quase um para um, com
    # a diferenca igual ao spread estacionario CDI-Selic.
    r = np.corrcoef(d.f1_t1, d.meta_selic)[0,1]
    spread = (d.f1_t1 - d.meta_selic)
    b = np.polyfit(d.meta_selic, d.f1_t1, 1)[0]
    add("B.0 construcao", "corr(f1 em t+1, meta decidida)", f"{r:.4f}",
        "proxima de 1 — a termo cobre o intervalo da decisao",
        "OK" if r > 0.99 else "ATENCAO", f"inclinacao={b:.3f}")
    add("B.0 construcao", "spread CDI-Selic implicito", f"{spread.mean()*100:.1f} bp (dp={spread.std()*100:.1f})",
        "pequeno, negativo e estavel",
        "OK" if -60 < spread.mean()*100 < 0 and spread.std()*100 < 40 else "ATENCAO")
    r0 = np.corrcoef(d.f1_t0, d.meta_selic)[0,1]
    add("B.0 construcao", "corr(f1 em t0, meta decidida)", f"{r0:.4f}",
        "alta mas inferior a de t+1 — parte da decisao e surpresa",
        "OK" if r0 < r else "ATENCAO")

    # ---- B.1 media -----------------------------------------------------------------
    for c in ["surpresa_nivel", "fator_nivel", "fator_trajetoria"]:
        x = d[c].to_numpy(); t, p = stats.ttest_1samp(x, 0.0)
        add("B.1 media", c, f"media={x.mean():.2f} bp (dp={x.std(ddof=1):.1f})",
            "media estatisticamente nula", "OK" if p > .10 else "ATENCAO",
            f"t={t:.2f}, p={p:.3f}")

    # ---- B.2 autocorrelacao --------------------------------------------------------
    for c in ["fator_nivel", "fator_trajetoria"]:
        x = pd.Series(d[c].to_numpy())
        p = float(acorr_ljungbox(x, lags=[4], return_df=True)["lb_pvalue"].iloc[0])
        add("B.2 autocorrelacao", c, f"rho(1)={x.autocorr(1):.3f}", "ausente",
            "OK" if p > .10 else "ATENCAO", f"Ljung-Box(4) p={p:.3f}")

    # ---- B.3 decisoes antecipadas --------------------------------------------------
    d["manutencao"] = (d.d_selic.abs() < 1e-9).astype(int)
    man, mov = d.loc[d.manutencao==1,"fator_nivel"].abs(), d.loc[d.manutencao==0,"fator_nivel"].abs()
    t, p = stats.ttest_ind(man, mov, equal_var=False)
    add("B.3 antecipacao", "|fator_nivel|: manutencao vs mudanca",
        f"{man.mean():.1f} vs {mov.mean():.1f} bp",
        "menor quando a meta e mantida", "OK" if man.mean() < mov.mean() and p < .10 else "ATENCAO",
        f"t={t:.2f}, p={p:.3f}, n={len(man)}/{len(mov)}")
    rr = np.corrcoef(d.fator_nivel, d.d_selic.fillna(0))[0,1]
    add("B.3 antecipacao", "corr(fator_nivel, variacao da meta)", f"{rr:.3f}",
        "positiva e longe de 1", "OK" if .05 < rr < .85 else "ATENCAO")

    # ---- B.4 trajetoria alta com nivel baixo ---------------------------------------
    lo, hi = d.fator_nivel.abs().quantile(.33), d.fator_trajetoria.abs().quantile(.90)
    ep = d[(d.fator_nivel.abs() <= lo) & (d.fator_trajetoria.abs() >= hi)]
    add("B.4 trajetoria", "episodios de comunicacao", f"{len(ep)} reunioes",
        "existencia de episodios", "OK" if len(ep) >= 3 else "ATENCAO",
        "; ".join(ep.data_reuniao.dt.strftime("%Y-%m-%d")))
    ep[["reuniao","data_reuniao","meta_selic","d_selic","fator_nivel","fator_trajetoria"]] \
        .to_csv(TAB/"bloco_b_episodios.csv", index=False)

    # ---- B.5 precos de ativos ------------------------------------------------------
    f = RAW/"ativos_janela.csv"
    if f.exists():
        a = pd.read_csv(f, parse_dates=["data_reuniao"])
        m = d.merge(a, on="data_reuniao", how="inner")
        for c, esp in [("ret_ibov","negativo"), ("ret_usdbrl","negativo")]:
            if c not in m: continue
            rr = np.corrcoef(m.fator_nivel, m[c])[0,1]
            ok = rr < 0 if esp == "negativo" else rr > 0
            add("B.5 ativos", f"corr(fator_nivel, {c})", f"{rr:.3f}",
                f"sinal esperado: {esp}",
                "OK" if ok else "ATENCAO — sintoma de efeito informacional", f"n={len(m)}")
    else:
        add("B.5 ativos", "correlacao com cambio e Ibovespa", "—", "sinais esperados",
            "PENDENTE", "arquivo data/raw/ativos_janela.csv ausente")

    # ---- B.6 sensibilidade a janela ------------------------------------------------
    sub = d.dropna(subset=[f"df{k}_2d" for k in KS])
    rr = np.corrcoef(sub.df1, sub.df1_2d)[0,1]
    add("B.6 janela", "corr(janela 1 dia, janela 2 dias) em f1",
        f"{rr:.3f} (dp relativo {sub.df1_2d.std()/sub.df1.std():.2f})",
        "estabilidade entre janelas", "OK" if rr > .70 else "ATENCAO", f"n={len(sub)}")

    out = pd.DataFrame(res); out.to_csv(TAB/"bloco_b_checagens.csv", index=False)
    print(out.to_string(index=False))

    # ---- figuras --------------------------------------------------------------------
    fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
    for a, c, cor, lab in [(ax[0],"fator_nivel","#2b6cb0","nivel (bp)"),
                           (ax[1],"fator_trajetoria","#c05621","trajetoria (bp)")]:
        a.axhline(0, lw=.8, color="k"); a.bar(d.data_reuniao, d[c], width=25, color=cor)
        a.set_ylabel(lab)
    ax[0].set_title("Surpresas do Copom — taxa a termo entre reunioes")
    fig.tight_layout(); fig.savefig(FIG/"fig_fatores.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,4))
    for i, row in load.iterrows():
        ax.plot(KS, row.values, marker="o", label=i)
    ax.axhline(0, lw=.8, color="k"); ax.set_xticks(KS)
    ax.set_xlabel("intervalo entre reunioes (k)"); ax.set_ylabel("bp por unidade do fator")
    ax.set_title("Cargas dos fatores"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG/"fig_cargas.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(d.d_selic*100, d.fator_nivel, s=18, alpha=.7)
    ax.axhline(0, lw=.8, color="k"); ax.axvline(0, lw=.8, color="k")
    ax.set_xlabel("variacao da meta Selic (bp)"); ax.set_ylabel("surpresa de nivel (bp)")
    ax.set_title("Parte antecipada versus surpresa")
    fig.tight_layout(); fig.savefig(FIG/"fig_nivel_vs_decisao.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(d.meta_selic, d.f1_t1, s=18, alpha=.7, label="t+1")
    ax.scatter(d.meta_selic, d.f1_t0, s=14, alpha=.5, label="t0")
    lim = [d.meta_selic.min()-1, d.meta_selic.max()+1]
    ax.plot(lim, lim, lw=.8, color="k", ls="--")
    ax.set_xlabel("meta Selic decidida (% a.a.)"); ax.set_ylabel("taxa a termo entre reunioes (% a.a.)")
    ax.set_title("Validacao da construcao (B.0)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG/"fig_forward_vs_meta.png", dpi=150); plt.close(fig)
    print("\nfiguras em output/figures/")


if __name__ == "__main__":
    main()
