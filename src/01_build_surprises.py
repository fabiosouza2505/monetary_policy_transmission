"""
Bloco B — surpresas monetarias do Copom pela taxa a termo entre reunioes.

Entradas
  data/interim/eventos_copom.csv  198 reunioes com t-1, t0, t1 e os prazos em dias
                                  uteis (n0..n4) ate as vigencias v0..v4, medidos
                                  de cada uma das tres datas de observacao
  data/raw/pre_terms.csv          curva DI x PRE interpolada (flat-forward, base 252)
                                  nos prazos exigidos por cada data

Saidas
  data/processed/surpresas_copom.csv
  data/processed/cargas_fatores.csv

Construcao
  Para a data de observacao d, com fator de desconto D_d(n) = (1 + r_d(n)/100)^(-n/252),
  a taxa a termo do k-esimo intervalo entre reunioes e

      f_d^(k) = [ D_d(n_{k-1}) / D_d(n_k) ]^( 252 / (n_k - n_{k-1}) ) - 1

  As datas de vigencia v_0..v_4 sao mantidas fixas em calendario entre t0 e t1; os
  prazos n_k sao recalculados a partir de cada data de observacao. A diferenca mede,
  portanto, revisao de expectativa sobre um intervalo fixo, e nao deslocamento de
  horizonte. O spread estacionario CDI-Selic se cancela na diferenca.

  Choque de nivel:        s_t = f_{t1}^(1) - f_{t0}^(1)
  Janela de dois dias:          f_{t1}^(k) - f_{tm1}^(k)
  Fatores:                dois componentes principais de [df^(1) .. df^(4)] com a
                          rotacao ortogonal de Gurkaynak-Sack-Swanson (2005) que
                          anula a carga do segundo fator sobre df^(1).
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, INT, PROC = ROOT/"data"/"raw", ROOT/"data"/"interim", ROOT/"data"/"processed"
PROC.mkdir(parents=True, exist_ok=True)
KS = [1, 2, 3, 4]


def disc(rate, n):
    """Fator de desconto na base 252. n = 0 => D = 1 (vigencia ja comecou)."""
    n = np.asarray(n, float)
    rate = np.asarray(rate, float)
    return np.where(n <= 0, 1.0, (1.0 + rate/100.0) ** (-n/252.0))


def forwards(row, base, terms):
    """Taxas a termo f^(1..4) a partir de uma data de observacao, em % a.a."""
    n = np.array([row[f"n{k}_{base}"] for k in range(5)], float)
    r = np.array([terms[f"r{k}"] for k in range(5)], float)
    D = disc(r, n)
    out = {}
    for k in KS:
        dn = n[k] - n[k-1]
        out[k] = np.nan if dn <= 0 else ((D[k-1]/D[k]) ** (252.0/dn) - 1.0) * 100.0
    return out


def gss_rotate(X, cols, short=0, longe=3):
    """
    Dois componentes principais de X (T x 4, em bp) com rotacao de GSS.
    Retorna (F, cargas, variancia explicada). A rotacao anula a carga do fator 2
    sobre a coluna `short`. Normalizacao: carga unitaria do fator 1 sobre `short`
    e do fator 2 sobre `longe`.
    """
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var_ratio = (S**2) / (S**2).sum()
    T = Xc.shape[0]
    Z = U[:, :2] * np.sqrt(T)
    Lam = (np.diag(S[:2]) @ Vt[:2, :]) / np.sqrt(T)

    a, b = Lam[0, short], Lam[1, short]
    nrm = np.hypot(a, b)
    R = np.array([[a/nrm, -b/nrm], [b/nrm, a/nrm]])
    F, Lt = Z @ R, R.T @ Lam                      # Lt[1, short] == 0

    s1 = Lt[0, short]; F[:, 0] *= s1; Lt[0, :] /= s1
    s2 = Lt[1, longe]; F[:, 1] *= s2; Lt[1, :] /= s2
    return F, pd.DataFrame(Lt, index=["fator_nivel", "fator_trajetoria"], columns=cols), var_ratio


def main():
    ev = pd.read_csv(INT/"eventos_copom.csv", parse_dates=["data_reuniao", "tm1", "t0", "t1"])
    pre = pd.read_csv(RAW/"pre_terms.csv", parse_dates=["date"]).set_index("date")

    rows = []
    for _, r in ev.iterrows():
        rec = {c: r[c] for c in ["reuniao", "data_reuniao", "meta_selic", "vies",
                                 "extraordinaria", "tm1", "t0", "t1"]}
        ok = True
        for base in ["tm1", "t0", "t1"]:
            d = r[base]
            if d not in pre.index:
                ok = False
                break
            f = forwards(r, base, pre.loc[d])
            for k in KS:
                rec[f"f{k}_{base}"] = f[k]
        rec["completo"] = ok
        rows.append(rec)

    d = pd.DataFrame(rows)
    for k in KS:                                   # variacoes em pontos-base
        d[f"df{k}"]  = (d[f"f{k}_t1"] - d[f"f{k}_t0"])  * 100.0
        d[f"df{k}_2d"] = (d[f"f{k}_t1"] - d[f"f{k}_tm1"]) * 100.0

    d["surpresa_nivel"] = d["df1"]                 # choque de baseline
    d["d_selic"] = d["meta_selic"].diff()

    cols = [f"df{k}" for k in KS]
    use = d[cols].notna().all(axis=1) & d["completo"]
    F, load, vr = gss_rotate(d.loc[use, cols].to_numpy(float), cols)
    d.loc[use, "fator_nivel"] = F[:, 0]
    d.loc[use, "fator_trajetoria"] = F[:, 1]

    d.to_csv(PROC/"surpresas_copom.csv", index=False)
    load.to_csv(PROC/"cargas_fatores.csv")

    print(f"reunioes com surpresa completa: {int(use.sum())} de {len(d)}")
    print(f"amostra: {d.data_reuniao.min().date()} a {d.data_reuniao.max().date()}")
    print(f"variancia explicada pelos dois primeiros CPs: {vr[0]:.3f}, {vr[1]:.3f} "
          f"(soma {vr[:2].sum():.3f})")
    print("\ncargas (bp de variacao por unidade do fator):")
    print(load.round(3).to_string())
    print("\nestatisticas (bp):")
    print(d[["surpresa_nivel", "fator_nivel", "fator_trajetoria"] + cols]
          .describe().round(2).to_string())
    falt = d.loc[~use, "data_reuniao"].dt.strftime("%Y-%m-%d").tolist()
    if falt:
        print(f"\nsem curva completa ({len(falt)}): {', '.join(falt[:12])}")


if __name__ == "__main__":
    main()
