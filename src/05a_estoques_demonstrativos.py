# -*- coding: utf-8 -*-
"""
Variaveis de estoque das demonstracoes financeiras Economatica (bloco Nao Consolidado).

Extrai, por empresa e trimestre: ativo total, imobilizado, financiamentos e debentures de
curto e longo prazo, ativo circulante e passivo circulante. Sao todas variaveis de
balanco -- estoques no fim do trimestre --, de modo que nao ha acumulacao a desfazer.

Por que estoques:
  - Ottonello e Winberry (2020) medem investimento como Delta log k, com k o valor
    contabil do capital tangivel. O `Imobilizado` e esse objeto.
  - A alavancagem deles e (divida CP + divida LP) / ativo total.
  Nenhuma das duas depende do CAPEX, que so existe de 2008 em diante.

O layout dos arquivos varia: em parte deles as datas estao na linha 0, em outros na linha
3, abaixo de um bloco de titulo. O script localiza o cabecalho do bloco pelo rotulo.

Cache por arquivo em data/interim/estoques/, de modo que uma execucao interrompida
retoma de onde parou. Opcao --max-segundos limita o tempo de uma execucao.

Uso:
    python3 05a_estoques_demonstrativos.py [dir_financeiro] [--max-segundos N]
"""
import sys, time, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
def _posicionais(argv):
    out, pula = [], False
    for a in argv:
        if pula: pula = False; continue
        if a == "--max-segundos": pula = True; continue
        if not a.startswith("--"): out.append(a)
    return out
args = _posicionais(sys.argv[1:])
FIN  = Path(args[0]) if args else (ROOT.parent/"Projeto Markup"/"GMM - ACF"/
                                   "markup-estimation"/"data"/"raw"/"financeiro")
CACHE = ROOT/"data"/"interim"/"estoques"; CACHE.mkdir(parents=True, exist_ok=True)
OUT   = ROOT/"data"/"interim"/"estoques_demonstrativos.csv"
TMAX  = float(sys.argv[sys.argv.index("--max-segundos")+1]) if "--max-segundos" in sys.argv else 1e12

ROT = {"Ativo total":"ativo", "Imobilizado":"imob",
       "Financiamento CP":"fin_cp", "Financiamento LP":"fin_lp",
       "Debentures CP":"deb_cp", "Debentures LP":"deb_lp",
       "Ativo Circulante":"ac", "Passivo Circulante":"pc"}
CABEC = ("Não Consolidado", "Consolidado")

def extrai(f):
    d = pd.read_excel(f, sheet_name=0, header=None)
    lab = d.iloc[:,0].astype(str).str.strip().str.lstrip("+-=").str.strip()
    cab = lab[lab.isin(CABEC)]
    if not len(cab): return None
    # prefere o bloco Nao Consolidado, que casa com o painel de markups
    nc = cab[cab == "Não Consolidado"]
    ini = (nc if len(nc) else cab).index[0]
    prox = [i for i in cab.index if i > ini]
    fim = prox[0] if prox else len(d)
    datas = pd.to_datetime(d.iloc[ini, 1:], errors="coerce")
    rec = {"data": datas.values}
    bl = lab.iloc[ini:fim]
    for r, v in ROT.items():
        m = bl[bl == r]
        rec[v] = (pd.to_numeric(d.iloc[m.index[0], 1:], errors="coerce").values
                  if len(m) else np.nan)
    x = pd.DataFrame(rec).dropna(subset=["data"])
    x["empresa"] = Path(f).stem
    x["bloco"] = lab.iloc[ini]
    return x

t0 = time.time(); arqs = sorted(glob.glob(str(FIN/"*.xlsx"))); feitos = novos = falhas = 0
for f in arqs:
    c = CACHE/(Path(f).stem + ".csv")
    if c.exists(): feitos += 1; continue
    if time.time() - t0 > TMAX: break
    try:
        x = extrai(f)
        (x if x is not None else pd.DataFrame()).to_csv(c, index=False)
        novos += 1
    except Exception as e:
        falhas += 1; print("falha", Path(f).name, str(e)[:80])

pend = sum(1 for f in arqs if not (CACHE/(Path(f).stem + ".csv")).exists())
print("arquivos: %d | em cache antes: %d | novos: %d | falhas: %d | pendentes: %d"
      % (len(arqs), feitos, novos, falhas, pend))
if not arqs:
    sys.exit("nenhum .xlsx em %s" % FIN)
if pend == 0:
    partes = [pd.read_csv(c) for c in sorted(CACHE.glob("*.csv")) if c.stat().st_size > 5]
    e = pd.concat(partes, ignore_index=True)
    e["data"] = pd.to_datetime(e.data)
    e["ano"] = e.data.dt.year; e["trimestre"] = e.data.dt.quarter
    e.to_csv(OUT, index=False)
    print("saida: %s | %d linhas, %d empresas, blocos %s"
          % (OUT.name, len(e), e.empresa.nunique(), e.bloco.value_counts().to_dict()))
